"""legacy-replay: stands in for Victor's LegacyAssistHarness offline.

It replays a golden legacy SSE stream (recorded on staging) through the legacy -> harness
mapping, with the recorded timing. Every eval case is first calibrated on this
implementation: a case that fails on legacy is a broken case, not a regression.
"""
from __future__ import annotations

from typing import Any

from harness.mapping.legacy_sse import LegacyMapper
from harness.ports import RunContext
from harness.runtime import QueueRunHandle
from harness.types import Capabilities, RunRequest

Frame = tuple[int, str, dict[str, Any]]  # (t_ms since request, sse event, data)


class LegacyReplay:
    name = "legacy-replay"
    version = "0.1.0"
    capabilities = Capabilities(scenarios=["home", "assist", "delegate", "oneshot"], relay=True, delegation=True)

    def __init__(self, streams: list[list[Frame]], author: str):
        self.streams = streams  # one recorded stream per run (first run, then continuations)
        self.author = author
        self._i = 0

    def start(self, req: RunRequest, ctx: RunContext) -> QueueRunHandle:
        frames = self.streams[min(self._i, len(self.streams) - 1)] if self.streams else []
        self._i += 1
        handle = QueueRunHandle(req, ctx)

        async def body(h: QueueRunHandle) -> None:
            m = LegacyMapper(self.name, self.version, req.scenario, self.author)
            t0 = ctx.clock.now_ms()
            for t_ms, event, data in frames:
                wait = t0 + t_ms - ctx.clock.now_ms()
                if wait > 0:
                    await ctx.clock.sleep(wait)
                for type_, payload in m.feed(event, data):
                    if type_ == "run.started":
                        payload = {**payload, "continues_run_id": req.continues_run_id}
                    await h.emitter.emit(type_, by_agent=self.author, **payload)
                    if h.emitter.finished:
                        return

        return handle.launch(body)
