"""Shared runtime every implementation can reuse: the event emitter and a queue-backed RunHandle.

Using this keeps the hard-to-get-right semantics (seq, write-before-yield, re-attach,
readers never cancelling) in one place instead of once per framework adapter.
"""
from __future__ import annotations

import asyncio
import datetime as _dt
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

from .errors import ErrorCode, ErrorInfo, NotSupported
from .events import HarnessEvent, make_payload
from .ports import RunContext
from .types import ResumeInput, RunRequest

_EPOCH = _dt.datetime(2026, 1, 1, tzinfo=_dt.timezone.utc)


def iso_from_ms(ms: int) -> str:
    """Virtual clocks count from 0; real clocks pass epoch ms. Both render as ISO-8601 UTC."""
    base = _EPOCH if ms < 10**11 else _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
    t = base + _dt.timedelta(milliseconds=ms)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


class Emitter:
    def __init__(self, run_id: str, ctx: RunContext, on_event: Callable[[HarnessEvent], None]):
        self.run_id = run_id
        self.ctx = ctx
        self._seq = 0
        self._on_event = on_event
        self.finished = False

    async def emit(self, type_: str, *, parent_id: str | None = None, by_agent: str | None = None,
                   **fields: Any) -> HarnessEvent:
        if self.finished:
            raise RuntimeError(f"event {type_} after run.finished")
        self._seq += 1
        ev = HarnessEvent(
            type=type_,
            run_id=self.run_id,
            seq=self._seq,
            ts=iso_from_ms(self.ctx.clock.now_ms()),
            parent_id=parent_id,
            agent=by_agent,
            payload=make_payload(type_, **fields),
        )
        await self.ctx.events.append(ev)  # write first ...
        self._on_event(ev)  # ... then fan out
        if type_ == "run.finished":
            self.finished = True
        return ev

    @property
    def last_seq(self) -> int:
        return self._seq


class QueueRunHandle:
    """RunHandle over an in-memory buffer. Production swaps the buffer for run_events + Redis,
    but the reader semantics are the same and are what the eval cases check."""

    def __init__(self, req: RunRequest, ctx: RunContext):
        self.run_id = req.run_id
        self.req = req
        self.ctx = ctx
        self._buf: list[HarnessEvent] = []
        self._cond = asyncio.Condition()
        self._done = asyncio.Event()
        self.cancel_requested: Literal["now", "after_step"] | None = None
        self.emitter = Emitter(req.run_id, ctx, self._push)
        self._task: asyncio.Task | None = None

    # -- producer side
    def _push(self, ev: HarnessEvent) -> None:
        self._buf.append(ev)
        if ev.type == "run.finished":
            self._done.set()
        asyncio.get_running_loop().create_task(self._notify())

    async def _notify(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    def launch(self, body: Callable[["QueueRunHandle"], Awaitable[None]]) -> "QueueRunHandle":
        async def guarded() -> None:
            try:
                await body(self)
            except asyncio.CancelledError:
                if not self.emitter.finished:
                    await self.emitter.emit("run.finished", outcome="cancelled",
                                            error=ErrorInfo.of(ErrorCode.CANCELLED, "cancel(now)"))
            except Exception as exc:  # last-resort guard: a bug must still end the run cleanly
                if not self.emitter.finished:
                    await self.emitter.emit("run.finished", outcome="failed",
                                            error=ErrorInfo.of(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}"))
                raise
            finally:
                if not self.emitter.finished:
                    await self.emitter.emit("run.finished", outcome="failed",
                                            error=ErrorInfo.of(ErrorCode.INTERNAL, "implementation returned without run.finished"))

        self._task = asyncio.get_running_loop().create_task(guarded())
        return self

    # -- reader side
    async def events(self, after_seq: int = 0) -> AsyncIterator[HarnessEvent]:
        i = after_seq  # seq is gap-free and starts at 1, so the buffer index of seq n is n-1
        while True:
            while i < len(self._buf):
                yield self._buf[i]
                i += 1
            if self._done.is_set() and i >= len(self._buf):
                return
            async with self._cond:
                if i >= len(self._buf) and not self._done.is_set():
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=0.05)
                    except asyncio.TimeoutError:
                        pass

    async def submit(self, inputs: list[ResumeInput]) -> None:
        raise NotSupported("v1: answers to interrupts start a continuation run")

    async def cancel(self, mode: Literal["now", "after_step"] = "after_step") -> None:
        self.cancel_requested = mode
        if mode == "now" and self._task is not None:
            self._task.cancel()

    async def wait(self) -> None:
        await self._done.wait()
        if self._task is not None:
            try:
                await self._task
            except Exception:
                pass

    @property
    def buffered(self) -> list[HarnessEvent]:
        return list(self._buf)
