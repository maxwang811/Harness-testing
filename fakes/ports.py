"""Fake ToolExecutor, MemoryPort, ThreadReader, CheckpointStore, BudgetGate, Tracer and EventSink.
Each one records what it was asked, so eval checks can assert on port traffic."""
from __future__ import annotations

import contextlib
import io
import zipfile
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from harness.errors import BudgetDenied, ErrorCode, ErrorInfo
from harness.events import HarnessEvent
from harness.ports import (ChatMessage, Checkpoint, MemoryHit, MemoryResult, Reservation, ToolOutcome, ToolSpec,
                           Usage)

from .clock import FakeClock


# ---------------------------------------------------------------- tools


class ToolFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    side_effects: bool = False
    requires_approval: bool = False
    latency_ms: int = 60
    results: list[dict[str, Any]] = [{"ok": True, "content": "ok", "summary": "ok"}]  # by call order; last repeats
    artifact: dict[str, Any] | None = None  # {"kind": "pptx", "title": ...} -> a real (minimal) file is stored


def minimal_pptx() -> bytes:
    """Smallest zip that passes the structural PPTX check used by E03 (not a renderable deck)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("ppt/presentation.xml", "<p:presentation/>")
        z.writestr("ppt/slides/slide1.xml", "<p:sld/>")
    return buf.getvalue()


class FakeToolExecutor:
    def __init__(self, fixtures: dict[str, ToolFixture], clock: FakeClock, mode: Literal["normal", "replay"] = "normal",
                 ledger: dict[str, ToolOutcome] | None = None):
        self.fixtures = fixtures
        self.clock = clock
        self.mode = mode
        self.ledger: dict[str, ToolOutcome] = ledger if ledger is not None else {}  # idempotency results table
        self.side_effect_runs: Counter[str] = Counter()
        self.calls: list[dict[str, Any]] = []
        self.files: dict[str, bytes] = {}
        self._n: Counter[str] = Counter()

    def list(self, scopes: list[str] | None = None) -> list[ToolSpec]:
        return [ToolSpec(name=n, description=f.description, parameters=f.parameters,
                         requires_approval=f.requires_approval, side_effects=f.side_effects)
                for n, f in self.fixtures.items()]

    async def call(self, name: str, args: dict[str, Any], idempotency_key: str) -> ToolOutcome:
        self.calls.append({"name": name, "args": args, "key": idempotency_key, "t": self.clock.now_ms()})
        if idempotency_key in self.ledger:
            return self.ledger[idempotency_key].model_copy(update={"replayed": True})
        fx = self.fixtures.get(name)
        if fx is None:
            return ToolOutcome(ok=False, summary=f"unknown tool {name}",
                               error=ErrorInfo.of(ErrorCode.TOOL_DENIED, f"{name} not in scope"))
        await self.clock.sleep(fx.latency_ms)
        i = self._n[name]
        self._n[name] += 1
        raw = dict(fx.results[min(i, len(fx.results) - 1)])
        if raw.get("error"):
            raw["error"] = ErrorInfo.of(ErrorCode(raw["error"]), raw.get("summary", ""))
        if self.mode == "replay":
            raw["replayed"] = True  # shadow: recorded result, no side effect
        elif fx.side_effects:
            self.side_effect_runs[name] += 1
        if fx.artifact and raw.get("ok", True):
            ref = f"files/{name}_{i + 1}.{fx.artifact.get('kind', 'bin')}"
            if self.mode != "replay":
                self.files[ref] = minimal_pptx() if fx.artifact.get("kind") == "pptx" else b"artifact"
            raw.setdefault("artifacts", [{"artifact_id": f"art_{name}_{i + 1}", "kind": fx.artifact.get("kind", "bin"),
                                          "title": fx.artifact.get("title", name), "ref": ref}])
            raw.setdefault("artifact_refs", [ref])
        out = ToolOutcome(**raw)
        self.ledger[idempotency_key] = out
        return out


# ---------------------------------------------------------------- memory and threads


class FakeMemory:
    def __init__(self, clock: FakeClock, hits: list[dict[str, Any]] | None = None, degraded: bool = False,
                 searchable_scopes: list[str] | None = None, context: str = ""):
        self.clock = clock
        self.hits = [MemoryHit(**h) for h in (hits or [])]
        self.degraded = degraded
        self.searchable = searchable_scopes  # None = every requested scope is searchable
        self.context = context
        self.searches: list[dict[str, Any]] = []

    async def search(self, query: str, scopes: list[str], limit: int = 8) -> MemoryResult:
        await self.clock.sleep(150 if self.degraded else 120)
        searched = [s for s in scopes if self.searchable is None or s in self.searchable]
        self.searches.append({"query": query, "scopes": scopes, "searched": searched})
        hits = [h for h in self.hits if h.scope in searched][:limit]
        return MemoryResult(hits=hits, scopes_searched=searched, degraded=["embeddings"] if self.degraded else [])

    async def project_context(self, thread_id: str) -> str:
        return self.context

    async def write(self, scope: str, text: str, source: str) -> str:
        return f"mem_{len(self.hits) + 1}"


class FakeThreads:
    def __init__(self, history: list[dict[str, Any]] | None = None):
        self._history = [ChatMessage(**m) for m in (history or [])]

    async def history(self, thread_id: str, limit: int = 40) -> list[ChatMessage]:
        return self._history[-limit:]


# ---------------------------------------------------------------- checkpoints, budget, tracing, events


class FakeCheckpoints:
    def __init__(self):
        self.saved: list[Checkpoint] = []

    async def save(self, run_id: str, step_id: str, state: dict[str, Any], version: int, impl: str) -> None:
        self.saved.append(Checkpoint(run_id=run_id, step_id=step_id, format_version=version, impl=impl, state=state))

    async def load_latest(self, run_id: str) -> Checkpoint | None:
        for cp in reversed(self.saved):
            if cp.run_id == run_id:
                return cp
        return None


class FakeBudget:
    def __init__(self, deny: bool = False, shadow: bool = False):
        self.deny = deny
        self.shadow = shadow
        self.admitted: list[Reservation] = []
        self.settled: list[Usage] = []
        self._n = 0

    async def admit(self, estimate: float, agent_ref: str) -> Reservation:
        if self.deny:
            raise BudgetDenied(ErrorInfo.of(ErrorCode.BUDGET_DENIED, "workspace out of credit"))
        self._n += 1
        r = Reservation(reservation_id=f"res_{self._n}", estimate=estimate)
        self.admitted.append(r)
        return r

    async def settle(self, reservation: Reservation, usage: Usage) -> None:
        if not self.shadow:  # shadow runs bill nothing (E13)
            if usage.usage_id not in {u.usage_id for u in self.settled}:
                self.settled.append(usage)


class FakeTracer:
    def __init__(self):
        self.spans: list[tuple[str, dict[str, Any]]] = []

    @contextlib.contextmanager
    def span(self, name: str, attrs: dict[str, Any] | None = None):
        self.spans.append((name, dict(attrs or {})))
        yield None


class ListSink:
    def __init__(self):
        self.events: list[HarnessEvent] = []

    async def append(self, event: HarnessEvent) -> None:
        self.events.append(event)
