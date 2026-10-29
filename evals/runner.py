"""Offline (Track A) case runner.

    python -m evals.runner                       # every case, every available implementation
    python -m evals.runner --impl loop-ref --cases E01,E02 --report reports/loop-ref.json

For each case the runner first runs legacy-replay (the golden baseline), then the
implementation under test against fresh fake ports, then evaluates the case's checks.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from fakes.context import FakeBundle, PortsSpec
from harness.events import HarnessEvent
from harness.types import ResumeInput, ResumeState, RunRequest
from impls.legacy_replay import LegacyReplay
from impls.reference_loop import ReferenceLoop

from .case import Case, agent_spec, load_cases


# ---------------------------------------------------------------- implementations


def _pydantic_ai():
    from impls.pydantic_ai_slice import PydanticAISlice

    return PydanticAISlice()


def _openai_agents():
    from impls.openai_agents_slice import OpenAIAgentsSlice

    return OpenAIAgentsSlice()


IMPLS: dict[str, Callable[[Case], Any]] = {
    "legacy-replay": lambda case: LegacyReplay(case.legacy, agent_spec(case.agent).ref),
    "loop-ref": lambda case: ReferenceLoop(),
    "pydantic-ai": lambda case: _pydantic_ai(),
    "openai-agents": lambda case: _openai_agents(),
}


def available_impls() -> list[str]:
    out = []
    for name, make in IMPLS.items():
        try:
            if name in ("pydantic-ai", "openai-agents"):
                make(None)  # type: ignore[arg-type]
            out.append(name)
        except ImportError:
            pass
    return out


# ---------------------------------------------------------------- driving runs


@dataclass
class RunRecord:
    request: RunRequest
    events: list[HarnessEvent]
    disconnected_at: int | None = None
    seen_before_disconnect: list[int] = field(default_factory=list)
    seen_after_reattach: list[int] = field(default_factory=list)

    @property
    def finished(self) -> HarnessEvent | None:
        return next((e for e in reversed(self.events) if e.type == "run.finished"), None)

    @property
    def outcome(self) -> str | None:
        f = self.finished
        return f.payload.outcome if f else None


@dataclass
class CaseRun:
    case: Case
    impl: str
    bundle: FakeBundle
    runs: list[RunRecord]
    baseline: "CaseRun | None" = None
    harness_obj: Any = None

    @property
    def all_events(self) -> list[HarnessEvent]:
        return [e for r in self.runs for e in r.events]


async def drive(harness: Any, req: RunRequest, bundle: FakeBundle, disconnect_after: int | None = None) -> RunRecord:
    handle = harness.start(req, bundle.context())
    rec = RunRecord(req, [])
    if disconnect_after is None:
        async for ev in handle.events():
            rec.events.append(ev)
    else:
        reader = handle.events()
        async for ev in reader:  # the reader leaves mid-reply (the SSE response is closed) ...
            rec.events.append(ev)
            rec.seen_before_disconnect.append(ev.seq)
            if ev.seq >= disconnect_after:
                break
        await reader.aclose()
        rec.disconnected_at = rec.events[-1].seq if rec.events else 0
        await asyncio.wait_for(handle.wait(), timeout=10)  # ... the run must finish without a reader
        async for ev in handle.events(after_seq=rec.disconnected_at):  # re-attach with Last-Event-ID
            rec.events.append(ev)
            rec.seen_after_reattach.append(ev.seq)
    await asyncio.wait_for(handle.wait(), timeout=10)
    return rec


async def execute(case: Case, impl: str, *, ports: PortsSpec | None = None, run_prefix: str = "run",
                  bundle: FakeBundle | None = None, with_continuations: bool = True) -> CaseRun:
    bundle = bundle or FakeBundle(ports or case.ports)
    harness = IMPLS[impl](case)
    runs: list[RunRecord] = []
    req = case.request(f"{run_prefix}_{case.id}_1")
    runs.append(await drive(harness, req, bundle, case.client.disconnect_after_seq))
    last_inputs: list[ResumeInput] = []
    for n, cont in enumerate(case.continuations if with_continuations else [], start=2):
        prev = runs[-1]
        ids = prev.finished.payload.interrupt_ids if prev.finished else []
        if cont.stale:
            inputs = last_inputs
        else:
            if not ids:
                break
            inputs = [ResumeInput(interrupt_id=i, status=a.status, payload={"approved": a.approved}, actor="user:1")
                      for i, a in zip(ids, cont.answers)]
        cp = await bundle.checkpoints.load_latest(prev.request.run_id)
        state = ResumeState(impl=cp.impl, format_version=cp.format_version, step_id=cp.step_id, blob=cp.state) if cp else None
        req = case.request(f"{run_prefix}_{case.id}_{n}", continues_run_id=prev.request.run_id,
                           resume_state=state, resume_inputs=inputs)
        runs.append(await drive(harness, req, bundle))
        last_inputs = inputs
    return CaseRun(case, impl, bundle, runs, harness_obj=harness)


# ---------------------------------------------------------------- evaluating


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    impl: str
    checks: list[CheckResult]
    metrics: dict[str, Any]
    skipped: bool = False

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"case": self.case_id, "impl": self.impl, "passed": self.passed, "skipped": self.skipped,
                "metrics": self.metrics,
                "checks": [c.__dict__ for c in self.checks]}


async def evaluate(case: Case, impl: str) -> CaseResult:
    from .assertions import CHECKS, metrics_of

    caps = getattr(IMPLS[impl](case), "capabilities", None)
    if caps is not None and case.scenario not in caps.scenarios:
        return CaseResult(case.id, impl, [], {}, skipped=True)
    baseline = await execute(case, "legacy-replay", run_prefix="legacy") if case.legacy else None
    cr = baseline if impl == "legacy-replay" and baseline else await execute(case, impl)
    cr.baseline = baseline
    results: list[CheckResult] = []
    for spec in case.checks + ["port_expectations"]:
        name, params = (spec, {}) if isinstance(spec, str) else next(iter(spec.items()))
        params = params if isinstance(params, dict) else {"value": params}
        fn = CHECKS.get(name)
        if fn is None:
            results.append(CheckResult(name, False, "unknown check"))
            continue
        if impl == "legacy-replay" and getattr(fn, "needs_ports", False):
            continue  # port-traffic checks are meaningless on a replayed stream
        try:
            ok, detail = await fn(cr, **params)
        except Exception as exc:  # a crashing check is a failing check, with the reason
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(CheckResult(name, ok, detail))
    return CaseResult(case.id, impl, results, metrics_of(cr))


async def evaluate_all(impls: list[str], case_ids: list[str] | None = None) -> list[CaseResult]:
    out = []
    for case in load_cases(case_ids):
        if "A" not in case.tracks:
            continue
        for impl in impls:
            out.append(await evaluate(case, impl))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", default=",".join(["legacy-replay", "loop-ref"]))
    ap.add_argument("--cases", default="")
    ap.add_argument("--report", default="")
    a = ap.parse_args(argv)
    avail = available_impls()
    impls = avail if a.impl == "all" else a.impl.split(",")
    missing = [i for i in impls if i not in avail]
    if missing:
        print(f"not installed, skipped: {missing} (uv sync --extra slices)")
        impls = [i for i in impls if i in avail]
    results = asyncio.run(evaluate_all(impls, [c for c in a.cases.split(",") if c] or None))
    width = max(len(r.case_id) for r in results) if results else 4
    for r in results:
        failed = [c for c in r.checks if not c.ok]
        if r.skipped:
            print(f"{r.case_id:<{width}}  {r.impl:<14} SKIP  (scenario not in the implementation's capabilities)")
            continue
        print(f"{r.case_id:<{width}}  {r.impl:<14} {'PASS' if r.passed else 'FAIL'}  "
              f"ttft={r.metrics.get('ttft_ms')}ms model_calls={r.metrics.get('model_calls')}"
              + "".join(f"\n      x {c.name}: {c.detail}" for c in failed))
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
