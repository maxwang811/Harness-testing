"""Reusable, machine-checkable assertions. Each check: async (CaseRun, **params) -> (ok, detail).
Checks marked needs_ports inspect fake-port traffic and are skipped on legacy-replay."""
from __future__ import annotations

import datetime as _dt
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable

import jsonschema

from harness.events import HarnessEvent

from .runner import CaseRun, RunRecord, execute

CHECKS: dict[str, Callable[..., Awaitable[tuple[bool, str]]]] = {}
_SCHEMA = json.loads((Path(__file__).resolve().parent.parent / "schemas" / "harness_event.v1.json").read_text())
_VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA)


def check(name: str, needs_ports: bool = False):
    def deco(fn):
        fn.needs_ports = needs_ports
        CHECKS[name] = fn
        return fn
    return deco


def ms(ts: str) -> int:
    t = _dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=_dt.timezone.utc)
    return int(t.timestamp() * 1000)


def of(events: list[HarnessEvent], *types: str) -> list[HarnessEvent]:
    return [e for e in events if e.type in types]


def ttft_ms(run: RunRecord) -> int | None:
    start = next((e for e in run.events if e.type == "run.started"), None)
    first = next((e for e in run.events if e.type == "message.delta"), None)
    return ms(first.ts) - ms(start.ts) if start and first else None


def observable(cr: CaseRun) -> list[tuple]:
    """What a user can see or what gets saved: the comparison projection between implementations."""
    out: list[tuple] = []
    for r in cr.runs:
        tools = {e.payload.call_id: e.payload.tool for e in of(r.events, "tool.called")}
        for e in r.events:
            p = e.payload
            if e.type == "message.completed":
                out.append(("message", p.author, p.text))
            elif e.type == "tool.result":
                out.append(("tool", tools.get(p.call_id, "?"), p.ok))
            elif e.type == "handoff":
                out.append(("handoff", p.from_agent, p.to_agent))
            elif e.type == "interrupt.requested":
                out.append(("interrupt", p.kind, tools.get(p.call_id) or p.reason))
            elif e.type == "artifact.created":
                out.append(("artifact", p.kind))
            elif e.type == "run.finished":
                out.append(("outcome", p.outcome, p.error.code.value if p.error else None))
    return out


def metrics_of(cr: CaseRun) -> dict[str, Any]:
    first = cr.runs[0]
    usage = of(cr.all_events, "usage")
    calls = getattr(cr.bundle.models, "calls", [])
    return {
        "ttft_ms": ttft_ms(first),
        "events": len(cr.all_events),
        "runs": len(cr.runs),
        "model_calls": len(calls) if cr.impl != "legacy-replay" else None,
        "cost_cny": round(sum(e.payload.cost for e in usage), 6),
        "outcomes": [r.outcome for r in cr.runs],
    }


# ---------------------------------------------------------------- contract-level


def grammar_errors(run: RunRecord) -> list[str]:
    ev, errs = run.events, []
    if not ev:
        return ["no events"]
    seqs = [e.seq for e in ev]
    if seqs != list(range(1, len(ev) + 1)):
        errs.append(f"seq not 1..n gap-free: {seqs[:12]}...")
    if ev[0].type != "run.started":
        errs.append(f"first event is {ev[0].type}")
    fin = of(ev, "run.finished")
    if len(fin) != 1 or ev[-1].type != "run.finished":
        errs.append(f"{len(fin)} run.finished events; last event {ev[-1].type}")
    if len(of(ev, "run.started")) != 1:
        errs.append("run.started not exactly once")
    open_steps: set[str] = set()
    msgs: dict[str, dict[str, Any]] = {}
    called: Counter[str] = Counter()
    results: Counter[str] = Counter()
    subs: set[str] = set()
    interrupts: list[str] = []
    pending_calls: set[str] = set()
    for e in ev:
        p = e.payload
        if e.type == "step.started":
            if p.step_id in open_steps:
                errs.append(f"step {p.step_id} started twice")
            open_steps.add(p.step_id)
        elif e.type == "step.finished":
            if p.step_id not in open_steps:
                errs.append(f"step {p.step_id} finished but not open")
            open_steps.discard(p.step_id)
        elif e.type == "message.started":
            if p.message_id in msgs:
                errs.append(f"message_id {p.message_id} reused")
            msgs[p.message_id] = {"text": "", "done": False}
        elif e.type == "message.delta":
            m = msgs.get(p.message_id)
            if m is None:
                errs.append(f"delta before message.started ({p.message_id})")
            else:
                m["text"] += p.text
        elif e.type == "message.completed":
            m = msgs.get(p.message_id)
            if m is None:
                errs.append(f"completed without started ({p.message_id})")
            elif m["done"]:
                errs.append(f"message {p.message_id} completed twice")
            else:
                m["done"] = True
                if m["text"] != p.text:
                    errs.append(f"message {p.message_id}: deltas != completed text")
        elif e.type == "tool.called":
            called[p.call_id] += 1
        elif e.type == "tool.result":
            results[p.call_id] += 1
            if not called[p.call_id]:
                errs.append(f"tool.result without tool.called ({p.call_id})")
        elif e.type == "subagent.started":
            subs.add(p.subagent_run_id)
        elif e.type == "subagent.finished":
            if p.subagent_run_id not in subs:
                errs.append(f"subagent {p.subagent_run_id} finished but not started")
            subs.discard(p.subagent_run_id)
        elif e.type == "interrupt.requested":
            interrupts.append(p.interrupt_id)
            if p.call_id:
                pending_calls.add(p.call_id)
        elif e.type == "run.finished":
            if open_steps:
                errs.append(f"steps still open at run.finished: {sorted(open_steps)}")
            if subs:
                errs.append(f"subagents still open: {sorted(subs)}")
            if (p.outcome == "interrupted") != bool(interrupts):
                errs.append(f"outcome {p.outcome} with {len(interrupts)} interrupts")
            if p.outcome == "interrupted" and sorted(p.interrupt_ids) != sorted(interrupts):
                errs.append("run.finished.interrupt_ids != interrupts requested")
            if p.outcome == "failed" and p.error is None:
                errs.append("failed run without error{code,retryable}")
    for cid, n in called.items():
        if n != 1:
            errs.append(f"tool call {cid} called {n} times")
        if results[cid] != 1 and cid not in pending_calls:
            errs.append(f"tool call {cid} has {results[cid]} results (want exactly 1)")
    for mid, m in msgs.items():
        if not m["done"]:
            errs.append(f"message {mid} never completed")
    usage_ids = [e.payload.usage_id for e in of(ev, "usage")]
    if len(usage_ids) != len(set(usage_ids)):
        errs.append("duplicate usage_id")
    return errs


@check("grammar")
async def _grammar(cr: CaseRun) -> tuple[bool, str]:
    errs = [f"run {i + 1}: {m}" for i, r in enumerate(cr.runs) for m in grammar_errors(r)]
    return not errs, "; ".join(errs[:6])


@check("schema_valid")
async def _schema_valid(cr: CaseRun) -> tuple[bool, str]:
    bad = []
    for e in cr.all_events:
        errors = list(_VALIDATOR.iter_errors(e.to_json_dict()))
        if errors:
            bad.append(f"{e.type}#{e.seq}: {errors[0].message[:120]}")
    return not bad, "; ".join(bad[:4])


@check("port_expectations", needs_ports=True)
async def _port_expectations(cr: CaseRun) -> tuple[bool, str]:
    f = getattr(cr.bundle.models, "expect_failures", [])
    return not f, "; ".join(f[:4])


# ---------------------------------------------------------------- comparison with legacy


@check("observable_matches_legacy")
async def _observable(cr: CaseRun) -> tuple[bool, str]:
    if cr.baseline is None:
        return False, "case has no legacy stream"
    a, b = observable(cr.baseline), observable(cr)
    if a == b:
        return True, f"{len(a)} observable items match"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return False, f"item {i}: legacy {x} vs {cr.impl} {y}"
    return False, f"length legacy {len(a)} vs {cr.impl} {len(b)}: extra {(a if len(a) > len(b) else b)[min(len(a), len(b)):][:3]}"


@check("ttft_ratio_max")
async def _ttft_ratio(cr: CaseRun, value: float) -> tuple[bool, str]:
    if cr.baseline is None:
        return False, "no legacy baseline"
    base, mine = ttft_ms(cr.baseline.runs[0]), ttft_ms(cr.runs[0])
    if base is None or mine is None:
        return False, f"ttft missing (legacy {base}, {cr.impl} {mine})"
    return mine <= base * value, f"{mine}ms vs legacy {base}ms (max x{value})"


@check("ttft_ms_max")
async def _ttft_max(cr: CaseRun, value: int) -> tuple[bool, str]:
    t = ttft_ms(cr.runs[0])
    return t is not None and t <= value, f"ttft {t}ms (max {value})"


@check("ttft_delta_vs_normal_max", needs_ports=True)
async def _ttft_delta(cr: CaseRun, value: int, **overrides: Any) -> tuple[bool, str]:
    """Re-run the case with the fault removed (memory not degraded, no faults) and compare."""
    spec = cr.case.ports.model_copy(deep=True)
    spec.faults = []
    spec.memory = spec.memory.model_copy(update={"degraded": False})
    normal = await execute(cr.case, cr.impl, ports=spec, run_prefix="normal")
    a, b = ttft_ms(cr.runs[0]), ttft_ms(normal.runs[0])
    return a is not None and b is not None and a - b <= value, f"degraded {a}ms vs normal {b}ms (max +{value}ms)"


# ---------------------------------------------------------------- outcomes and errors


@check("outcomes")
async def _outcomes(cr: CaseRun, value: list[str]) -> tuple[bool, str]:
    got = [r.outcome for r in cr.runs]
    return got == value, f"outcomes {got} (want {value})"


@check("error")
async def _error(cr: CaseRun, code: str, retryable: bool | None = None, within_ms: int | None = None) -> tuple[bool, str]:
    r = cr.runs[-1]
    fin, start = r.finished, r.events[0]
    err = fin.payload.error if fin else None
    if err is None:
        return False, f"no error (outcome {fin.payload.outcome if fin else None})"
    problems = []
    if err.code.value != code:
        problems.append(f"code {err.code.value} != {code}")
    if retryable is not None and err.retryable != retryable:
        problems.append(f"retryable {err.retryable} != {retryable}")
    took = ms(fin.ts) - ms(start.ts)
    if within_ms is not None and took > within_ms:
        problems.append(f"cause shown after {took}ms (max {within_ms})")
    return not problems, "; ".join(problems) or f"{code} after {took}ms"


@check("no_error_visible")
async def _no_error_visible(cr: CaseRun) -> tuple[bool, str]:
    bad = [f"{e.type}:{e.payload.error.code.value}" for e in of(cr.all_events, "run.finished", "tool.result")
           if e.payload.error is not None and e.payload.error.code.value.startswith("model_")]
    failed = [r.outcome for r in cr.runs if r.outcome == "failed"]
    return not bad and not failed, "; ".join(bad) or ("failed run" if failed else "no model errors reached the user")


@check("notice")
async def _notice(cr: CaseRun, code: str) -> tuple[bool, str]:
    got = [e.payload.code for e in of(cr.all_events, "notice")]
    return code in got, f"notices {got}"


# ---------------------------------------------------------------- tools (E04)


@check("side_effects_once", needs_ports=True)
async def _side_effects_once(cr: CaseRun, tool: str) -> tuple[bool, str]:
    """Run the same request again against the same idempotency ledger (what failover does):
    deterministic keys mean the side effect is not repeated."""
    first = cr.bundle.tools.side_effect_runs[tool]
    from fakes.context import FakeBundle

    again = FakeBundle(cr.case.ports, shared_ledger=cr.bundle.tools.ledger)
    await execute(cr.case, cr.impl, bundle=again, with_continuations=True)
    second = again.tools.side_effect_runs[tool]
    return first == 1 and second == 0, f"{tool}: first run {first} executions, re-run {second} (want 1, 0)"


@check("untrusted_wrapped", needs_ports=True)
async def _untrusted(cr: CaseRun, tool: str) -> tuple[bool, str]:
    calls = cr.bundle.models.calls
    seen = [m for c in calls for m in c["messages"] if m["role"] == "tool" and "<untrusted_content" in m["content"]]
    return bool(seen), f"{len(seen)} wrapped {tool} results reached the model"


@check("tool_not_called")
async def _tool_not_called(cr: CaseRun, tool: str) -> tuple[bool, str]:
    n = sum(1 for e in of(cr.all_events, "tool.called") if e.payload.tool == tool)
    return n == 0, f"{tool} called {n} times"


# ---------------------------------------------------------------- relay (E02)


@check("relay_completes")
async def _relay(cr: CaseRun) -> tuple[bool, str]:
    problems = []
    for r in cr.runs:
        ev = r.events
        for i, h in enumerate(ev):
            if h.type != "handoff":
                continue
            after = ev[i + 1:]
            spoke = [e for e in after if e.type == "message.completed" and e.payload.author == h.payload.to_agent]
            if not spoke:
                problems.append(f"handed off to {h.payload.to_agent} but it never spoke")
            again = [e for e in after if e.type == "message.completed" and e.payload.author == h.payload.from_agent]
            if again:
                problems.append(f"{h.payload.from_agent} replied again after handing off")
        texts = [e.payload.text for e in of(ev, "message.completed")]
        dups = [t for t, n in Counter(texts).items() if n > 1 and t]
        if dups:
            problems.append(f"duplicate replies: {dups[:1]}")
    if not any(of(r.events, "handoff") for r in cr.runs):
        problems.append("no handoff happened")
    return not problems, "; ".join(problems) or "relay completed, one reply"


# ---------------------------------------------------------------- people in the loop (E06)


@check("interrupt_card_complete")
async def _card(cr: CaseRun) -> tuple[bool, str]:
    items = of(cr.all_events, "interrupt.requested")
    bad = []
    for e in items:
        p = e.payload
        if not (p.assignee and p.default_action and p.expires_at and p.fingerprint):
            bad.append(p.interrupt_id)
        elif ms(p.expires_at) <= ms(e.ts):
            bad.append(f"{p.interrupt_id} already expired")
    return bool(items) and not bad, f"{len(items)} cards; incomplete {bad}" if bad else f"{len(items)} complete cards"


@check("never_asked_twice")
async def _never_twice(cr: CaseRun) -> tuple[bool, str]:
    fps = [e.payload.fingerprint for e in of(cr.all_events, "interrupt.requested")]
    dups = [f for f, n in Counter(fps).items() if n > 1]
    return not dups, f"asked {len(fps)} questions; repeated {dups}" if dups else f"{len(fps)} distinct questions"


@check("stale_resume_rejected")
async def _stale(cr: CaseRun, tool: str | None = None) -> tuple[bool, str]:
    last = cr.runs[-1]
    err = last.finished.payload.error if last.finished else None
    executed = [e for e in of(last.events, "tool.called")]
    ok = err is not None and err.code.value == "resume_invalid" and not executed
    return ok, f"duplicate submit -> {err.code.value if err else last.outcome}, {len(executed)} tool calls"


# ---------------------------------------------------------------- reconnect (E07)


@check("reattach_exact")
async def _reattach(cr: CaseRun) -> tuple[bool, str]:
    r = cr.runs[0]
    if r.disconnected_at is None:
        return False, "case does not disconnect"
    seqs = r.seen_before_disconnect + r.seen_after_reattach
    full = list(range(1, (r.finished.seq if r.finished else 0) + 1))
    problems = []
    if seqs != full:
        problems.append(f"seen {len(seqs)} events, want {len(full)} exactly once")
    if r.outcome != "completed":
        problems.append(f"run ended {r.outcome} after the reader left")
    pre = [e for e in r.events if e.seq in r.seen_before_disconnect]
    if any(e.type == "message.delta" for e in pre) and not any(e.type == "message.started" for e in pre):
        problems.append("message_id not known before the drop")
    return not problems, "; ".join(problems) or f"dropped at seq {r.disconnected_at}, re-attached, nothing lost or duplicated"


# ---------------------------------------------------------------- gateway (E08-E11)


@check("no_model_marked_offline", needs_ports=True)
async def _offline(cr: CaseRun) -> tuple[bool, str]:
    m = cr.bundle.models.marked_offline
    return not m, f"marked offline: {sorted(m)}" if m else "no model marked offline"


@check("route_used", needs_ports=True)
async def _route_used(cr: CaseRun, route: str) -> tuple[bool, str]:
    routes = [e.payload.route for e in of(cr.all_events, "usage")]
    return bool(routes) and all(r == route for r in routes), f"routes {routes}"


@check("model_calls", needs_ports=True)
async def _model_calls(cr: CaseRun, purpose: str, max: int, response_format: str | None = None) -> tuple[bool, str]:
    calls = [c for c in cr.bundle.models.calls if c["purpose"] == purpose]
    fmt_ok = response_format is None or all(c["response_format"] == response_format for c in calls)
    return 0 < len(calls) <= max and fmt_ok, f"{len(calls)} {purpose} calls (max {max}), formats {[c['response_format'] for c in calls]}"


@check("provider_flag", needs_ports=True)
async def _provider_flag(cr: CaseRun, purpose: str, flag: str, value: Any) -> tuple[bool, str]:
    got = [c["provider_flags"].get(flag) for c in cr.bundle.models.calls if c["purpose"] == purpose]
    return bool(got) and all(g == value for g in got), f"{flag}={got} on {purpose} calls (want {value})"


@check("call_latency_ms_max", needs_ports=True)
async def _latency(cr: CaseRun, purpose: str, value: int) -> tuple[bool, str]:
    lat = [c["t_end"] - c["t_start"] for c in cr.bundle.models.calls if c["purpose"] == purpose and "t_end" in c]
    return bool(lat) and max(lat) <= value, f"{purpose} latency {lat}ms (max {value})"


# ---------------------------------------------------------------- memory (E05, E12)


@check("memory_scopes_truthful", needs_ports=True)
async def _scopes(cr: CaseRun) -> tuple[bool, str]:
    reported = [e.payload for e in of(cr.all_events, "memory.searched")]
    actual = cr.bundle.memory.searches
    if not reported or len(reported) != len(actual):
        return False, f"{len(reported)} memory.searched events for {len(actual)} searches"
    bad = [(r.scopes_searched, a["searched"]) for r, a in zip(reported, actual) if r.scopes_searched != a["searched"]]
    return not bad, f"reported vs actual {bad}" if bad else f"scopes reported as searched: {reported[0].scopes_searched}"


@check("memory_degraded_flagged")
async def _degraded(cr: CaseRun) -> tuple[bool, str]:
    ms_ = of(cr.all_events, "memory.searched")
    flagged = any("embeddings" in e.payload.degraded for e in ms_)
    noticed = any(e.payload.code == "memory_degraded" for e in of(cr.all_events, "notice"))
    return flagged and noticed, f"memory.searched.degraded={flagged}, notice={noticed}"


# ---------------------------------------------------------------- delegation (E03, E09)


@check("artifact_valid", needs_ports=True)
async def _artifact(cr: CaseRun, kind: str) -> tuple[bool, str]:
    arts = [e.payload for e in of(cr.all_events, "artifact.created") if e.payload.kind == kind]
    if not arts:
        return False, f"no {kind} artifact"
    data = cr.bundle.tools.files.get(arts[-1].ref)
    if data is None:
        return False, f"{arts[-1].ref} not stored"
    try:
        names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    except zipfile.BadZipFile:
        return False, "not a zip"
    ok = "ppt/presentation.xml" in names and "[Content_Types].xml" in names
    return ok, f"{arts[-1].ref}: {len(names)} parts" if ok else f"missing PPTX parts: {names}"


@check("artifact_delivered")
async def _delivered(cr: CaseRun, kind: str) -> tuple[bool, str]:
    arts = [e.payload.kind for e in of(cr.all_events, "artifact.created")]
    return kind in arts and cr.runs[-1].outcome == "completed", f"artifacts {arts}, outcome {cr.runs[-1].outcome}"


@check("plan_kept_on_failure")
async def _plan_kept(cr: CaseRun) -> tuple[bool, str]:
    plans = of(cr.all_events, "plan.updated")
    if not plans:
        return False, "no plan"
    steps = plans[-1].payload.steps
    st = [s.status for s in steps]
    ok = "failed" in st and "done" in st and all(s in ("done", "failed", "pending") for s in st)
    return ok and cr.runs[-1].outcome == "failed", f"final plan {st}, outcome {cr.runs[-1].outcome}"


# ---------------------------------------------------------------- switch (E13)


@check("switch_and_shadow", needs_ports=True)
async def _switch(cr: CaseRun) -> tuple[bool, str]:
    from router.check import check_switch

    return await check_switch(cr)
