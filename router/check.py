"""E13 offline: switching, pinning, rollback and shadow runs are invisible to users."""
from __future__ import annotations

from typing import Any

from fakes.clock import FakeClock
from fakes.context import FakeBundle

from .router import HarnessRouter, Requester, Rule, RulesSnapshot


async def check_switch(cr: Any) -> tuple[bool, str]:
    from evals.assertions import observable
    from evals.runner import execute

    problems: list[str] = []
    clock = FakeClock()
    snap = RulesSnapshot(rules=[Rule("canary-delegate", "delegate", primary="loop-v1", canary_workspaces=("ws_canary",)),
                                Rule("shadow-assist", "assist", shadows=("loop-v1",), percent=100)])
    router = HarnessRouter(lambda: snap, clock.now_ms)

    # 1. only internal accounts may pin with the header
    if router.route("r1", "assist", Requester("u", "ws_a", is_internal=True, header_impl="loop-v1")).primary != "loop-v1":
        problems.append("internal header not honored")
    if router.route("r2", "assist", Requester("u", "ws_a", is_internal=False, header_impl="loop-v1")).primary == "loop-v1":
        problems.append("external account could pick an implementation")
    # 2. canary by workspace, stable buckets
    if router.route("r3", "delegate", Requester("u", "ws_canary")).primary != "loop-v1":
        problems.append("canary workspace not routed")
    if len({router.route(f"b{i}", "delegate", Requester("u", "ws_x")).primary for i in range(50)}) != 1:
        problems.append("bucket not stable")
    # 3. pinned per run: an interrupted run continues on its implementation even after rollback
    router.route("r4", "delegate", Requester("u", "ws_canary"))
    snap.kill_switch.add("delegate")
    clock._now += 30_000  # rules cache expires within 30 s
    if router.route("r5", "delegate", Requester("u", "ws_canary"), continues_run_id="r4").primary != "loop-v1":
        problems.append("continuation not pinned")
    # 4. rollback: new runs go back to legacy within 30 s
    if router.route("r6", "delegate", Requester("u", "ws_canary")).primary != "legacy-delegation":
        problems.append("kill switch not applied after cache TTL")
    # 5. shadow: same request, replaying tools, no billing, no user-visible output
    d = router.route("r7", "assist", Requester("u", "ws_a"))
    if d.shadows != ("loop-v1",):
        problems.append(f"shadow rule not applied: {d}")
    alone = await execute(cr.case, "legacy-replay", run_prefix="alone")
    primary = await execute(cr.case, "legacy-replay", run_prefix="primary")
    shadow_bundle = FakeBundle(cr.case.ports, shadow=True)
    shadow = await execute(cr.case, "loop-ref", run_prefix="shadow", bundle=shadow_bundle)
    if observable(alone) != observable(primary):
        problems.append("primary output changed when a shadow ran")
    if sum(shadow_bundle.tools.side_effect_runs.values()):
        problems.append(f"shadow had side effects {dict(shadow_bundle.tools.side_effect_runs)}")
    if shadow_bundle.budget.settled:
        problems.append(f"shadow billed {len(shadow_bundle.budget.settled)} usages")
    if any(e.run_id.startswith("shadow") for e in primary.bundle.sink.events):
        problems.append("shadow events leaked into the primary event log")
    if shadow_bundle.files if hasattr(shadow_bundle, "files") else shadow_bundle.tools.files:
        problems.append("shadow wrote files")
    return not problems, "; ".join(problems) or "header, canary, pinning, 30 s rollback and shadow all behave"
