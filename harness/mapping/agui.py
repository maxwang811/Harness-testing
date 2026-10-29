"""Harness events -> AG-UI events (client protocol v1). Lives in the experience layer in
production; kept next to the contract here because the mapping table is a Goal-1 deliverable.

Wire format is AG-UI camelCase JSON. Notes on AG-UI status (checked 2026-10-29):
* RUN_FINISHED.outcome {type: "success"|"interrupt", interrupts: [...]} and the RunAgentInput
  `resume` array are in the AG-UI docs (concepts/interrupts) but the Python SDK's
  RunFinishedEvent has no `outcome` field yet -> we emit plain dicts, and also a
  CUSTOM "captain.interrupt" so clients on an older SDK still see the card.
* SUBAGENT_STARTED / SUBAGENT_FINISHED exist; events inside a subagent carry subagentRunId.
* AG-UI has no artifact, memory, notice, author or cancellation concepts -> CUSTOM "captain.*".
"""
from __future__ import annotations

from typing import Any

from ..events import INTERNAL_EVENTS, HarnessEvent

INTERRUPT_REASON = {"approval": "confirmation", "input": "input_required", "choice": "input_required"}


class AguiTranslator:
    """Stateful per run: buffers interrupts until run.finished and diffs plans into JSON Patch."""

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self._interrupts: list[dict[str, Any]] = []
        self._plans: dict[str, list[dict[str, Any]]] = {}
        self._subagent_runs: set[str] = set()
        self._owner: dict[str, str] = {}  # step id or subagent run id -> owning subagent run id

    def _base(self, ev: HarnessEvent, type_: str, **fields: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"type": type_, "timestamp": ev.ts}
        sub = self._owner.get(ev.parent_id or "")
        if sub:
            out["subagentRunId"] = sub
        out.update(fields)
        return out

    def translate(self, ev: HarnessEvent) -> list[dict[str, Any]]:
        if ev.type in INTERNAL_EVENTS or not ev.known:
            return []  # usage never reaches clients; unknown types are ignored (versioning rule)
        p = ev.payload
        b = lambda t, **f: self._base(ev, t, **f)  # noqa: E731
        t = ev.type

        if t == "run.started":
            fields: dict[str, Any] = {"threadId": self.thread_id, "runId": ev.run_id}
            if p.continues_run_id:
                fields["parentRunId"] = p.continues_run_id
            return [b("RUN_STARTED", **fields)]

        if t == "run.finished":
            if p.outcome == "completed":
                return [b("RUN_FINISHED", threadId=self.thread_id, runId=ev.run_id, outcome={"type": "success"})]
            if p.outcome == "interrupted":
                return [b("RUN_FINISHED", threadId=self.thread_id, runId=ev.run_id,
                          outcome={"type": "interrupt", "interrupts": list(self._interrupts)})]
            if p.outcome == "cancelled":
                # AG-UI has no cancelled outcome; a user cancel is not an error.
                return [b("RUN_FINISHED", threadId=self.thread_id, runId=ev.run_id,
                          outcome={"type": "success"}, result={"captain": {"status": "cancelled"}})]
            err = p.error
            return [b("RUN_ERROR", message=err.message if err else "failed",
                      code=err.code.value if err else "internal",
                      **({"rawEvent": {"retryable": err.retryable}} if err else {}))]

        if t == "step.started":
            if ev.parent_id in self._owner:
                self._owner[p.step_id] = self._owner[ev.parent_id]
            return [b("STEP_STARTED", stepName=p.step_id)]
        if t == "step.finished":
            return [b("STEP_FINISHED", stepName=p.step_id)]

        if t == "message.started":
            return [b("TEXT_MESSAGE_START", messageId=p.message_id, role="assistant"),
                    b("CUSTOM", name="captain.message_meta", value={"messageId": p.message_id, "author": p.author})]
        if t == "message.delta":
            return [b("TEXT_MESSAGE_CONTENT", messageId=p.message_id, delta=p.text)] if p.text else []
        if t == "message.completed":
            return [b("TEXT_MESSAGE_END", messageId=p.message_id)]

        if t == "reasoning.summary":
            mid = f"reasoning_{ev.seq}"
            return [b("REASONING_START", messageId=mid),
                    b("REASONING_MESSAGE_START", messageId=mid, role="reasoning"),
                    b("REASONING_MESSAGE_CONTENT", messageId=mid, delta=p.text),
                    b("REASONING_MESSAGE_END", messageId=mid),
                    b("REASONING_END", messageId=mid)]

        if t == "tool.called":
            import json

            return [b("TOOL_CALL_START", toolCallId=p.call_id, toolCallName=p.tool),
                    b("TOOL_CALL_ARGS", toolCallId=p.call_id, delta=json.dumps(p.args_preview, ensure_ascii=False)),
                    b("TOOL_CALL_END", toolCallId=p.call_id)]
        if t == "tool.result":
            return [b("TOOL_CALL_RESULT", messageId=f"{p.call_id}:result", toolCallId=p.call_id,
                      content=p.summary, role="tool")]

        if t == "plan.updated":
            steps = [s.model_dump() for s in p.steps]
            old = self._plans.get(p.plan_id)
            self._plans[p.plan_id] = steps
            if old is None:
                return [b("STATE_SNAPSHOT", snapshot={"plans": {pid: {"steps": s} for pid, s in self._plans.items()}})]
            return [b("STATE_DELTA", delta=_plan_patch(p.plan_id, old, steps))]

        if t == "handoff":
            return [b("CUSTOM", name="captain.handoff",
                      value={"from": p.from_agent, "to": p.to_agent, "reason": p.reason})]
        if t == "subagent.started":
            self._subagent_runs.add(p.subagent_run_id)
            out = b("SUBAGENT_STARTED", subagentRunId=p.subagent_run_id, name=p.agent,
                    **({"parentToolCallId": p.parent_call_id} if p.parent_call_id else {}))
            self._owner[p.subagent_run_id] = p.subagent_run_id  # later events under it carry subagentRunId
            return [out]
        if t == "subagent.finished":
            if p.outcome in ("completed", "cancelled"):
                return [b("SUBAGENT_FINISHED", subagentRunId=p.subagent_run_id, outcome={"type": "success"})]
            if p.outcome == "interrupted":
                return [b("SUBAGENT_FINISHED", subagentRunId=p.subagent_run_id, outcome={"type": "suspended"})]
            return [b("SUBAGENT_ERROR", subagentRunId=p.subagent_run_id, message="subagent failed")]

        if t == "artifact.created":
            return [b("CUSTOM", name="captain.artifact", value=p.model_dump())]

        if t == "interrupt.requested":
            item = {
                "id": p.interrupt_id,
                "reason": "tool_call" if p.call_id and p.kind == "approval" else INTERRUPT_REASON[p.kind],
                "message": p.prompt,
                "expiresAt": p.expires_at,
                "metadata": {"captain": {"kind": p.kind, "reason": p.reason, "assignee": p.assignee,
                                         "defaultAction": p.default_action,
                                         "options": [o.model_dump() for o in p.options]}},
            }
            if p.call_id:
                item["toolCallId"] = p.call_id
            if p.response_schema:
                item["responseSchema"] = p.response_schema
            if self._owner.get(ev.parent_id or ""):
                item["subagentRunId"] = self._owner[ev.parent_id]
            self._interrupts.append(item)
            return [b("CUSTOM", name="captain.interrupt", value=item)]

        if t == "memory.searched":
            return [b("CUSTOM", name="captain.memory", value=p.model_dump())]
        if t == "notice":
            return [b("CUSTOM", name="captain.notice", value=p.model_dump())]
        return []


def _plan_patch(plan_id: str, old: list[dict], new: list[dict]) -> list[dict[str, Any]]:
    """RFC 6902 ops turning old steps into new: replace changed fields, append new steps."""
    ops: list[dict[str, Any]] = []
    base = f"/plans/{plan_id}/steps"
    for i, step in enumerate(new):
        if i >= len(old):
            ops.append({"op": "add", "path": f"{base}/-", "value": step})
            continue
        for k, v in step.items():
            if old[i].get(k) != v:
                ops.append({"op": "replace", "path": f"{base}/{i}/{k}", "value": v})
    for i in range(len(old) - 1, len(new) - 1, -1):
        ops.append({"op": "remove", "path": f"{base}/{i}"})
    return ops


def translate_all(thread_id: str, events: list[HarnessEvent]) -> list[dict[str, Any]]:
    tr = AguiTranslator(thread_id)
    out: list[dict[str, Any]] = []
    for ev in events:
        out.extend(tr.translate(ev))
    return out
