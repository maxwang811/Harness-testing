"""Legacy SSE (today's four paths) -> harness events. This is what Victor's legacy wrappers
(LegacyAssistHarness / LegacyDelegationHarness) do at the cut points.

STATUS: the 13 type names below are the ones named in the split design and the goals doc;
the remaining 2 of the 15 and every payload field name are ASSUMED until the inventory in
week 0 (grep the four paths for the SSE writer) replaces them. Field names marked (?) are guesses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import ErrorCode, ErrorInfo

# legacy type -> (typed today?, harness event(s), notes)
LEGACY_SSE_TYPES: dict[str, tuple[bool, str, str]] = {
    "meta": (True, "message.started (+ CUSTOM routing info)", "carries message_id(?) and model(?) at the start"),
    "token": (True, "message.delta", "text(?)"),
    "done": (True, "message.completed, usage, run.finished(completed)", "message_id(?), usage(?)"),
    "error": (True, "run.finished(failed, error{code,retryable})", "today has no code; map by message pattern"),
    "memory": (True, "memory.searched", "scopes(?), hits(?)"),
    "tool_call_started": (False, "tool.called", "call_id(?), name(?), args(?)"),
    "tool_call_result": (False, "tool.result", "call_id(?), ok(?), summary(?)"),
    "handoff": (False, "handoff + subagent.started", "from(?), to(?), reason(?)"),
    "loop_in": (False, "interrupt.requested + run.finished(interrupted)", "question(?), options(?), approver(?), deadline(?)"),
    "credit_limit": (False, "interrupt.requested(kind=input, reason=credit_limit) + run.finished(interrupted)", "someone must top up"),
    "notice": (False, "notice{code}", "today carries English titles that iOS parses -> becomes a code"),
    "coordination_plan": (False, "plan.updated", "plan_id(?), steps(?)"),
    "delegation": (False, "plan.updated / step.* (delegation progress, polled today)", "status(?)"),
    "<inventory-14>": (False, "TBD", "fill from the week-0 inventory"),
    "<inventory-15>": (False, "TBD", "fill from the week-0 inventory"),
}


@dataclass
class LegacyMapper:
    """Stateful: synthesizes run/step framing that the legacy stream does not have."""

    harness: str
    version: str
    scenario: str
    author: str
    _message_id: str | None = None
    _message_open: bool = False
    _text: list[str] = field(default_factory=list)
    _started: bool = False
    _step_open: bool = False
    _steps: int = 0
    _interrupts: list[str] = field(default_factory=list)
    _msg_n: int = 0
    _msg_author: str = ""

    def _ensure_started(self, out: list[tuple[str, dict[str, Any]]]) -> None:
        if not self._started:
            self._started = True
            out.append(("run.started", {"harness": self.harness, "version": self.version, "scenario": self.scenario}))
        if not self._step_open:
            self._steps += 1
            self._step_open = True
            out.append(("step.started", {"step_id": f"legacy_step_{self._steps}", "kind": "respond"}))

    def _close_step(self, out: list[tuple[str, dict[str, Any]]], status: str = "ok") -> None:
        if self._step_open:
            out.append(("step.finished", {"step_id": f"legacy_step_{self._steps}", "status": status}))
            self._step_open = False

    def _close_message(self, out: list[tuple[str, dict[str, Any]]]) -> None:
        if self._message_open:
            out.append(("message.completed", {"message_id": self._message_id, "author": self._msg_author,
                                              "parts": [{"type": "text", "text": "".join(self._text)}]}))
            self._message_open = False
            self._message_id = None
            self._text = []

    def feed(self, event: str, data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """One legacy SSE frame -> zero or more (harness_type, payload) pairs."""
        out: list[tuple[str, dict[str, Any]]] = []
        self._ensure_started(out)
        if event == "meta":
            self._message_id = data.get("message_id") or self._message_id
        elif event == "token":
            if not self._message_open:
                self._message_open = True
                self._msg_n += 1
                self._message_id = self._message_id or f"legacy_msg_{self._msg_n}"
                self._msg_author = data.get("author", self.author)
                out.append(("message.started", {"message_id": self._message_id, "author": self._msg_author}))
            self._text.append(data.get("text", ""))
            out.append(("message.delta", {"message_id": self._message_id, "text": data.get("text", "")}))
        elif event == "tool_call_started":
            self._close_message(out)
            out.append(("tool.called", {"call_id": data["call_id"], "tool": data["name"],
                                        "idempotency_key": f"legacy:{data['call_id']}", "args_preview": data.get("args", {})}))
        elif event == "tool_call_result":
            if data.get("artifact"):  # contract order: artifact.created before the tool.result that made it
                out.append(("artifact.created", data["artifact"]))
            out.append(("tool.result", {"call_id": data["call_id"], "ok": data.get("ok", True), "summary": data.get("summary", ""),
                                        "artifact_refs": [data["artifact"]["ref"]] if data.get("artifact") else []}))
        elif event == "handoff":
            self._close_message(out)
            out.append(("handoff", {"from_agent": data["from"], "to_agent": data["to"], "reason": data.get("reason", "")}))
            self.author = data["to"]
        elif event == "memory":
            out.append(("memory.searched", {"query": data.get("query", ""), "scopes_requested": data.get("scopes", []),
                                            "scopes_searched": data.get("scopes", []), "hits": data.get("hits", 0),
                                            "degraded": data.get("degraded", [])}))
        elif event in ("coordination_plan", "delegation"):
            out.append(("plan.updated", {"plan_id": data["plan_id"], "steps": data["steps"]}))
        elif event == "notice":
            out.append(("notice", {"code": data.get("code", "legacy_notice"), "level": data.get("level", "info"),
                                   "params": data.get("params", {})}))
        elif event in ("loop_in", "credit_limit"):
            self._close_message(out)
            iid = data.get("id", f"legacy_int_{len(self._interrupts) + 1}")
            self._interrupts.append(iid)
            out.append(("interrupt.requested", {
                "interrupt_id": iid,
                "kind": data.get("kind", "approval" if event == "loop_in" else "input"),
                "reason": "credit_limit" if event == "credit_limit" else data.get("reason", "tool_approval"),
                "prompt": data.get("question", ""),
                "assignee": data.get("approver", "requester"),
                "default_action": data.get("default", "deny"),
                "expires_at": data.get("deadline", "2026-01-02T00:00:00.000Z"),
                "fingerprint": data.get("fingerprint", iid),
                "call_id": data.get("call_id"),
            }))
            self._close_step(out)
            out.append(("run.finished", {"outcome": "interrupted", "interrupt_ids": list(self._interrupts)}))
        elif event == "done":
            self._close_message(out)
            if data.get("usage"):
                out.append(("usage", data["usage"]))
            self._close_step(out)
            out.append(("run.finished", {"outcome": "completed"}))
        elif event == "error":
            self._close_message(out)
            code = ErrorCode(data["code"]) if data.get("code") in ErrorCode._value2member_map_ else ErrorCode.INTERNAL
            self._close_step(out, "failed")
            out.append(("run.finished", {"outcome": "failed", "error": ErrorInfo.of(code, data.get("message", ""))}))
        return out
