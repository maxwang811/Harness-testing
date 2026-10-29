"""Shared glue for the framework slices (Goal 2). Everything a framework does NOT give us lives
here once, so the two slice files contain only framework-specific code and their size can be
compared fairly: events and their order, steps, usage/budget, idempotency keys, middleware,
memory retrieval, resume validation and checkpoints.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from harness.errors import ErrorCode, ErrorInfo, HarnessError
from harness.ports import ChatMessage, ModelCall, ModelResult, RunContext, ToolOutcome, Transform
from harness.runtime import QueueRunHandle, iso_from_ms
from harness.types import AgentSpec, MentionInput, RunRequest, TextInput

RELAY_PREFIX = "relay_to_"
DECLINED = "The user declined this action."
APPROVAL_SCHEMA = {"type": "object", "properties": {"approved": {"type": "boolean"}}, "required": ["approved"]}


def short(ref: str) -> str:
    return ref.split(":", 1)[-1].split("@", 1)[0]


def fingerprint(name: str, args: dict[str, Any]) -> str:
    return hashlib.sha1((name + json.dumps(args, sort_keys=True, ensure_ascii=False)).encode()).hexdigest()[:16]


def system_prompt(agent: AgentSpec, extra: str = "") -> str:
    lines = [f"You are {agent.persona.name}, {agent.persona.role}. Reply in {agent.persona.language}."]
    lines += [b.text for b in agent.instructions]
    if agent.skills:
        lines.append("Skills you can load: " + "; ".join(f"{s.name}: {s.description}" for s in agent.skills))
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def find_harness_error(exc: BaseException) -> HarnessError | None:
    """Frameworks may wrap our port errors; walk the chain to find the typed one."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, HarnessError):
            return exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]
    return None


class SliceRunBase:
    impl_name = "slice"
    impl_version = "0.0.0"

    def __init__(self, req: RunRequest, ctx: RunContext, h: QueueRunHandle):
        self.req, self.ctx, self.h = req, ctx, h
        self.rid = req.run_id
        self._n: dict[str, int] = {}
        self.open_step: dict[str, str] = {}  # agent ref -> its open step id
        self.parent_of: dict[str, str | None] = {}  # agent ref -> parent id for its steps (subagent run id)
        self.asked: dict[str, bool] = {}  # fingerprint -> the user's answer, carried across continuations
        self.last_text: dict[str, str] = {}

    @property
    def impl_tag(self) -> str:
        return f"{self.impl_name}@{self.impl_version}"

    def nid(self, prefix: str) -> str:
        self._n[prefix] = self._n.get(prefix, 0) + 1
        return f"{self.rid}:{prefix}{self._n[prefix]}"

    async def emit(self, type_: str, _parent: str | None = None, _by: str | None = None, **f: Any):
        return await self.h.emitter.emit(type_, parent_id=_parent, by_agent=_by, **f)

    def user_text(self) -> str:
        return " ".join(i.text if isinstance(i, TextInput) else i.display for i in self.req.input
                        if isinstance(i, (TextInput, MentionInput)))

    def agent_by_ref(self, ref: str) -> AgentSpec:
        if ref == self.req.agent.ref:
            return self.req.agent
        if ref in self.req.roster:
            return self.req.roster[ref]
        raise HarnessError(ErrorInfo.of(ErrorCode.INVALID_REQUEST, f"{ref} not in roster"))

    # ---------------------------------------------------------------- steps (one per model call)
    async def begin_step(self, agent: AgentSpec, kind: str = "think") -> str:
        await self.end_step(agent)
        sid = self.nid("s")
        self.open_step[agent.ref] = sid
        await self.emit("step.started", self.parent_of.get(agent.ref), agent.ref, step_id=sid, kind=kind)
        return sid

    async def end_step(self, agent_ref: AgentSpec | str, status: str = "ok") -> None:
        ref = agent_ref if isinstance(agent_ref, str) else agent_ref.ref
        sid = self.open_step.pop(ref, None)
        if sid:
            await self.emit("step.finished", self.parent_of.get(ref), ref, step_id=sid, status=status)

    async def end_all_steps(self, status: str = "ok") -> None:
        for ref in list(self.open_step)[::-1]:
            await self.end_step(ref, status)

    # ---------------------------------------------------------------- model calls through the port
    async def call_model(self, agent: AgentSpec, call: ModelCall) -> ModelResult:
        """Stream through ModelGateway, emitting message events live (keeps TTFT), then usage."""
        sid = await self.begin_step(agent)
        resv = await self.ctx.budget.admit(0.01, agent.ref)
        message_id, parts, result = None, [], None
        async for ch in self.ctx.models.stream(call):
            if ch.kind == "text":
                if message_id is None:
                    message_id = self.nid("m")
                    await self.emit("message.started", sid, agent.ref, message_id=message_id, author=agent.ref)
                parts.append(ch.text)
                await self.emit("message.delta", sid, agent.ref, message_id=message_id, text=ch.text)
            else:
                result = ch.result
        if message_id is not None:
            await self.emit("message.completed", sid, agent.ref, message_id=message_id, author=agent.ref,
                            parts=[{"type": "text", "text": "".join(parts)}])
        assert result is not None
        u = result.usage
        await self.emit("usage", sid, agent.ref, usage_id=u.usage_id, model=u.model, route=u.route,
                        input_tokens=u.input_tokens, output_tokens=u.output_tokens, cost=u.cost, currency=u.currency)
        await self.ctx.budget.settle(resv, u)
        if result.fallbacks:
            await self.emit("notice", sid, agent.ref, code="model_fallback",
                            params={"skipped": result.fallbacks, "used": result.route})
        self.last_text[agent.ref] = result.text
        return result

    # ---------------------------------------------------------------- tools through the port
    async def run_tool(self, agent: AgentSpec, name: str, args: dict[str, Any], call_id: str, key: str,
                       approved: bool = True) -> str:
        sid = self.open_step.get(agent.ref)
        await self.emit("tool.called", sid, agent.ref, call_id=call_id, tool=name, idempotency_key=key,
                        args_preview={k: (v if len(str(v)) < 200 else str(v)[:200] + "...") for k, v in args.items()})
        if not approved:
            outcome = ToolOutcome(ok=False, content=DECLINED, summary="declined",
                                  error=ErrorInfo.of(ErrorCode.TOOL_DENIED, "declined by user"))
        else:
            outcome = await self.ctx.tools.call(name, args, key)
            for mw in self.ctx.middleware:
                d = await mw.after_tool(name, args, outcome)
                if isinstance(d, Transform):
                    outcome = d.new_value
        for a in outcome.artifacts:
            await self.emit("artifact.created", sid, agent.ref, **a)
        await self.emit("tool.result", sid, agent.ref, call_id=call_id, ok=outcome.ok, summary=outcome.summary,
                        artifact_refs=outcome.artifact_refs, error=outcome.error)
        return outcome.content

    async def ask_approval(self, agent: AgentSpec, name: str, args: dict[str, Any], call_id: str, key: str) -> dict:
        iid = self.nid("i")
        fp = fingerprint(name, args)
        await self.emit("interrupt.requested", self.open_step.get(agent.ref), agent.ref, interrupt_id=iid,
                        kind="approval", reason="tool_approval", prompt=self.last_text.get(agent.ref, ""),
                        assignee="requester", default_action="deny", fingerprint=fp, call_id=call_id,
                        expires_at=iso_from_ms(self.ctx.clock.now_ms() + 24 * 3600 * 1000),
                        response_schema=APPROVAL_SCHEMA)
        return {"interrupt_id": iid, "call_id": call_id, "tool": name, "args": args, "fingerprint": fp, "key": key}

    # ---------------------------------------------------------------- memory, resume, checkpoints
    async def memory_context(self, agent: AgentSpec) -> str:
        if not self.req.features.has("memory"):
            return ""
        scopes = self.req.features.params.get("memory", {}).get("scopes", ["workspace", "thread"])
        sid = await self.begin_step(agent, "retrieve")
        res = await self.ctx.memory.search(self.user_text(), scopes)
        await self.emit("memory.searched", sid, agent.ref, query=self.user_text(), scopes_requested=scopes,
                        scopes_searched=res.scopes_searched, hits=len(res.hits), degraded=res.degraded)
        if res.degraded:
            await self.emit("notice", sid, agent.ref, code="memory_degraded", level="warning",
                            params={"degraded": res.degraded})
        await self.end_step(agent)
        return ("Relevant memory:\n" + "\n".join(f"[{h.id}] {h.text}" for h in res.hits)) if res.hits else ""

    def record_answers(self, pending: list[dict]) -> None:
        """All open interrupts must be answered in one continuation; anything else is resume_invalid."""
        open_ids = {p["interrupt_id"] for p in pending}
        answers = {ri.interrupt_id: ri for ri in self.req.resume_inputs}
        if set(answers) != open_ids:
            raise HarnessError(ErrorInfo.of(ErrorCode.RESUME_INVALID,
                                            f"expected answers for {sorted(open_ids)}, got {sorted(answers)}"))
        for p in pending:
            ri = answers[p["interrupt_id"]]
            self.asked[p["fingerprint"]] = ri.status == "resolved" and bool((ri.payload or {}).get("approved"))

    async def checkpoint(self, state: dict[str, Any]) -> None:
        state = {**state, "asked": self.asked}
        await self.ctx.checkpoints.save(self.rid, self.nid("cp"), state, version=1, impl=self.impl_tag)

    def load_state(self) -> dict[str, Any] | None:
        rs = self.req.resume_state
        if rs is None:
            if self.req.resume_inputs:
                raise HarnessError(ErrorInfo.of(ErrorCode.RESUME_INVALID, "resume_inputs without resume_state"))
            return None
        if rs.impl != self.impl_tag:
            raise HarnessError(ErrorInfo.of(ErrorCode.RESUME_INVALID, f"state written by {rs.impl}"))
        state = rs.blob if isinstance(rs.blob, dict) else json.loads(rs.blob)
        self.asked = dict(state.get("asked", {}))
        return state

    async def finish(self, outcome: str, interrupt_ids: list[str] | None = None) -> None:
        await self.end_all_steps()
        await self.emit("run.finished", outcome=outcome, interrupt_ids=interrupt_ids or [])

    async def fail(self, exc: BaseException) -> None:
        err = find_harness_error(exc)
        await self.end_all_steps("failed")
        info = err.info if err else ErrorInfo.of(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")
        await self.emit("run.finished", outcome="failed", error=info)

    def chat_input(self) -> list[ChatMessage]:
        return [ChatMessage(role="user", content=self.user_text())]
