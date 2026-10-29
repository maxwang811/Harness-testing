"""loop-ref: the slice written straight against the ports, no agent framework.

Two jobs: (1) prove the v1 contract is implementable and give every eval case a passing
reference; (2) be the "borrow the ideas, write our own" option in the Goal-2 comparison,
so LOC and behavior of the framework slices are measured against something concrete.

Slice: home chat / assist turn with model-driven tool calls, one @-relay (handoff), one
approval (interrupt -> continuation run), a delegation plan, and one-shot calls.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from harness.errors import ErrorCode, ErrorInfo, HarnessError
from harness.ports import (ChatMessage, Deny, Escalate, ModelCall, ModelResult, RunContext, ToolCallRequest,
                           ToolOutcome, ToolSpec, Transform)
from harness.runtime import QueueRunHandle, iso_from_ms
from harness.types import AgentSpec, Capabilities, RunRequest, TextInput, MentionInput

RELAY_PREFIX = "relay_to_"
APPROVAL_TTL_MS = 24 * 3600 * 1000
APPROVAL_SCHEMA = {"type": "object", "properties": {"approved": {"type": "boolean"}}, "required": ["approved"]}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "title": {"type": "string"}, "assignee": {"type": "string"}},
        "required": ["id", "title", "assignee"]}}},
    "required": ["steps"],
}


def short(ref: str) -> str:
    return ref.split(":", 1)[-1].split("@", 1)[0]


def fingerprint(tc: ToolCallRequest) -> str:
    raw = tc.name + json.dumps(tc.args, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def system_prompt(agent: AgentSpec) -> str:
    lines = [f"You are {agent.persona.name}, {agent.persona.role}. Reply in {agent.persona.language}."]
    lines += [b.text for b in agent.instructions]
    if agent.skills:  # progressive disclosure: names and descriptions only
        lines.append("Skills you can load: " + "; ".join(f"{s.name}: {s.description}" for s in agent.skills))
    return "\n".join(lines)


@dataclass
class Outcome:
    kind: str  # completed | interrupted | cancelled
    interrupt_ids: list[str] = field(default_factory=list)


class ReferenceLoop:
    name = "loop-ref"
    version = "0.1.0"
    capabilities = Capabilities(scenarios=["home", "assist", "delegate", "oneshot"], interrupts=True, relay=True,
                                delegation=True, model_tool_calls=True)

    @property
    def impl_tag(self) -> str:
        return f"{self.name}@{self.version}"

    def start(self, req: RunRequest, ctx: RunContext) -> QueueRunHandle:
        handle = QueueRunHandle(req, ctx)
        return handle.launch(lambda h: _Run(self, req, ctx, h).main())


class _Run:
    def __init__(self, impl: ReferenceLoop, req: RunRequest, ctx: RunContext, h: QueueRunHandle):
        self.impl, self.req, self.ctx, self.h = impl, req, ctx, h
        self.em = h.emitter
        self.rid = req.run_id
        self._n: dict[str, int] = {}
        self.open_steps: list[str] = []
        self.asked: dict[str, bool] = {}  # fingerprint -> answer, carried across continuation runs
        self.plan: dict[str, Any] | None = None
        self.origin_run = req.run_id

    # ------------------------------------------------------------ helpers
    def nid(self, prefix: str) -> str:
        self._n[prefix] = self._n.get(prefix, 0) + 1
        return f"{self.rid}:{prefix}{self._n[prefix]}"

    def agent_by_ref(self, ref: str) -> AgentSpec:
        if ref == self.req.agent.ref:
            return self.req.agent
        if ref in self.req.roster:
            return self.req.roster[ref]
        raise HarnessError(ErrorInfo.of(ErrorCode.INVALID_REQUEST, f"{ref} not in roster"))

    async def emit(self, type_: str, _parent: str | None = None, _by: str | None = None, **f: Any):
        return await self.em.emit(type_, parent_id=_parent, by_agent=_by, **f)

    async def step_start(self, kind: str, parent: str | None, agent: AgentSpec, title: str | None = None) -> str:
        sid = self.nid("s")
        self.open_steps.append(sid)
        await self.emit("step.started", parent, agent.ref, step_id=sid, kind=kind, title=title)
        return sid

    async def step_finish(self, sid: str, parent: str | None, agent: AgentSpec, status: str = "ok") -> None:
        self.open_steps.remove(sid)
        await self.emit("step.finished", parent, agent.ref, step_id=sid, status=status)

    async def checkpoint(self, sid: str, agent: AgentSpec, messages: list[ChatMessage], pending: list[dict]) -> None:
        state = {"agent": agent.ref, "messages": [m.model_dump() for m in messages], "pending": pending,
                 "asked": self.asked, "plan": self.plan, "origin_run": self.origin_run}
        await self.ctx.checkpoints.save(self.rid, sid, state, version=1, impl=self.impl.impl_tag)

    def tool_specs(self, agent: AgentSpec) -> list[ToolSpec]:
        offered = {t.id: t for t in agent.tools}
        specs = []
        for s in self.ctx.tools.list([t.scope for t in agent.tools]):
            if s.name in offered:
                ref = offered[s.name]
                specs.append(s.model_copy(update={"requires_approval": s.requires_approval or ref.requires_approval,
                                                  "side_effects": s.side_effects or ref.side_effects}))
        for sub in agent.subagents:
            specs.append(ToolSpec(name=RELAY_PREFIX + short(sub), description=f"Hand the conversation to {sub}",
                                  parameters={"type": "object", "properties": {"reason": {"type": "string"}}}))
        return specs

    # ------------------------------------------------------------ main
    async def main(self) -> None:
        req = self.req
        rs = req.resume_state
        await self.emit("run.started", None, req.agent.ref, harness=self.impl.name, version=self.impl.version,
                        scenario=req.scenario, continues_run_id=req.continues_run_id,
                        resumed_from_step=rs.step_id if rs else None)
        try:
            if rs is not None:
                if rs.impl != self.impl.impl_tag:
                    raise HarnessError(ErrorInfo.of(ErrorCode.RESUME_INVALID, f"state written by {rs.impl}"))
                state = rs.blob if isinstance(rs.blob, dict) else json.loads(rs.blob)
                self.asked = dict(state.get("asked", {}))
                self.plan = state.get("plan")
                self.origin_run = state.get("origin_run", req.run_id)
                agent = self.agent_by_ref(state["agent"])
                messages = [ChatMessage(**m) for m in state["messages"]]
                await self.apply_resume(agent, messages, state.get("pending", []))
                out = await self.loop(agent, messages, parent=None)
            else:
                if req.resume_inputs:
                    raise HarnessError(ErrorInfo.of(ErrorCode.RESUME_INVALID, "resume_inputs without resume_state"))
                agent = req.agent
                messages = await self.initial_messages(agent)
                if req.scenario == "oneshot":
                    out = await self.oneshot(agent, messages)
                elif req.scenario == "delegate":
                    out = await self.delegate(agent, messages)
                else:
                    out = await self.loop(agent, messages, parent=None)
            if out.kind == "interrupted":
                await self.emit("run.finished", outcome="interrupted", interrupt_ids=out.interrupt_ids)
            elif out.kind == "cancelled":
                await self.emit("run.finished", outcome="cancelled")
            else:
                await self.emit("run.finished", outcome="completed")
        except HarnessError as e:
            for sid in reversed(list(self.open_steps)):
                self.open_steps.remove(sid)
                await self.emit("step.finished", step_id=sid, status="failed")
            await self.emit("run.finished", outcome="failed", error=e.info)

    # ------------------------------------------------------------ context
    async def initial_messages(self, agent: AgentSpec) -> list[ChatMessage]:
        req = self.req
        msgs = [ChatMessage(role="system", content=system_prompt(agent))]
        msgs += await self.ctx.threads.history(req.thread_id)
        text = " ".join(i.text if isinstance(i, TextInput) else i.display for i in req.input
                        if isinstance(i, (TextInput, MentionInput)))
        if req.features.has("memory"):
            scopes = req.features.params.get("memory", {}).get("scopes", ["workspace", "thread"])
            sid = await self.step_start("retrieve", None, agent)
            res = await self.ctx.memory.search(text, scopes)
            await self.emit("memory.searched", sid, agent.ref, query=text, scopes_requested=scopes,
                            scopes_searched=res.scopes_searched, hits=len(res.hits), degraded=res.degraded)
            if res.degraded:
                await self.emit("notice", sid, agent.ref, code="memory_degraded", level="warning",
                                params={"degraded": res.degraded})
            if res.hits:
                msgs.append(ChatMessage(role="system", content="Relevant memory:\n" +
                                        "\n".join(f"[{h.id}] {h.text}" for h in res.hits)))
            await self.step_finish(sid, None, agent)
        msgs.append(ChatMessage(role="user", content=text))
        return msgs

    # ------------------------------------------------------------ model calls
    async def account(self, sid: str, agent: AgentSpec, resv, result: ModelResult) -> None:
        u = result.usage
        await self.emit("usage", sid, agent.ref, usage_id=u.usage_id, model=u.model, route=u.route,
                        input_tokens=u.input_tokens, output_tokens=u.output_tokens, cost=u.cost, currency=u.currency)
        await self.ctx.budget.settle(resv, u)
        if result.fallbacks:
            await self.emit("notice", sid, agent.ref, code="model_fallback", params={"skipped": result.fallbacks,
                                                                                     "used": result.route})

    async def before_model(self, call: ModelCall) -> ModelCall:
        for mw in self.ctx.middleware:
            d = await mw.before_model(call)
            if isinstance(d, Deny):
                raise HarnessError(ErrorInfo.of(ErrorCode.TOOL_DENIED, d.reason))
            if isinstance(d, Transform):
                call = d.new_value
        return call

    async def stream_reply(self, call: ModelCall, sid: str, agent: AgentSpec) -> ModelResult:
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
        assert result is not None, "ModelGateway.stream must end with a FinalChunk"
        return result

    # ------------------------------------------------------------ tools
    async def execute(self, tc: ToolCallRequest, key: str, sid: str, agent: AgentSpec, messages: list[ChatMessage],
                      approved: bool = True) -> None:
        await self.emit("tool.called", sid, agent.ref, call_id=tc.id, tool=tc.name, idempotency_key=key,
                        args_preview={k: (v if len(str(v)) < 200 else str(v)[:200] + "...") for k, v in tc.args.items()})
        if not approved:
            outcome = ToolOutcome(ok=False, content="The user declined this action.", summary="declined",
                                  error=ErrorInfo.of(ErrorCode.TOOL_DENIED, "declined by user"))
        else:
            outcome = await self.ctx.tools.call(tc.name, tc.args, key)
            for mw in self.ctx.middleware:
                d = await mw.after_tool(tc.name, tc.args, outcome)
                if isinstance(d, Transform):
                    outcome = d.new_value
        for a in outcome.artifacts:
            await self.emit("artifact.created", sid, agent.ref, **a)
        await self.emit("tool.result", sid, agent.ref, call_id=tc.id, ok=outcome.ok, summary=outcome.summary,
                        artifact_refs=outcome.artifact_refs, error=outcome.error)
        messages.append(ChatMessage(role="tool", tool_call_id=tc.id, content=outcome.content))

    async def apply_resume(self, agent: AgentSpec, messages: list[ChatMessage], pending: list[dict]) -> None:
        open_ids = {p["interrupt_id"] for p in pending if p.get("interrupt_id")}
        answers = {ri.interrupt_id: ri for ri in self.req.resume_inputs}
        if set(answers) != open_ids:  # all open interrupts in one continuation; stale/duplicate answers rejected
            raise HarnessError(ErrorInfo.of(ErrorCode.RESUME_INVALID,
                                            f"expected answers for {sorted(open_ids)}, got {sorted(answers)}"))
        if not pending:
            return
        sid = await self.step_start("tool", None, agent, title="resume")
        for p in pending:
            tc = ToolCallRequest(**p["call"])
            approved = True
            if p.get("interrupt_id"):
                ri = answers[p["interrupt_id"]]
                approved = ri.status == "resolved" and bool((ri.payload or {}).get("approved"))
                self.asked[p["fingerprint"]] = approved
            await self.execute(tc, p["key"], sid, agent, messages, approved)
        await self.step_finish(sid, None, agent)

    # ------------------------------------------------------------ the loop
    async def loop(self, agent: AgentSpec, messages: list[ChatMessage], parent: str | None) -> Outcome:
        tools = self.tool_specs(agent)
        by_name = {t.name: t for t in tools}
        for _ in range(self.req.options.max_steps):
            if self.h.cancel_requested == "after_step":
                return Outcome("cancelled")
            sid = await self.step_start("think", parent, agent)
            call = await self.before_model(ModelCall(agent_ref=agent.ref, purpose="reply", policy=agent.model_policy,
                                                     messages=list(messages), tools=tools))
            resv = await self.ctx.budget.admit(0.01, agent.ref)
            result = await self.stream_reply(call, sid, agent)
            await self.account(sid, agent, resv, result)
            messages.append(ChatMessage(role="assistant", content=result.text, name=agent.ref,
                                        tool_calls=result.tool_calls))
            if not result.tool_calls:
                await self.step_finish(sid, parent, agent)
                await self.checkpoint(sid, agent, messages, [])
                return Outcome("completed")

            pending: list[dict] = []
            interrupts: list[str] = []
            for tc in result.tool_calls:
                key = f"{self.rid}:{sid.split(':')[-1]}:{tc.id}"  # pending items keep the key of the run that created them
                if tc.name.startswith(RELAY_PREFIX):
                    target = self.agent_by_ref(next(s for s in agent.subagents if short(s) == tc.name[len(RELAY_PREFIX):]))
                    await self.emit("handoff", sid, agent.ref, from_agent=agent.ref, to_agent=target.ref,
                                    reason=tc.args.get("reason", ""), call_id=tc.id)
                    sub_id = self.nid("sub")
                    await self.emit("subagent.started", sid, agent.ref, subagent_run_id=sub_id, agent=target.ref,
                                    parent_call_id=tc.id)
                    messages.append(ChatMessage(role="tool", tool_call_id=tc.id, content=f"handed off to {target.ref}"))
                    sub_msgs = [ChatMessage(role="system", content=system_prompt(target))] + \
                               [m for m in messages if m.role == "user"]
                    out = await self.loop(target, sub_msgs, parent=sub_id)
                    await self.emit("subagent.finished", sid, agent.ref, subagent_run_id=sub_id, outcome=out.kind)
                    await self.step_finish(sid, parent, agent)
                    return out  # a relay hands the turn over: the caller does not speak again
                spec = by_name.get(tc.name)
                needs = bool(spec and spec.requires_approval)
                denied = None
                for mw in self.ctx.middleware:
                    d = await mw.before_tool(tc.name, tc.args)
                    if isinstance(d, Escalate):
                        needs = True
                    elif isinstance(d, Deny):
                        denied = d
                if pending or needs:
                    fp = fingerprint(tc)
                    if needs and fp in self.asked:  # E06: never ask the same question twice
                        await self.execute(tc, key, sid, agent, messages, self.asked[fp])
                        continue
                    item = {"call": tc.model_dump(), "key": key, "step_id": sid}
                    if needs:
                        iid = self.nid("i")
                        item.update(interrupt_id=iid, fingerprint=fp)
                        interrupts.append(iid)
                        await self.emit("interrupt.requested", sid, agent.ref, interrupt_id=iid, kind="approval",
                                        reason="tool_approval", prompt=result.text, assignee="requester",
                                        default_action="deny", fingerprint=fp, call_id=tc.id,
                                        expires_at=iso_from_ms(self.ctx.clock.now_ms() + APPROVAL_TTL_MS),
                                        response_schema=APPROVAL_SCHEMA)
                    pending.append(item)
                    continue
                if denied is not None:
                    await self.execute(tc, key, sid, agent, messages, approved=False)
                    continue
                await self.execute(tc, key, sid, agent, messages)
            await self.step_finish(sid, parent, agent)
            await self.checkpoint(sid, agent, messages, pending)
            if interrupts:
                return Outcome("interrupted", interrupts)
        raise HarnessError(ErrorInfo.of(ErrorCode.MAX_STEPS_EXCEEDED, f"{self.req.options.max_steps} steps"))

    # ------------------------------------------------------------ one-shot and delegation
    async def oneshot(self, agent: AgentSpec, messages: list[ChatMessage]) -> Outcome:
        sid = await self.step_start("respond", None, agent)
        resv = await self.ctx.budget.admit(0.001, agent.ref)
        result = await self.ctx.models.chat(ModelCall(agent_ref=agent.ref, purpose="classify",
                                                      policy=agent.model_policy, messages=messages))
        await self.account(sid, agent, resv, result)
        mid = self.nid("m")
        await self.emit("message.started", sid, agent.ref, message_id=mid, author=agent.ref)
        await self.emit("message.delta", sid, agent.ref, message_id=mid, text=result.text)
        await self.emit("message.completed", sid, agent.ref, message_id=mid, author=agent.ref,
                        parts=[{"type": "text", "text": result.text}])
        await self.step_finish(sid, None, agent)
        return Outcome("completed")

    async def emit_plan(self, sid: str, agent: AgentSpec) -> None:
        await self.emit("plan.updated", sid, agent.ref, plan_id=self.plan["plan_id"], steps=self.plan["steps"])

    async def delegate(self, agent: AgentSpec, messages: list[ChatMessage]) -> Outcome:
        sid = await self.step_start("plan", None, agent)
        call = ModelCall(agent_ref=agent.ref, purpose="plan", policy=agent.model_policy, response_format="json",
                         json_schema=PLAN_SCHEMA, messages=messages + [ChatMessage(
                             role="system", content="Plan the work as JSON: steps with id, title, assignee.")])
        resv = await self.ctx.budget.admit(0.01, agent.ref)
        result = await self.ctx.models.json(call)
        await self.account(sid, agent, resv, result)
        steps = (result.json_value or {}).get("steps") or []
        if not steps:
            raise HarnessError(ErrorInfo.of(ErrorCode.INTERNAL, "planner returned no steps"))
        self.plan = {"plan_id": self.nid("plan"),
                     "steps": [{"id": s["id"], "title": s["title"], "assignee": s["assignee"], "status": "pending"}
                               for s in steps]}
        await self.emit_plan(sid, agent)
        await self.step_finish(sid, None, agent)
        await self.checkpoint(sid, agent, messages, [])
        user_text = next((m.content for m in reversed(messages) if m.role == "user"), "")
        notes: list[str] = []
        for st in self.plan["steps"]:
            dsid = await self.step_start("delegate", None, agent, title=st["title"])
            st["status"] = "running"
            await self.emit_plan(dsid, agent)
            target = self.agent_by_ref(st["assignee"])
            sub_id = self.nid("sub")
            await self.emit("subagent.started", dsid, agent.ref, subagent_run_id=sub_id, agent=target.ref)
            sub_msgs = [ChatMessage(role="system", content=system_prompt(target)),
                        ChatMessage(role="user", content=f"Task: {st['title']}\nRequest: {user_text}\n" + "\n".join(notes))]
            try:
                out = await self.loop(target, sub_msgs, parent=sub_id)
            except HarnessError:
                st["status"] = "failed"  # the plan is kept: done steps stay done, later steps stay pending (E09)
                await self.emit_plan(dsid, agent)
                await self.emit("subagent.finished", dsid, agent.ref, subagent_run_id=sub_id, outcome="failed")
                await self.checkpoint(dsid, agent, messages, [])
                raise
            await self.emit("subagent.finished", dsid, agent.ref, subagent_run_id=sub_id, outcome=out.kind)
            if out.kind != "completed":
                await self.step_finish(dsid, None, agent)
                return out
            st["status"] = "done"
            await self.emit_plan(dsid, agent)
            notes.append(next((m.content for m in reversed(sub_msgs) if m.role == "assistant" and m.content), ""))
            await self.step_finish(dsid, None, agent)
            await self.checkpoint(dsid, agent, messages, [])
        return Outcome("completed")
