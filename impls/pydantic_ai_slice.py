"""Goal-2 candidate A: the slice on Pydantic AI (pydantic-ai-slim >= 2.0).

STATUS: written against the documented 2.x API (Agent, FunctionModel, Tool.from_schema,
ApprovalRequired + ctx.tool_call_approved, DeferredToolRequests/Results, ToolOutput,
ModelMessagesTypeAdapter) but NOT executed yet -- PyPI was unreachable where it was written.
First task of Goal 2 week: `uv sync --extra slices && pytest tests/test_slices.py`.

Mapping (contract concept -> Pydantic AI):
  ModelGateway port      -> FunctionModel whose function calls ctx.models.stream (gateway adapter below)
  ToolExecutor port      -> one Tool.from_schema per ToolSpec; the function calls ctx.tools.call
  approval interrupt     -> ApprovalRequired -> run output DeferredToolRequests -> interrupt.requested
  continuation run       -> message_history (ModelMessagesTypeAdapter) + DeferredToolResults
  @-relay (handoff)      -> output function relay_to_<staff> that runs the target agent; its text is the output
  events                 -> emitted by our adapters (model function, tool function); the framework's own
                            event stream is not needed, which keeps events identical across implementations
"""
from __future__ import annotations

from importlib.metadata import version as _pkg_version
from typing import Any

from pydantic_ai import (Agent, ApprovalRequired, DeferredToolRequests, DeferredToolResults,
                         ModelMessagesTypeAdapter, Tool, ToolOutput)
from pydantic_ai import RunContext as PAIContext
from pydantic_ai.messages import (ModelRequest, ModelResponse, RetryPromptPart, SystemPromptPart, TextPart,
                                  ToolCallPart, ToolReturnPart, UserPromptPart)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_core import to_jsonable_python

from harness.ports import ChatMessage, ModelCall, RunContext, ToolCallRequest, ToolSpec
from harness.runtime import QueueRunHandle
from harness.types import AgentSpec, Capabilities, RunRequest

from .common import DECLINED, RELAY_PREFIX, SliceRunBase, fingerprint, short, system_prompt


# ============================================================ gateway adapter (cost of connecting our gateway)


def to_chat(messages: list[Any]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    instructions = next((m.instructions for m in reversed(messages)
                         if isinstance(m, ModelRequest) and getattr(m, "instructions", None)), None)
    if instructions:
        out.append(ChatMessage(role="system", content=instructions))
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, SystemPromptPart):
                    out.append(ChatMessage(role="system", content=p.content))
                elif isinstance(p, UserPromptPart):
                    text = p.content if isinstance(p.content, str) else " ".join(str(c) for c in p.content)
                    out.append(ChatMessage(role="user", content=text))
                elif isinstance(p, ToolReturnPart):
                    out.append(ChatMessage(role="tool", tool_call_id=p.tool_call_id, content=p.model_response_str()))
                elif isinstance(p, RetryPromptPart):
                    role = "tool" if p.tool_name and p.tool_call_id else "user"
                    out.append(ChatMessage(role=role, tool_call_id=p.tool_call_id if role == "tool" else None,
                                           content=p.model_response()))
        elif isinstance(m, ModelResponse):
            text = "".join(p.content for p in m.parts if isinstance(p, TextPart))
            calls = [ToolCallRequest(id=p.tool_call_id, name=p.tool_name, args=p.args_as_dict())
                     for p in m.parts if isinstance(p, ToolCallPart)]
            out.append(ChatMessage(role="assistant", content=text, tool_calls=calls))
    return out


def to_specs(info: AgentInfo, by_name: dict[str, ToolSpec]) -> list[ToolSpec]:
    defs = list(info.function_tools) + list(getattr(info, "output_tools", []) or [])
    return [by_name.get(d.name) or ToolSpec(name=d.name, description=d.description or "",
                                            parameters=d.parameters_json_schema) for d in defs]


def to_response(result) -> ModelResponse:
    parts: list[Any] = [TextPart(content=result.text)] if result.text else []
    parts += [ToolCallPart(tool_name=tc.name, args=tc.args, tool_call_id=tc.id) for tc in result.tool_calls]
    return ModelResponse(parts=parts or [TextPart(content="")])


# ============================================================ the slice


class PydanticAISlice:
    name = "pydantic-ai"
    version = "0.1.0+pai" + _pkg_version("pydantic-ai-slim")
    capabilities = Capabilities(scenarios=["home", "assist"], interrupts=True, relay=True, model_tool_calls=True)

    def start(self, req: RunRequest, ctx: RunContext) -> QueueRunHandle:
        handle = QueueRunHandle(req, ctx)
        return handle.launch(lambda h: _PAIRun(req, ctx, h).main())


class _PAIRun(SliceRunBase):
    impl_name = PydanticAISlice.name
    impl_version = PydanticAISlice.version

    def build_agent(self, spec: AgentSpec, extra: str = "") -> Agent:
        offered = {t.id: t for t in spec.tools}
        specs = [s.model_copy(update={"requires_approval": s.requires_approval or offered[s.name].requires_approval})
                 for s in self.ctx.tools.list([t.scope for t in spec.tools]) if s.name in offered]
        by_name = {s.name: s for s in specs}

        async def model_fn(messages: list[Any], info: AgentInfo) -> ModelResponse:
            call = ModelCall(agent_ref=spec.ref, purpose="reply", policy=spec.model_policy,
                             messages=to_chat(messages), tools=to_specs(info, by_name))
            return to_response(await self.call_model(spec, call))

        tools = [self.make_tool(spec, s) for s in specs]
        outputs: list[Any] = [str, DeferredToolRequests]
        outputs += [self.make_relay(spec, self.agent_by_ref(ref)) for ref in spec.subagents]
        return Agent(FunctionModel(model_fn), instructions=system_prompt(spec, extra), tools=tools,
                     output_type=outputs)

    def make_tool(self, agent: AgentSpec, spec: ToolSpec) -> Tool:
        async def fn(ctx: PAIContext, **args: Any) -> str:
            key = f"{self.rid}:{ctx.run_step}:{ctx.tool_call_id}"
            if spec.requires_approval:
                fp = fingerprint(spec.name, args)
                if fp in self.asked:  # answered before (this run or an earlier one): never ask twice
                    return await self.run_tool(agent, spec.name, args, ctx.tool_call_id, key, self.asked[fp])
                if not ctx.tool_call_approved:
                    raise ApprovalRequired(metadata={"key": key})
            return await self.run_tool(agent, spec.name, args, ctx.tool_call_id, key)

        return Tool.from_schema(fn, name=spec.name, description=spec.description, json_schema=spec.parameters,
                                takes_ctx=True)

    def make_relay(self, agent: AgentSpec, target: AgentSpec) -> ToolOutput:
        async def relay(ctx: PAIContext, reason: str = "") -> str:
            sid = self.open_step.get(agent.ref)
            await self.emit("handoff", sid, agent.ref, from_agent=agent.ref, to_agent=target.ref, reason=reason,
                            call_id=ctx.tool_call_id)
            sub_id = self.nid("sub")
            self.parent_of[target.ref] = sub_id
            await self.emit("subagent.started", sid, agent.ref, subagent_run_id=sub_id, agent=target.ref,
                            parent_call_id=ctx.tool_call_id)
            result = await self.build_agent(target).run(self.user_text())
            await self.end_step(target)
            await self.emit("subagent.finished", sid, agent.ref, subagent_run_id=sub_id, outcome="completed")
            return result.output if isinstance(result.output, str) else ""

        return ToolOutput(relay, name=RELAY_PREFIX + short(target.ref), description=f"Hand the conversation to {target.ref}")

    async def main(self) -> None:
        req = self.req
        await self.emit("run.started", None, req.agent.ref, harness=self.impl_name, version=self.impl_version,
                        scenario=req.scenario, continues_run_id=req.continues_run_id,
                        resumed_from_step=req.resume_state.step_id if req.resume_state else None)
        try:
            state = self.load_state()
            lead = req.agent
            if state is None:
                self.extra = await self.memory_context(lead)
                agent = self.build_agent(lead, self.extra)
                result = await agent.run(self.user_text())
            else:
                pending = state["pending"]
                self.record_answers(pending)
                self.extra = state.get("extra", "")
                agent = self.build_agent(lead, self.extra)
                history = ModelMessagesTypeAdapter.validate_python(state["messages"])
                # Approve at the framework level; our tool function applies the user's real answer, so
                # denials produce the same events and tool message on every implementation.
                results = DeferredToolResults(approvals={p["call_id"]: True for p in pending})
                await self.begin_step(lead, "tool")
                result = await agent.run(message_history=history, deferred_tool_results=results)
            if isinstance(result.output, DeferredToolRequests):
                pending = []
                for part in result.output.approvals:
                    meta = result.output.metadata.get(part.tool_call_id, {})
                    pending.append(await self.ask_approval(lead, part.tool_name, part.args_as_dict(),
                                                           part.tool_call_id, meta.get("key", part.tool_call_id)))
                await self.checkpoint({"agent": lead.ref, "messages": to_jsonable_python(result.all_messages()),
                                       "pending": pending, "extra": self.extra})
                await self.finish("interrupted", [p["interrupt_id"] for p in pending])
                return
            await self.checkpoint({"agent": lead.ref, "messages": to_jsonable_python(result.all_messages()),
                                   "pending": [], "extra": self.extra})
            await self.finish("completed")
        except Exception as exc:  # port errors arrive wrapped or bare; fail() finds the typed one
            await self.fail(exc)
