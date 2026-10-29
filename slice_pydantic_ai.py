"""Goal 2, candidate A: the slice on Pydantic AI (pydantic-ai-slim 2.x).

Not run yet (written from the docs). Contract -> Pydantic AI:
  ModelGateway  -> FunctionModel whose function calls ctx.models.chat
  tools         -> Tool.from_schema; needs_approval -> ApprovalRequired -> DeferredToolRequests
  continuation  -> message history + DeferredToolResults
  @-relay       -> output function that runs the target agent (Pydantic AI has no handoff)
"""
from __future__ import annotations

from typing import Any

from pydantic_ai import Agent, ApprovalRequired, DeferredToolRequests, DeferredToolResults, ModelMessagesTypeAdapter
from pydantic_ai import RunContext as PydanticContext
from pydantic_ai import Tool as PydanticTool
from pydantic_ai import ToolOutput
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_core import to_jsonable_python

from contract import AgentSpec, Event, Recorder, RunContext, RunRequest, Tool, ToolCall, approval_card, relay_tool_name


def to_chat(messages: list[Any]) -> list[dict]:
    """Pydantic AI message history -> our gateway's chat messages."""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart):
                    out.append({"role": "user", "content": p.content if isinstance(p.content, str) else str(p.content)})
                elif isinstance(p, ToolReturnPart):
                    out.append({"role": "tool", "tool_call_id": p.tool_call_id, "content": p.model_response_str()})
        elif isinstance(m, ModelResponse):
            out.append({"role": "assistant",
                        "content": "".join(p.content for p in m.parts if isinstance(p, TextPart)),
                        "tool_calls": [{"id": p.tool_call_id, "name": p.tool_name, "args": p.args_as_dict()}
                                       for p in m.parts if isinstance(p, ToolCallPart)]})
    instructions = next((m.instructions for m in reversed(messages)
                         if isinstance(m, ModelRequest) and m.instructions), None)
    return ([{"role": "system", "content": instructions}] if instructions else []) + out


class PydanticAISlice:
    name = "pydantic-ai"

    async def run(self, req: RunRequest, ctx: RunContext) -> list[Event]:
        rec = Recorder()
        rec.emit("run.started", req.agent.ref)
        agent = self.build(req.agent, req, ctx, rec)
        if req.resume_state:
            calls = req.resume_state["calls"]  # interrupt_id -> tool_call_id
            result = await agent.run(
                message_history=ModelMessagesTypeAdapter.validate_python(req.resume_state["messages"]),
                deferred_tool_results=DeferredToolResults(
                    approvals={calls[iid]: approved for iid, approved in req.resume_inputs.items()}))
        else:
            result = await agent.run(req.input)

        if isinstance(result.output, DeferredToolRequests):
            cards = [approval_card(req.run_id, ToolCall(p.tool_call_id, p.tool_name, p.args_as_dict()))
                     for p in result.output.approvals]
            for card in cards:
                rec.emit("interrupt.requested", req.agent.ref, **card)
            rec.emit("run.finished", req.agent.ref, outcome="interrupted",
                     interrupt_ids=[c["interrupt_id"] for c in cards],
                     resume_state={"messages": to_jsonable_python(result.all_messages()),
                                   "calls": {c["interrupt_id"]: c["call_id"] for c in cards}})
        else:
            rec.emit("run.finished", req.agent.ref, outcome="completed")
        return rec.events

    def build(self, spec: AgentSpec, req: RunRequest, ctx: RunContext, rec: Recorder) -> Agent:
        async def model(messages: list[Any], info: AgentInfo) -> ModelResponse:
            tools = [{"name": t.name, "description": t.description or "", "parameters": t.parameters_json_schema}
                     for t in [*info.function_tools, *info.output_tools]]
            reply = await ctx.models.chat(spec.ref, to_chat(messages), tools)
            if reply.text:
                rec.emit("message.completed", spec.ref, text=reply.text)
            parts: list[Any] = [TextPart(content=reply.text)] if reply.text else []
            parts += [ToolCallPart(tool_name=c.name, args=c.args, tool_call_id=c.id) for c in reply.tool_calls]
            return ModelResponse(parts=parts)

        def tool(t: Tool) -> PydanticTool:
            async def call(pctx: PydanticContext, **args: Any) -> str:
                if t.needs_approval and not pctx.tool_call_approved:
                    raise ApprovalRequired()
                key = f"{req.run_id}:{pctx.tool_call_id}"
                rec.emit("tool.called", spec.ref, call_id=pctx.tool_call_id, tool=t.name, args=args, idempotency_key=key)
                output = await ctx.tools.call(t.name, args, key)
                rec.emit("tool.result", spec.ref, call_id=pctx.tool_call_id, output=output)
                return output

            return PydanticTool.from_schema(call, name=t.name, description=t.description, json_schema=t.parameters,
                                            takes_ctx=True)

        def relay(target: AgentSpec) -> ToolOutput:
            async def hand_over() -> str:
                rec.emit("handoff", spec.ref, to=target.ref)
                result = await self.build(target, req, ctx, rec).run(req.input)
                return result.output

            return ToolOutput(hand_over, name=relay_tool_name(target.ref))

        return Agent(FunctionModel(model), instructions=spec.instructions, tools=[tool(t) for t in spec.tools],
                     output_type=[str, DeferredToolRequests, *[relay(req.roster[r]) for r in spec.relays]])
