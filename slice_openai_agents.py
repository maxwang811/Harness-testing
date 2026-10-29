"""Goal 2, candidate B: the slice on the OpenAI Agents SDK (openai-agents 0.24).

Not run yet (written from the docs). Contract -> OpenAI Agents SDK:
  ModelGateway  -> a custom Model whose get_response calls ctx.models.chat and builds Responses-API items
  tools         -> FunctionTool; needs_approval -> result.interruptions
  continuation  -> RunState.to_string / RunState.from_string, then approve or reject
  @-relay       -> native handoff() named relay_to_<staff>
"""
from __future__ import annotations

import json
from typing import Any

from agents import Agent, FunctionTool, RunConfig, Runner, handoff
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

try:
    from agents import RunState
except ImportError:
    from agents.run_state import RunState  # type: ignore[no-redef]

from contract import AgentSpec, Event, Recorder, RunContext, RunRequest, Tool, ToolCall, approval_card, relay_tool_name


def _get(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _text(content: Any) -> str:
    return content if isinstance(content, str) else "".join(_get(c, "text") or "" for c in content or [])


def to_chat(system: str | None, items: str | list[Any]) -> list[dict]:
    """Responses-API input items -> our gateway's chat messages."""
    out: list[dict] = [{"role": "system", "content": system}] if system else []
    for it in [{"role": "user", "content": items}] if isinstance(items, str) else items:
        kind, role = _get(it, "type"), _get(it, "role")
        if kind == "function_call":
            call = {"id": _get(it, "call_id"), "name": _get(it, "name"), "args": json.loads(_get(it, "arguments") or "{}")}
            if out and out[-1]["role"] == "assistant":
                out[-1]["tool_calls"].append(call)
            else:
                out.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            out.append({"role": "tool", "tool_call_id": _get(it, "call_id"), "content": _text(_get(it, "output"))})
        elif role == "user":
            out.append({"role": "user", "content": _text(_get(it, "content"))})
        elif role == "assistant":
            out.append({"role": "assistant", "content": _text(_get(it, "content")), "tool_calls": []})
    return out


class GatewayModel(Model):
    def __init__(self, spec: AgentSpec, ctx: RunContext, rec: Recorder):
        self.spec, self.ctx, self.rec = spec, ctx, rec

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing,
                           *args: Any, **kwargs: Any) -> ModelResponse:
        specs = [{"name": t.name, "description": t.description, "parameters": t.params_json_schema}
                 for t in tools if isinstance(t, FunctionTool)]
        specs += [{"name": h.tool_name, "description": h.tool_description, "parameters": h.input_json_schema}
                  for h in handoffs]
        reply = await self.ctx.models.chat(self.spec.ref, to_chat(system_instructions, input), specs)
        output: list[Any] = []
        if reply.text:
            self.rec.emit("message.completed", self.spec.ref, text=reply.text)
            output.append(ResponseOutputMessage.model_construct(
                id="msg", type="message", role="assistant", status="completed",
                content=[ResponseOutputText.model_construct(type="output_text", text=reply.text, annotations=[])]))
        for c in reply.tool_calls:
            output.append(ResponseFunctionToolCall.model_construct(
                type="function_call", id=f"fc_{c.id}", call_id=c.id, name=c.name,
                arguments=json.dumps(c.args, ensure_ascii=False), status="completed"))
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("the slice uses Runner.run")


class OpenAIAgentsSlice:
    name = "openai-agents"
    config = RunConfig(tracing_disabled=True)

    async def run(self, req: RunRequest, ctx: RunContext) -> list[Event]:
        rec = Recorder()
        rec.emit("run.started", req.agent.ref)
        agent = self.build(req.agent, req, ctx, rec)
        if req.resume_state:
            state = await RunState.from_string(agent, req.resume_state["sdk_state"])
            interrupt_of = {call_id: iid for iid, call_id in req.resume_state["calls"].items()}
            for item in state.get_interruptions():
                if req.resume_inputs.get(interrupt_of[_get(item.raw_item, "call_id")]):
                    state.approve(item)
                else:
                    state.reject(item)
            result = await Runner.run(agent, state, run_config=self.config)
        else:
            result = await Runner.run(agent, req.input, run_config=self.config)

        if result.interruptions:
            cards = [approval_card(req.run_id, ToolCall(_get(i.raw_item, "call_id"), _get(i.raw_item, "name"),
                                                        json.loads(_get(i.raw_item, "arguments") or "{}")))
                     for i in result.interruptions]
            for card in cards:
                rec.emit("interrupt.requested", req.agent.ref, **card)
            rec.emit("run.finished", req.agent.ref, outcome="interrupted",
                     interrupt_ids=[c["interrupt_id"] for c in cards],
                     resume_state={"sdk_state": result.to_state().to_string(),
                                   "calls": {c["interrupt_id"]: c["call_id"] for c in cards}})
        else:
            rec.emit("run.finished", req.agent.ref, outcome="completed")
        return rec.events

    def build(self, spec: AgentSpec, req: RunRequest, ctx: RunContext, rec: Recorder) -> Agent:
        def tool(t: Tool) -> FunctionTool:
            async def invoke(tool_ctx: Any, args_json: str) -> str:
                args = json.loads(args_json or "{}")
                key = f"{req.run_id}:{tool_ctx.tool_call_id}"
                rec.emit("tool.called", spec.ref, call_id=tool_ctx.tool_call_id, tool=t.name, args=args,
                         idempotency_key=key)
                output = await ctx.tools.call(t.name, args, key)
                rec.emit("tool.result", spec.ref, call_id=tool_ctx.tool_call_id, output=output)
                return output

            return FunctionTool(name=t.name, description=t.description, params_json_schema=t.parameters,
                                on_invoke_tool=invoke, strict_json_schema=False, needs_approval=t.needs_approval)

        def relay(target: AgentSpec):
            async def on_handoff(_ctx: Any) -> None:
                rec.emit("handoff", spec.ref, to=target.ref)

            return handoff(self.build(target, req, ctx, rec), tool_name_override=relay_tool_name(target.ref),
                           on_handoff=on_handoff)

        return Agent(name=relay_tool_name(spec.ref).removeprefix("relay_to_"), instructions=spec.instructions,
                     model=GatewayModel(spec, ctx, rec), tools=[tool(t) for t in spec.tools],
                     handoffs=[relay(req.roster[r]) for r in spec.relays])
