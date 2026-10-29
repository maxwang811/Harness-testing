"""Goal-2 candidate B: the slice on the OpenAI Agents SDK (openai-agents >= 0.24).

STATUS: written against the documented API (Model.get_response signature, FunctionTool with
needs_approval callable, handoff(tool_name_override, on_handoff), Runner.run with a RunState,
RunState.to_string/from_string/get_interruptions/approve) but NOT executed yet -- PyPI was
unreachable where it was written. First task of Goal 2 week 2: make tests/test_slices.py green.

Mapping (contract concept -> OpenAI Agents SDK):
  ModelGateway port   -> a custom Model whose get_response calls ctx.models.stream and builds
                         Responses-API output items (gateway adapter below; this is the main cost)
  ToolExecutor port   -> one FunctionTool per ToolSpec; on_invoke_tool calls ctx.tools.call
  approval interrupt  -> needs_approval callable -> result.interruptions -> interrupt.requested
  continuation run    -> RunState.to_string() in the checkpoint; from_string + approve on resume
  @-relay (handoff)   -> native handoff() named relay_to_<staff>; on_handoff emits handoff/subagent events
  events              -> emitted by our adapters, as in the Pydantic AI slice
"""
from __future__ import annotations

import json
from importlib.metadata import version as _pkg_version
from typing import Any

from agents import Agent, FunctionTool, RunConfig, Runner, handoff
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage as SDKUsage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

try:
    from agents import RunState
except ImportError:  # older layouts
    from agents.run_state import RunState  # type: ignore[no-redef]

from harness.ports import ChatMessage, ModelCall, RunContext, ToolCallRequest, ToolSpec
from harness.runtime import QueueRunHandle
from harness.types import AgentSpec, Capabilities, RunRequest

from .common import RELAY_PREFIX, SliceRunBase, fingerprint, short, system_prompt


# ============================================================ gateway adapter (cost of connecting our gateway)


def _get(item: Any, key: str, default: Any = None) -> Any:
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(_get(c, "text", "") or "" for c in (content or []))


def items_to_chat(system: str | None, items: str | list[Any]) -> list[ChatMessage]:
    out = [ChatMessage(role="system", content=system)] if system else []
    if isinstance(items, str):
        return out + [ChatMessage(role="user", content=items)]
    for it in items:
        typ, role = _get(it, "type"), _get(it, "role")
        if typ == "function_call":
            tc = ToolCallRequest(id=_get(it, "call_id"), name=_get(it, "name"),
                                 args=json.loads(_get(it, "arguments") or "{}"))
            if out and out[-1].role == "assistant":
                out[-1] = out[-1].model_copy(update={"tool_calls": [*out[-1].tool_calls, tc]})
            else:
                out.append(ChatMessage(role="assistant", tool_calls=[tc]))
        elif typ == "function_call_output":
            out.append(ChatMessage(role="tool", tool_call_id=_get(it, "call_id"), content=_text(_get(it, "output"))))
        elif role in ("user", "system", "developer"):
            out.append(ChatMessage(role="user" if role == "user" else "system", content=_text(_get(it, "content"))))
        elif role == "assistant":
            out.append(ChatMessage(role="assistant", content=_text(_get(it, "content"))))
        # reasoning and other item types are not sent to our gateway
    return out


def to_output_items(result, n: int) -> list[Any]:
    out: list[Any] = []
    if result.text:
        out.append(ResponseOutputMessage.model_construct(
            id=f"msg_{n}", type="message", role="assistant", status="completed",
            content=[ResponseOutputText.model_construct(type="output_text", text=result.text, annotations=[])]))
    for tc in result.tool_calls:
        out.append(ResponseFunctionToolCall.model_construct(
            id=f"fc_{tc.id}", type="function_call", call_id=tc.id, name=tc.name,
            arguments=json.dumps(tc.args, ensure_ascii=False), status="completed"))
    return out


class CaptainModel(Model):
    def __init__(self, run: "_OARun", spec: AgentSpec):
        self.run, self.spec, self._n = run, spec, 0

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema, handoffs, tracing,
                           *args: Any, **kwargs: Any) -> ModelResponse:
        specs = [ToolSpec(name=t.name, description=t.description or "", parameters=t.params_json_schema)
                 for t in tools if hasattr(t, "params_json_schema")]
        specs += [ToolSpec(name=h.tool_name, description=h.tool_description or "",
                           parameters=h.input_json_schema or {"type": "object", "properties": {}}) for h in handoffs]
        call = ModelCall(agent_ref=self.spec.ref, purpose="reply", policy=self.spec.model_policy,
                         messages=items_to_chat(system_instructions, input), tools=specs)
        result = await self.run.call_model(self.spec, call)
        self._n += 1
        u = result.usage
        return ModelResponse(output=to_output_items(result, self._n),
                             usage=SDKUsage(requests=1, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                                            total_tokens=u.input_tokens + u.output_tokens),
                             response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any):
        raise NotImplementedError("the slice uses Runner.run; message events stream live from get_response")


# ============================================================ the slice


class OpenAIAgentsSlice:
    name = "openai-agents"
    version = "0.1.0+oa" + _pkg_version("openai-agents")
    capabilities = Capabilities(scenarios=["home", "assist"], interrupts=True, relay=True, model_tool_calls=True)

    def start(self, req: RunRequest, ctx: RunContext) -> QueueRunHandle:
        handle = QueueRunHandle(req, ctx)
        return handle.launch(lambda h: _OARun(req, ctx, h).main())


class _OARun(SliceRunBase):
    impl_name = OpenAIAgentsSlice.name
    impl_version = OpenAIAgentsSlice.version
    config = RunConfig(tracing_disabled=True)

    def __init__(self, *a: Any, **kw: Any):
        super().__init__(*a, **kw)
        self.extra = ""
        self.handed_to: list[tuple[AgentSpec, AgentSpec, str]] = []

    def key(self, agent: AgentSpec, call_id: str) -> str:
        return f"{self.rid}:{self.open_step.get(agent.ref, 's0').split(':')[-1]}:{call_id}"

    def build(self, spec: AgentSpec, extra: str = "") -> Agent:
        offered = {t.id: t for t in spec.tools}
        specs = [s.model_copy(update={"requires_approval": s.requires_approval or offered[s.name].requires_approval})
                 for s in self.ctx.tools.list([t.scope for t in spec.tools]) if s.name in offered]
        relays = [handoff(self.build(self.agent_by_ref(ref)), tool_name_override=RELAY_PREFIX + short(ref),
                          tool_description_override=f"Hand the conversation to {ref}",
                          on_handoff=self.on_handoff(spec, self.agent_by_ref(ref)))
                  for ref in spec.subagents]
        return Agent(name=short(spec.ref), instructions=system_prompt(spec, extra), model=CaptainModel(self, spec),
                     tools=[self.make_tool(spec, s) for s in specs], handoffs=relays)

    def make_tool(self, agent: AgentSpec, spec: ToolSpec) -> FunctionTool:
        async def invoke(tctx: Any, args_json: str) -> str:
            args = json.loads(args_json or "{}")
            approved = self.asked.get(fingerprint(spec.name, args), True) if spec.requires_approval else True
            return await self.run_tool(agent, spec.name, args, tctx.tool_call_id, self.key(agent, tctx.tool_call_id),
                                       approved)

        async def needs(_ctx: Any, params: dict[str, Any], _call_id: str) -> bool:
            return spec.requires_approval and fingerprint(spec.name, params) not in self.asked  # never ask twice

        return FunctionTool(name=spec.name, description=spec.description, params_json_schema=spec.parameters,
                            on_invoke_tool=invoke, strict_json_schema=False, needs_approval=needs)

    def on_handoff(self, agent: AgentSpec, target: AgentSpec):
        async def cb(_ctx: Any) -> None:
            sid = self.open_step.get(agent.ref)
            await self.emit("handoff", sid, agent.ref, from_agent=agent.ref, to_agent=target.ref)
            sub_id = self.nid("sub")
            self.parent_of[target.ref] = sub_id
            self.handed_to.append((agent, target, sub_id))
            await self.emit("subagent.started", sid, agent.ref, subagent_run_id=sub_id, agent=target.ref)
        return cb

    async def close_handoffs(self) -> None:
        for agent, target, sub_id in reversed(self.handed_to):
            await self.end_step(target)
            await self.emit("subagent.finished", self.open_step.get(agent.ref), agent.ref, subagent_run_id=sub_id,
                            outcome="completed")
        self.handed_to = []

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
                agent = self.build(lead, self.extra)
                result = await Runner.run(agent, self.user_text(), run_config=self.config, max_turns=req.options.max_steps)
            else:
                self.record_answers(state["pending"])
                self.extra = state.get("extra", "")
                agent = self.build(lead, self.extra)
                sdk_state = await RunState.from_string(agent, state["sdk_state"])
                for item in sdk_state.get_interruptions():
                    sdk_state.approve(item)  # our tool applies the user's real answer (see make_tool)
                await self.begin_step(lead, "tool")
                result = await Runner.run(agent, sdk_state, run_config=self.config, max_turns=req.options.max_steps)
            await self.close_handoffs()
            if result.interruptions:
                pending = []
                for item in result.interruptions:
                    raw = item.raw_item
                    args = json.loads(_get(raw, "arguments") or "{}")
                    call_id = _get(raw, "call_id")
                    pending.append(await self.ask_approval(lead, _get(raw, "name"), args, call_id, self.key(lead, call_id)))
                await self.checkpoint({"agent": lead.ref, "sdk_state": result.to_state().to_string(),
                                       "pending": pending, "extra": self.extra})
                await self.finish("interrupted", [p["interrupt_id"] for p in pending])
                return
            await self.checkpoint({"agent": lead.ref, "pending": [], "extra": self.extra})
            await self.finish("completed")
        except Exception as exc:
            await self.fail(exc)
