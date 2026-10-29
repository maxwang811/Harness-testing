"""ScriptedModelGateway: a fake ModelGateway that replays golden model turns and injects faults.

It implements the *port contract* (route fallback inside the port, typed GatewayErrors,
402 never marks a model offline, JSON mode chosen per route). The same contract is checked
against the real adapter by the port conformance suite, so the fake cannot drift from it.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, ConfigDict

from harness.errors import ErrorCode, ErrorInfo, GatewayError
from harness.ports import FinalChunk, ModelCall, ModelResult, TextChunk, ToolCallRequest, Usage

from .clock import FakeClock


class ScriptExhausted(AssertionError):
    pass


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = ""
    chunks: list[str] | None = None
    tool_calls: list[dict[str, Any]] = []
    json_value: Any = None
    first_chunk_ms: int = 300
    chunk_ms: int = 25
    input_tokens: int = 800
    output_tokens: int | None = None
    expect: dict[str, Any] = {}  # checks on the request: last_role, last_contains, system_contains, tools_include

    def all_chunks(self) -> list[str]:
        if self.chunks is not None:
            return self.chunks
        return [self.text[i:i + 6] for i in range(0, len(self.text), 6)] if self.text else []


class Fault(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["rate_limited", "timeout", "credit_exhausted", "unavailable", "dead"]
    route: str | None = None  # None = every route
    agent: str | None = None
    purpose: str | None = None
    times: int = 1  # how many attempts it affects; -1 = always


class RouteCaps(BaseModel):
    model_config = ConfigDict(extra="forbid")
    json_mode: bool = True
    thinks_by_default: bool = False  # glm-5.3-flash (E11)
    thinking_ms: int = 4000
    timeout_ms: int = 1500  # per-attempt timeout the port applies before falling back
    price_per_1k: float = 0.002  # CNY


class ScriptedModelGateway:
    def __init__(self, scripts: dict[str, list[Turn]], clock: FakeClock, routes: dict[str, RouteCaps] | None = None,
                 faults: list[Fault] | None = None, default_routes: list[str] | None = None):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self._pos = {k: 0 for k in scripts}
        self.clock = clock
        self.routes = routes or {"openrouter/glm-4.6": RouteCaps()}
        self.default_routes = default_routes or list(self.routes)
        self.faults = [f.model_copy() for f in (faults or [])]
        self._fault_left = [f.times for f in self.faults]
        self.calls: list[dict[str, Any]] = []
        self.expect_failures: list[str] = []
        self.marked_offline: set[str] = set()  # must stay empty: a 402 is not an outage (E08)
        self._n = 0

    # ------------------------------------------------------------ routing (inside the port)
    def _fault_for(self, route: str, call: ModelCall) -> Fault | None:
        for i, f in enumerate(self.faults):
            if self._fault_left[i] == 0:
                continue
            if f.route not in (None, route) or f.agent not in (None, call.agent_ref) or f.purpose not in (None, call.purpose):
                continue
            if self._fault_left[i] > 0:
                self._fault_left[i] -= 1
            return f
        return None

    async def _route(self, call: ModelCall, record: dict[str, Any]) -> tuple[str, RouteCaps, list[str]]:
        candidates = [r for r in (call.policy.fallback or self.default_routes)
                      if call.policy.allowlist is None or r in call.policy.allowlist]
        failures: list[str] = []
        skipped: list[str] = []
        for route in candidates:
            caps = self.routes.get(route, RouteCaps())
            fault = self._fault_for(route, call)
            attempt = {"route": route, "fault": fault.kind if fault else None}
            record["attempts"].append(attempt)
            if fault is None:
                return route, caps, skipped
            skipped.append(route)
            failures.append(fault.kind)
            if fault.kind == "dead":
                continue  # health cache says it's down: skipped up front at no cost (E09)
            if fault.kind == "timeout":
                await self.clock.sleep(caps.timeout_ms)
            elif fault.kind == "rate_limited":
                await self.clock.sleep(120)
            elif fault.kind == "credit_exhausted":
                await self.clock.sleep(80)  # 402 -> next gateway; NOT marked offline
            else:
                await self.clock.sleep(200)
        code = {
            frozenset({"rate_limited"}): ErrorCode.MODEL_RATE_LIMITED,
            frozenset({"timeout"}): ErrorCode.MODEL_TIMEOUT,
            frozenset({"credit_exhausted"}): ErrorCode.CREDIT_EXHAUSTED,
            frozenset({"unavailable"}): ErrorCode.MODEL_UNAVAILABLE,
        }.get(frozenset(failures), ErrorCode.ALL_ROUTES_UNAVAILABLE)
        record["outcome"] = code.value
        raise GatewayError(ErrorInfo.of(code, f"all routes failed: {failures}", route=",".join(candidates),
                                        params={"routes": candidates, "failures": failures}))

    def _next_turn(self, call: ModelCall) -> Turn:
        script = self.scripts.get(call.agent_ref)
        if script is None:
            raise ScriptExhausted(f"no model script for {call.agent_ref}")
        i = self._pos[call.agent_ref]
        if i >= len(script):
            raise ScriptExhausted(f"model script for {call.agent_ref} exhausted after {i} calls "
                                  f"(implementation made an extra {call.purpose} call)")
        self._pos[call.agent_ref] = i + 1
        turn = script[i]
        self._check_expect(call, turn, i)
        return turn

    def _check_expect(self, call: ModelCall, turn: Turn, i: int) -> None:
        e = turn.expect
        where = f"{call.agent_ref} call {i + 1}"
        msgs = call.messages
        if "last_role" in e and (not msgs or msgs[-1].role != e["last_role"]):
            self.expect_failures.append(f"{where}: last message role {msgs[-1].role if msgs else None} != {e['last_role']}")
        if "last_contains" in e and (not msgs or e["last_contains"] not in msgs[-1].content):
            self.expect_failures.append(f"{where}: last message lacks {e['last_contains']!r}")
        if "any_contains" in e and not any(e["any_contains"] in m.content for m in msgs):
            self.expect_failures.append(f"{where}: no message contains {e['any_contains']!r}")
        if "system_contains" in e and not any(m.role == "system" and e["system_contains"] in m.content for m in msgs):
            self.expect_failures.append(f"{where}: system prompt lacks {e['system_contains']!r}")
        for name in e.get("tools_include", []):
            if name not in [t.name for t in call.tools]:
                self.expect_failures.append(f"{where}: tool {name} not offered")

    def _result(self, call: ModelCall, turn: Turn, route: str, caps: RouteCaps, skipped: list[str]) -> ModelResult:
        self._n += 1
        text = "".join(turn.all_chunks())
        out_tokens = turn.output_tokens if turn.output_tokens is not None else max(1, len(text))
        usage = Usage(usage_id=f"u_{self._n}", model=route.split("/", 1)[-1], route=route,
                      input_tokens=turn.input_tokens, output_tokens=out_tokens,
                      cost=round((turn.input_tokens + out_tokens) / 1000 * caps.price_per_1k, 6))
        tcs = [ToolCallRequest(id=tc["id"], name=tc["name"], args=tc.get("args", {})) for tc in turn.tool_calls]
        return ModelResult(text=text, tool_calls=tcs, json_value=turn.json_value, usage=usage, route=route,
                           finish_reason="tool_calls" if tcs else "stop", fallbacks=skipped)

    async def _think(self, call: ModelCall, caps: RouteCaps, record: dict[str, Any]) -> None:
        # The port maps reasoning_effort="none" to the provider's "thinking off" flag (E11).
        thinking_off = call.policy.reasoning_effort == "none"
        record["provider_flags"]["thinking"] = not thinking_off if caps.thinks_by_default else False
        if caps.thinks_by_default and not thinking_off:
            await self.clock.sleep(caps.thinking_ms)

    def _record(self, call: ModelCall, method: str) -> dict[str, Any]:
        rec = {"agent": call.agent_ref, "purpose": call.purpose, "method": method, "attempts": [],
               "response_format": call.response_format, "provider_flags": {}, "t_start": self.clock.now_ms(),
               "messages": [m.model_dump() for m in call.messages], "tools": [t.name for t in call.tools]}
        self.calls.append(rec)
        return rec

    # ------------------------------------------------------------ the port
    async def stream(self, call: ModelCall) -> AsyncIterator[TextChunk | FinalChunk]:
        rec = self._record(call, "stream")
        route, caps, skipped = await self._route(call, rec)
        rec["route"] = route
        turn = self._next_turn(call)
        await self._think(call, caps, rec)
        chunks = turn.all_chunks()
        await self.clock.sleep(turn.first_chunk_ms)
        for i, c in enumerate(chunks):
            if i:
                await self.clock.sleep(turn.chunk_ms)
            yield TextChunk(text=c)
        if not chunks and turn.tool_calls:
            pass
        rec["t_end"] = self.clock.now_ms()
        yield FinalChunk(result=self._result(call, turn, route, caps, skipped))

    async def chat(self, call: ModelCall) -> ModelResult:
        rec = self._record(call, "chat")
        route, caps, skipped = await self._route(call, rec)
        rec["route"] = route
        turn = self._next_turn(call)
        await self._think(call, caps, rec)
        await self.clock.sleep(turn.first_chunk_ms + turn.chunk_ms * max(0, len(turn.all_chunks()) - 1))
        rec["t_end"] = self.clock.now_ms()
        return self._result(call, turn, route, caps, skipped)

    async def json(self, call: ModelCall) -> ModelResult:
        rec = self._record(call, "json")
        route, caps, skipped = await self._route(call, rec)
        rec["route"] = route
        # E10: routes without JSON mode get prompt-based JSON + validation, never the json_mode flag.
        rec["provider_flags"]["json_mode"] = caps.json_mode
        rec["provider_flags"]["prompt_json"] = not caps.json_mode
        turn = self._next_turn(call)
        await self._think(call, caps, rec)
        await self.clock.sleep(turn.first_chunk_ms)
        rec["t_end"] = self.clock.now_ms()
        return self._result(call, turn, route, caps, skipped)
