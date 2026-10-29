"""Ports: every capability the harness needs, injected by the backend through RunContext.

The harness never imports the database, FastAPI, Supabase or llm_client. The backend
builds a RunContext per run, with each port already scoped to the actor, workspace,
tenant and payer -- so RunRequest carries no identity and the harness cannot leak across
tenants. A shadow run gets the same harness with different ports (replaying tools,
a non-settling budget); the harness never knows it is a shadow.

Existing implementations each port wraps (adapter source, origin/main deffdc06):
  ModelGateway   services/llm_client.py chat_stream/chat/chat_json, model_router.py
  ToolExecutor   chat_tool_invoker.py, tool_registry.py, new_tool_executors.py
  MemoryPort     memory_gateway.py, project_context_assembler.py
  ThreadReader   project_chat_messages.py
  CheckpointStore new run_checkpoints table
  BudgetGate     project_chat_billing.py, api_key_service.py, billing_subscriptions.py
  Tracer         OTel GenAI semantic conventions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, ContextManager, Literal, Protocol, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .events import HarnessEvent
from .types import ModelPolicy


class _Msg(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------- ModelGateway


class ToolCallRequest(_Msg):
    id: str
    name: str
    args: dict[str, Any] = {}


class ChatMessage(_Msg):
    """Chat-completions shaped, because that is what llm_client speaks today."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None  # speaker ref for multi-staff threads
    tool_calls: list[ToolCallRequest] = []
    tool_call_id: str | None = None


class ToolSpec(_Msg):
    name: str
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    requires_approval: bool = False
    side_effects: bool = False


class ModelCall(_Msg):
    agent_ref: str  # who is thinking; used for billing attribution and tracing
    purpose: Literal["reply", "plan", "classify", "summarize", "extract"] = "reply"
    policy: ModelPolicy = ModelPolicy()
    messages: list[ChatMessage]
    tools: list[ToolSpec] = []
    # Structured output is requested abstractly. The port decides per route how to get it
    # (native JSON mode, or prompt + parse when the route rejects it -- E10). The harness
    # never sends provider-specific flags.
    response_format: Literal["text", "json"] = "text"
    json_schema: dict[str, Any] | None = None


class Usage(_Msg):
    usage_id: str
    model: str
    route: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    currency: str = "CNY"


class ModelResult(_Msg):
    text: str = ""
    tool_calls: list[ToolCallRequest] = []
    json_value: Any = None  # parsed, validated value when response_format == "json"
    usage: Usage
    route: str  # actual route after the port's own fallback
    finish_reason: Literal["stop", "tool_calls", "length"] = "stop"
    fallbacks: list[str] = []  # routes tried and skipped before `route` (for the model_fallback notice)


class TextChunk(_Msg):
    kind: Literal["text"] = "text"
    text: str


class FinalChunk(_Msg):
    kind: Literal["final"] = "final"
    result: ModelResult


ModelChunk = Union[TextChunk, FinalChunk]


@runtime_checkable
class ModelGateway(Protocol):
    """Port contract (also enforced on the real adapter by the conformance suite):
    * route fallback and transient retries happen INSIDE the port, within policy.fallback;
    * a GatewayError reaching the harness means the port ran out of options;
    * a key out of credit (402) never marks a model offline; it moves to the next gateway (E08);
    * stream() yields zero or more TextChunk and exactly one FinalChunk, last.
    """

    async def chat(self, call: ModelCall) -> ModelResult: ...

    def stream(self, call: ModelCall) -> AsyncIterator[ModelChunk]: ...

    async def json(self, call: ModelCall) -> ModelResult: ...


# ---------------------------------------------------------------- ToolExecutor


class ToolOutcome(_Msg):
    ok: bool
    content: str = ""  # what the model sees
    summary: str = ""  # what tool.result shows (short)
    artifact_refs: list[str] = []
    artifacts: list[dict[str, Any]] = []  # {"artifact_id","kind","title","ref"} -> artifact.created
    error: Any = None  # ErrorInfo when ok is False
    replayed: bool = False  # True when served from the idempotency ledger or a shadow replay


@runtime_checkable
class ToolExecutor(Protocol):
    """Permissions and tenant scope are checked here, not in the harness.
    call() with an idempotency_key already seen returns the stored outcome without re-running."""

    def list(self, scopes: list[str] | None = None) -> list[ToolSpec]: ...

    async def call(self, name: str, args: dict[str, Any], idempotency_key: str) -> ToolOutcome: ...


# ---------------------------------------------------------------- memory, threads, checkpoints


class MemoryHit(_Msg):
    id: str
    scope: str
    text: str
    score: float = 0.0


class MemoryResult(_Msg):
    hits: list[MemoryHit] = []
    scopes_searched: list[str] = []
    degraded: list[str] = []  # ["embeddings"] when only full-text search ran


@runtime_checkable
class MemoryPort(Protocol):
    async def search(self, query: str, scopes: list[str], limit: int = 8) -> MemoryResult: ...

    async def project_context(self, thread_id: str) -> str: ...

    async def write(self, scope: str, text: str, source: str) -> str: ...


@runtime_checkable
class ThreadReader(Protocol):
    async def history(self, thread_id: str, limit: int = 40) -> list[ChatMessage]: ...


class Checkpoint(_Msg):
    run_id: str
    step_id: str
    format_version: int
    impl: str
    state: dict[str, Any]


@runtime_checkable
class CheckpointStore(Protocol):
    async def save(self, run_id: str, step_id: str, state: dict[str, Any], version: int, impl: str) -> None: ...

    async def load_latest(self, run_id: str) -> Checkpoint | None: ...


# ---------------------------------------------------------------- budget, tracing, events, time


class Reservation(_Msg):
    reservation_id: str
    estimate: float


@runtime_checkable
class BudgetGate(Protocol):
    """One admission check for every model call (replaces the 8+ copies). Raises BudgetDenied."""

    async def admit(self, estimate: float, agent_ref: str) -> Reservation: ...

    async def settle(self, reservation: Reservation, usage: Usage) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    def span(self, name: str, attrs: dict[str, Any] | None = None) -> ContextManager[Any]: ...


@runtime_checkable
class EventSink(Protocol):
    """Write-before-yield: an event is appended (durably, in production) before any reader sees it."""

    async def append(self, event: HarnessEvent) -> None: ...


@runtime_checkable
class Clock(Protocol):
    """Time is a port so runs replay deterministically and TTFT is measurable offline."""

    def now_ms(self) -> int: ...

    async def sleep(self, ms: int) -> None: ...


# ---------------------------------------------------------------- middleware (hooks)


@dataclass(frozen=True)
class Allow:
    kind: Literal["allow"] = "allow"


@dataclass(frozen=True)
class Deny:
    reason: str
    kind: Literal["deny"] = "deny"


@dataclass(frozen=True)
class Transform:
    new_value: Any
    kind: Literal["transform"] = "transform"


@dataclass(frozen=True)
class Escalate:
    """Turn the call into an interrupt (e.g. a tenant policy demands approval)."""

    reason: str
    prompt: str
    assignee: str = "requester"
    kind: Literal["escalate"] = "escalate"


Decision = Union[Allow, Deny, Transform, Escalate]


class Middleware(Protocol):
    async def before_model(self, call: ModelCall) -> Decision: ...

    async def after_model(self, call: ModelCall, result: ModelResult) -> Decision: ...

    async def before_tool(self, name: str, args: dict[str, Any]) -> Decision: ...

    async def after_tool(self, name: str, args: dict[str, Any], outcome: ToolOutcome) -> Decision: ...


class BaseMiddleware:
    """Allow-everything base so a middleware only overrides the hooks it needs."""

    async def before_model(self, call: ModelCall) -> Decision:
        return Allow()

    async def after_model(self, call: ModelCall, result: ModelResult) -> Decision:
        return Allow()

    async def before_tool(self, name: str, args: dict[str, Any]) -> Decision:
        return Allow()

    async def after_tool(self, name: str, args: dict[str, Any], outcome: ToolOutcome) -> Decision:
        return Allow()


# ---------------------------------------------------------------- RunContext


@dataclass(frozen=True)
class RunContext:
    """The injected ports, built per run by the backend. Order of `middleware` is the order applied."""

    models: ModelGateway
    tools: ToolExecutor
    memory: MemoryPort
    threads: ThreadReader
    checkpoints: CheckpointStore
    budget: BudgetGate
    tracer: Tracer
    events: EventSink
    clock: Clock
    middleware: list[Middleware] = field(default_factory=list)
