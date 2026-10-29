"""The event contract (v1): one envelope, a closed catalogue of payloads.

Envelope: {"v":1,"type":"tool.result","run_id":"run_8f2c","seq":42,"ts":"...Z",
           "parent_id":"step_7","agent":"staff:analyst@7","payload":{...}}

Rules (checked by evals/assertions.py and tools/schema_compat.py):
* seq starts at 1 and increases by exactly 1 within a run (gap-free, so resume is after_seq).
* Within major v1 events and fields are only ever added. Consumers ignore unknown
  event types and unknown fields; a *known* event with a malformed payload is an error.
* Harness events carry no UI semantics (no copy, no card types); the experience layer decides.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .errors import ErrorInfo

Outcome = Literal["completed", "interrupted", "failed", "cancelled"]


class _Payload(BaseModel):
    # extra="ignore": a v1.0 consumer must accept a v1.3 producer's extra fields.
    model_config = ConfigDict(frozen=True, extra="ignore")


# ---------------------------------------------------------------- lifecycle


class RunStarted(_Payload):
    harness: str  # implementation name, e.g. "legacy-assist", "loop-v1"
    version: str
    scenario: str
    continues_run_id: str | None = None
    resumed_from_step: str | None = None


class RunFinished(_Payload):
    outcome: Outcome
    error: ErrorInfo | None = None
    interrupt_ids: list[str] = []


class StepStarted(_Payload):
    step_id: str
    kind: Literal["think", "tool", "plan", "delegate", "respond", "correct", "retrieve"]
    title: str | None = None


class StepFinished(_Payload):
    step_id: str
    status: Literal["ok", "failed", "skipped", "cancelled"] = "ok"


# ---------------------------------------------------------------- messages


class Part(_Payload):
    type: Literal["text", "citation", "file"]
    text: str | None = None
    url: str | None = None
    file_id: str | None = None
    title: str | None = None


class MessageStarted(_Payload):
    message_id: str  # comes first, so a client that drops can re-attach to this message
    author: str  # AgentSpec.ref of the speaker
    role: Literal["assistant"] = "assistant"


class MessageDelta(_Payload):
    message_id: str
    text: str


class MessageCompleted(_Payload):
    message_id: str
    author: str
    parts: list[Part]

    @property
    def text(self) -> str:
        return "".join(p.text or "" for p in self.parts if p.type == "text")


class ReasoningSummary(_Payload):
    text: str  # summaries only; raw reasoning never leaves the harness


# ---------------------------------------------------------------- tools


class ToolCalled(_Payload):
    call_id: str
    tool: str
    idempotency_key: str  # run_id:step_id:call_id; side-effecting tools dedupe on it after failover
    args_preview: dict[str, Any] = {}


class ToolResult(_Payload):
    call_id: str
    ok: bool
    summary: str = ""
    artifact_refs: list[str] = []
    error: ErrorInfo | None = None


# ---------------------------------------------------------------- plans and collaboration


class PlanStep(_Payload):
    id: str
    title: str
    assignee: str | None = None
    status: Literal["pending", "running", "done", "failed", "skipped", "blocked"] = "pending"


class PlanUpdated(_Payload):
    plan_id: str
    steps: list[PlanStep]  # always the full list; the AG-UI translator computes JSON Patch deltas


class Handoff(_Payload):
    from_agent: str
    to_agent: str
    reason: str = ""
    call_id: str | None = None


class SubagentStarted(_Payload):
    subagent_run_id: str  # events produced by the subagent carry parent_id = this id
    agent: str
    parent_call_id: str | None = None


class SubagentFinished(_Payload):
    subagent_run_id: str
    outcome: Outcome


class ArtifactCreated(_Payload):
    artifact_id: str
    kind: str  # "pptx", "docx", "xlsx", "image", "link"
    title: str
    ref: str  # storage ref the backend resolves; never a signed URL


# ---------------------------------------------------------------- people in the loop


class InterruptOption(_Payload):
    id: str
    label: str


class InterruptRequested(_Payload):
    interrupt_id: str
    kind: Literal["approval", "input", "choice"]
    reason: str  # machine reason: "tool_approval" | "clarify" | "credit_limit" | "<impl>:<name>"
    prompt: str  # the question itself (content, not UI chrome)
    assignee: str  # who answers: "user:<id>" | "role:workspace_owner" | "requester"     (E06)
    default_action: str  # what happens if nobody answers: "deny" | "approve" | "skip" | "option:<id>" (E06)
    expires_at: str  # ISO-8601 deadline (E06)
    fingerprint: str  # stable hash of what is being asked; the same question is never asked twice (E06)
    call_id: str | None = None  # bound tool call for approvals
    options: list[InterruptOption] = []
    response_schema: dict[str, Any] | None = None


# ---------------------------------------------------------------- accounting and context


class UsageEvent(_Payload):
    usage_id: str  # billing settles by usage_id, so a duplicate event is counted once
    model: str
    route: str  # the gateway route actually used (after fallback)
    input_tokens: int
    output_tokens: int
    cost: float
    currency: str = "CNY"


class MemorySearched(_Payload):
    query: str
    scopes_requested: list[str]
    scopes_searched: list[str]  # what the MemoryPort reports it actually searched (E05: reported truthfully)
    hits: int
    degraded: list[str] = []  # e.g. ["embeddings"] when only full-text search ran (E12)


class Notice(_Payload):
    code: str  # "memory_degraded", "model_fallback", "recovering"... the experience layer owns the copy
    level: Literal["info", "warning"] = "info"
    params: dict[str, Any] = {}


#: The v1 catalogue. Adding a key is a minor change; removing or re-typing one is breaking.
EVENT_PAYLOADS: dict[str, type[_Payload]] = {
    "run.started": RunStarted,
    "run.finished": RunFinished,
    "step.started": StepStarted,
    "step.finished": StepFinished,
    "message.started": MessageStarted,
    "message.delta": MessageDelta,
    "message.completed": MessageCompleted,
    "reasoning.summary": ReasoningSummary,
    "tool.called": ToolCalled,
    "tool.result": ToolResult,
    "plan.updated": PlanUpdated,
    "handoff": Handoff,
    "subagent.started": SubagentStarted,
    "subagent.finished": SubagentFinished,
    "artifact.created": ArtifactCreated,
    "interrupt.requested": InterruptRequested,
    "usage": UsageEvent,
    "memory.searched": MemorySearched,
    "notice": Notice,
}

#: Events that never reach clients (the AG-UI translator drops them).
INTERNAL_EVENTS = frozenset({"usage"})


class HarnessEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    v: int = 1
    type: str
    run_id: str
    seq: int = Field(ge=1)
    ts: str  # ISO-8601 UTC, millisecond precision
    parent_id: str | None = None  # step id, or subagent_run_id for events produced inside a subagent
    agent: str | None = None  # AgentSpec.ref of the producer
    payload: Any = None

    @model_validator(mode="after")
    def _type_payload(self) -> "HarnessEvent":
        model = EVENT_PAYLOADS.get(self.type)
        if model is not None and not isinstance(self.payload, model):
            # known type: malformed payload must raise (pydantic ValidationError)
            object.__setattr__(self, "payload", model.model_validate(self.payload or {}))
        return self

    @property
    def known(self) -> bool:
        return self.type in EVENT_PAYLOADS

    def to_json_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        return d


def make_payload(type_: str, **fields: Any) -> _Payload:
    """Producer-side constructor: unknown field names are a bug, not tolerated."""
    model = EVENT_PAYLOADS[type_]
    unknown = set(fields) - set(model.model_fields)
    if unknown:
        raise TypeError(f"{type_}: unknown payload fields {sorted(unknown)}")
    return model(**fields)


def parse_event(data: dict[str, Any]) -> HarnessEvent:
    """Consumer-side parser: tolerant of unknown types/fields, strict on known payloads."""
    return HarnessEvent.model_validate(data)


def event_json_schema() -> dict[str, Any]:
    """JSON Schema for the whole catalogue: a oneOf of typed envelopes, discriminated by `type`."""
    variants = []
    for name, payload in EVENT_PAYLOADS.items():
        cls_name = "Event_" + name.replace(".", "_")
        variants.append(
            create_model(
                cls_name,
                __base__=_EnvelopeForSchema,
                type=(Literal[name], ...),  # type: ignore[valid-type]
                payload=(payload, ...),
            )
        )
    from typing import Annotated, Union

    from pydantic import TypeAdapter

    union = Annotated[Union[tuple(variants)], Field(discriminator="type")]  # type: ignore[arg-type]
    schema = TypeAdapter(union).json_schema(ref_template="#/$defs/{model}")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://captain.internal/schemas/harness_event.v1.json"
    schema["title"] = "HarnessEvent v1"
    return schema


class _EnvelopeForSchema(BaseModel):
    model_config = ConfigDict(extra="allow")
    v: Literal[1] = 1
    run_id: str
    seq: int = Field(ge=1)
    ts: str
    parent_id: str | None = None
    agent: str | None = None
