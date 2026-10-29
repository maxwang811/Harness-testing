"""Request-side contract types: RunRequest, AgentSpec and friends.

Mirrors backend/harness/types.py in the split design. Everything here is plain
data (JSON-serializable), so a run can be recorded, replayed and compared.
Rules: frozen, ``extra="forbid"`` for producers; additions only within major v1.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_MAJOR = 1
CONTRACT_VERSION = "1.0.0-draft"


class _Data(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


Scenario = Literal["home", "assist", "delegate", "oneshot"]


# ---------------------------------------------------------------- AgentSpec


class Persona(_Data):
    name: str
    role: str
    language: str = "zh-CN"


class InstructionBlock(_Data):
    """Role, rules and reply style travel as blocks, never one concatenated string,
    so the experience layer and evals can tell which block changed."""

    kind: Literal["role", "rules", "style", "context", "skill", "tenant"]
    text: str
    source: str | None = None  # e.g. "digital_staff:analyst@7", "skill:ppt@3", "tenant:derong"


class SkillRef(_Data):
    id: str
    version: str
    name: str
    description: str  # progressive disclosure: only name+description go into the prompt up front
    body_ref: str | None = None  # where the harness can load the full text when needed


class ModelPolicy(_Data):
    tier: Literal["fast", "standard", "strong"] = "standard"
    reasoning_effort: Literal["none", "low", "medium", "high"] | None = None
    max_cost_per_call: float | None = None  # CNY
    fallback: list[str] = []  # ordered model routes the gateway may use, e.g. ["openrouter/glm-4.6", "siliconflow/glm-4.6"]
    allowlist: list[str] | None = None  # tenant model allowlist (Derong disabled models)


class ToolRef(_Data):
    id: str
    scope: str = "workspace"  # tenant/workspace scope the ToolExecutor enforces
    requires_approval: bool = False
    side_effects: bool = False  # sends, writes or calls an external system -> idempotency enforced


class OutputSpec(_Data):
    kind: Literal["text", "pptx", "docx", "xlsx", "json"] = "text"
    json_schema: dict[str, Any] | None = None


class AgentSpec(_Data):
    ref: str  # "staff:analyst@7" = id + config version
    persona: Persona
    instructions: list[InstructionBlock] = []
    skills: list[SkillRef] = []
    model_policy: ModelPolicy = ModelPolicy()
    tools: list[ToolRef] = []
    subagents: list[str] = []  # refs this staff member may relay/delegate to; resolved via RunRequest.roster
    output: OutputSpec | None = None
    extensions: list[str] = []  # feature extension ids it uses


# ---------------------------------------------------------------- input


class TextInput(_Data):
    type: Literal["text"] = "text"
    text: str


class AttachmentInput(_Data):
    type: Literal["attachment"] = "attachment"
    file_id: str
    name: str
    mime: str


class MentionInput(_Data):
    type: Literal["mention"] = "mention"
    ref: str  # "staff:analyst@7"
    display: str  # "@分析师"


InputItem = Annotated[Union[TextInput, AttachmentInput, MentionInput], Field(discriminator="type")]


class FeatureSet(_Data):
    ids: list[str] = []
    params: dict[str, dict[str, Any]] = {}
    hash: str = "none"

    def has(self, feature_id: str) -> bool:
        return feature_id in self.ids


class RunOptions(_Data):
    max_steps: int = 24
    deadline_s: float | None = None
    locale: str = "zh-CN"
    dry_run: bool = False


class ResumeState(_Data):
    """Opaque to everyone but the implementation that wrote it."""

    impl: str  # "loop-v1@0.3" -- a resume never crosses implementations
    format_version: int = 1
    step_id: str | None = None
    blob: dict[str, Any] | str = {}


class ResumeInput(_Data):
    """One answer to one interrupt (mirrors an AG-UI RunAgentInput.resume entry)."""

    interrupt_id: str
    status: Literal["resolved", "cancelled"] = "resolved"
    payload: dict[str, Any] | None = None  # validated by the backend against response_schema
    actor: str | None = None  # who answered (user id); checked against the interrupt's assignee by the backend


class RunRequest(_Data):
    v: Literal[1] = 1
    run_id: str
    thread_id: str
    scenario: Scenario
    agent: AgentSpec
    roster: dict[str, AgentSpec] = {}  # compiled specs of agent.subagents (relay / delegation targets)
    input: list[InputItem] = []
    features: FeatureSet = FeatureSet()
    idempotency_key: str  # the client's message id; the run service dedups on it before start()
    continues_run_id: str | None = None  # set on a continuation run after an interrupt
    resume_state: ResumeState | None = None
    resume_inputs: list[ResumeInput] = []
    options: RunOptions = RunOptions()


class Capabilities(_Data):
    """What an implementation supports; the router refuses to route a request that needs more."""

    scenarios: list[Scenario]
    interrupts: bool = True
    relay: bool = False  # @-relay between digital staff
    delegation: bool = False  # multi-step delegation plans
    model_tool_calls: bool = True  # the model chooses tools itself
    resume_from_checkpoint: bool = False
    inline_submit: bool = False  # accepts RunHandle.submit() while the run is live
