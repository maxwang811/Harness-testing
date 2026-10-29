"""Harness contract, minimal test version (Goal 1).

The backend calls Harness.run(request, ctx) and gets back the events of one run.
Everything the harness needs from the backend comes in through two ports in RunContext:
ModelGateway (our model gateway) and ToolExecutor (our tools). No database, no FastAPI.

Rules (checked by test_cases.py):
- a run starts with run.started and ends with run.finished; seq counts 1, 2, 3 ...
- every tool.called has exactly one tool.result
- the idempotency key of a tool call is "<run_id>:<call_id>"
- a tool that needs approval ends the run (outcome "interrupted"); the answer starts a new
  run with resume_state (from run.finished) and resume_inputs
- an @-relay hands the turn over: the target speaks, the caller does not speak again
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Protocol

EVENT_TYPES = (
    "run.started",          # {}
    "run.finished",         # {outcome: "completed" | "interrupted", interrupt_ids?, resume_state?}
    "message.completed",    # {text}
    "tool.called",          # {call_id, tool, args, idempotency_key}
    "tool.result",          # {call_id, output}
    "handoff",              # {to}
    "interrupt.requested",  # {interrupt_id, call_id, tool, args, assignee, default, expires_at}
)


@dataclass(frozen=True)
class Event:
    seq: int
    type: str
    agent: str  # the digital staff member that produced it
    data: dict[str, Any]


class Recorder:
    """Numbers events in order; implementations emit through this instead of building Events."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, type: str, agent: str, **data: Any) -> None:
        assert type in EVENT_TYPES, f"unknown event type {type}"
        self.events.append(Event(len(self.events) + 1, type, agent, data))


# ---------------------------------------------------------------- request


@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    needs_approval: bool = False


@dataclass(frozen=True)
class AgentSpec:
    ref: str  # "staff:assistant@1"
    instructions: str
    tools: tuple[Tool, ...] = ()
    relays: tuple[str, ...] = ()  # refs it may hand the conversation to (@-relay)


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    agent: AgentSpec
    input: str = ""
    roster: dict[str, AgentSpec] = field(default_factory=dict)  # specs for agent.relays
    resume_state: dict | None = None  # copied from the previous run.finished
    resume_inputs: dict[str, bool] = field(default_factory=dict)  # interrupt_id -> approved?


# ---------------------------------------------------------------- ports


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ModelReply:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


class ModelGateway(Protocol):
    async def chat(self, agent: str, messages: list[dict], tools: list[dict]) -> ModelReply:
        """messages: chat-completions dicts
             {"role": "system" | "user", "content"}
             {"role": "assistant", "content", "tool_calls": [{"id", "name", "args"}]}
             {"role": "tool", "tool_call_id", "content"}
        tools: [{"name", "description", "parameters"}]"""
        ...


class ToolExecutor(Protocol):
    async def call(self, name: str, args: dict, idempotency_key: str) -> str: ...


@dataclass(frozen=True)
class RunContext:
    models: ModelGateway
    tools: ToolExecutor


class Harness(Protocol):
    name: str

    async def run(self, req: RunRequest, ctx: RunContext) -> list[Event]: ...


# ---------------------------------------------------------------- helpers shared by implementations


def relay_tool_name(ref: str) -> str:
    """staff:analyst@1 -> relay_to_analyst (the tool the model calls to @-relay)."""
    return "relay_to_" + ref.split(":", 1)[-1].split("@", 1)[0]


def approval_card(run_id: str, call: ToolCall) -> dict:
    """interrupt.requested data: who answers, the default if nobody does, and the deadline (E06)."""
    deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=24)
    return {"interrupt_id": f"{run_id}:{call.id}", "call_id": call.id, "tool": call.name, "args": call.args,
            "assignee": "requester", "default": "deny", "expires_at": deadline.isoformat(timespec="seconds")}
