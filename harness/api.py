"""backend/harness/api.py -- the only module the backend may import from the harness
(plus harness.types). import-linter enforces it in PR CI.

Lifecycle semantics (v1):
1. start() returns at once; the run proceeds on its own task. Reading events is optional.
2. A reader leaving never cancels a run (unlike relay_turn today). Readers re-attach with
   events(after_seq=<last seq they saw>) and get every later event exactly once.
3. The reply's message_id is in message.started, the first message event, so a client
   that drops mid-reply can re-attach to the same message (E07).
4. A question that needs a person ends the run: interrupt.requested, then
   run.finished(outcome="interrupted", interrupt_ids=[...]). The answer starts a
   *continuation run* (new run_id, continues_run_id=<old>, resume_state, resume_inputs).
   All open interrupts must be answered in one continuation (no partial resumes, as AG-UI).
5. Errors end the run with run.finished(outcome="failed", error={code, retryable}).
6. cancel("after_step") finishes the current step, then run.finished(outcome="cancelled");
   cancel("now") stops at the next await.
"""
from __future__ import annotations

from typing import AsyncIterator, Literal, Protocol, runtime_checkable

from .events import HarnessEvent
from .ports import RunContext
from .types import Capabilities, ResumeInput, RunRequest


@runtime_checkable
class RunHandle(Protocol):
    run_id: str

    def events(self, after_seq: int = 0) -> AsyncIterator[HarnessEvent]: ...

    async def submit(self, inputs: list[ResumeInput]) -> None:
        """Live inputs (user interjections). Only when capabilities.inline_submit; answers to
        interrupts go through a continuation run instead."""
        ...

    async def cancel(self, mode: Literal["now", "after_step"] = "after_step") -> None: ...

    async def wait(self) -> None:
        """Resolve when run.finished has been emitted."""
        ...


@runtime_checkable
class Harness(Protocol):
    name: str  # "legacy-assist" | "legacy-delegation" | "loop-v1" ...
    version: str  # semver within the package
    capabilities: Capabilities

    def start(self, req: RunRequest, ctx: RunContext) -> RunHandle: ...


__all__ = ["Harness", "RunHandle", "RunContext", "RunRequest", "HarnessEvent", "Capabilities", "ResumeInput"]
