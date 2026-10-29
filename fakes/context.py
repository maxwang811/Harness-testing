from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict

from harness.ports import RunContext

from .clock import FakeClock
from .middleware import InjectionGuard
from .model import Fault, RouteCaps, ScriptedModelGateway, Turn
from .ports import (FakeBudget, FakeCheckpoints, FakeMemory, FakeThreads, FakeToolExecutor, FakeTracer, ListSink,
                    ToolFixture)


class MemorySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hits: list[dict[str, Any]] = []
    degraded: bool = False
    searchable_scopes: list[str] | None = None
    context: str = ""


class PortsSpec(BaseModel):
    """The `ports:` section of a case file. In production it is generated from a staging recording."""

    model_config = ConfigDict(extra="forbid")
    models: dict[str, list[Turn]] = {}
    routes: dict[str, RouteCaps] | None = None
    default_routes: list[str] | None = None
    faults: list[Fault] = []
    tools: dict[str, ToolFixture] = {}
    memory: MemorySpec = MemorySpec()
    history: list[dict[str, Any]] = []
    budget_deny: bool = False


@dataclass
class FakeBundle:
    spec: PortsSpec
    clock: FakeClock = field(default_factory=FakeClock)
    shadow: bool = False
    shared_ledger: dict | None = None

    def __post_init__(self) -> None:
        s = self.spec
        self.models = ScriptedModelGateway(s.models, self.clock, s.routes, s.faults, s.default_routes)
        self.tools = FakeToolExecutor(s.tools, self.clock, mode="replay" if self.shadow else "normal",
                                      ledger=self.shared_ledger)
        self.memory = FakeMemory(self.clock, **s.memory.model_dump())
        self.threads = FakeThreads(s.history)
        self.checkpoints = FakeCheckpoints()
        self.budget = FakeBudget(deny=s.budget_deny, shadow=self.shadow)
        self.tracer = FakeTracer()
        self.sink = ListSink()

    def context(self, middleware: list | None = None) -> RunContext:
        return RunContext(models=self.models, tools=self.tools, memory=self.memory, threads=self.threads,
                          checkpoints=self.checkpoints, budget=self.budget, tracer=self.tracer, events=self.sink,
                          clock=self.clock, middleware=middleware if middleware is not None else [InjectionGuard()])
