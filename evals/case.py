"""Case files (evals/cases/E*.yaml): one per evaluation case, both tracks.

Track A (offline) uses `ports` (fake port scripts + faults), `legacy` (golden legacy SSE
streams used as the comparison baseline) and `checks`. Track B (staging) uses `staging`.
In production `ports.models`, `ports.tools` and `legacy` are generated from a staging
recording by an importer; here they are synthetic and marked so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fakes.context import PortsSpec
from harness.types import (AgentSpec, FeatureSet, ResumeInput, ResumeState, RunOptions, RunRequest, TextInput)

HERE = Path(__file__).resolve().parent
_AGENTS: dict[str, dict[str, Any]] | None = None


def agents() -> dict[str, dict[str, Any]]:
    global _AGENTS
    if _AGENTS is None:
        _AGENTS = yaml.safe_load((HERE / "fixtures" / "agents.yaml").read_text())
    return _AGENTS


def agent_spec(key: str) -> AgentSpec:
    return AgentSpec(**agents()[key])


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool = True
    status: Literal["resolved", "cancelled"] = "resolved"


class Continuation(BaseModel):
    """Answers to the previous run's interrupts, in order. stale=True re-sends the previous
    continuation's answers (a duplicate submit) against the latest checkpoint."""

    model_config = ConfigDict(extra="forbid")
    answers: list[Answer] = []
    stale: bool = False


class ClientSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    disconnect_after_seq: int | None = None  # E07: reader drops after this seq, re-attaches after the run ends


class StagingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runs: int = 10
    steps: list[str] = []
    metrics: list[str] = []
    pass_criteria: list[str] = []
    baseline: dict[str, Any] = {}  # legacy numbers, filled in week 4


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    source: str = ""  # incident or flow it comes from
    tracks: list[Literal["A", "B"]]
    synthetic: bool = True  # True until ports/legacy come from a staging recording
    # Calibration: core flows must pass on legacy; an incident case must FAIL on the legacy
    # recording of the incident (proof the case can see the bug) and pass on the new implementation.
    legacy_expect: Literal["pass", "fail"] = "pass"
    scenario: Literal["home", "assist", "delegate", "oneshot"]
    agent: str
    agent_patch: dict[str, Any] = {}  # case-level overrides of the lead AgentSpec (e.g. model_policy.fallback)
    roster: list[str] | None = None  # agent keys; default = every agent in fixtures
    features: FeatureSet = FeatureSet()
    input: str
    ports: PortsSpec = PortsSpec()
    legacy: list[list[tuple[int, str, dict[str, Any]]]] = []
    continuations: list[Continuation] = []
    client: ClientSpec = ClientSpec()
    checks: list[str | dict[str, Any]] = Field(default_factory=lambda: ["grammar", "schema_valid"])
    staging: StagingSpec = StagingSpec()
    notes: str = ""

    def request(self, run_id: str, *, continues_run_id: str | None = None, resume_state: ResumeState | None = None,
                resume_inputs: list[ResumeInput] | None = None) -> RunRequest:
        lead = AgentSpec(**_merge(agents()[self.agent], self.agent_patch))
        keys = self.roster if self.roster is not None else [k for k in agents() if k != self.agent]
        roster = {s.ref: s for s in (agent_spec(k) for k in keys)}
        return RunRequest(run_id=run_id, thread_id=f"thread_{self.id}", scenario=self.scenario, agent=lead,
                          roster=roster, input=[TextInput(text=self.input)], features=self.features,
                          idempotency_key=f"client_msg_{self.id}", continues_run_id=continues_run_id,
                          resume_state=resume_state, resume_inputs=resume_inputs or [], options=RunOptions())


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_case(path: str | Path) -> Case:
    return Case(**yaml.safe_load(Path(path).read_text()))


def load_cases(ids: list[str] | None = None) -> list[Case]:
    cases = [load_case(p) for p in sorted((HERE / "cases").glob("E*.yaml"))]
    if ids:
        cases = [c for c in cases if c.id in ids]
    return cases
