"""Goal 2: the same offline cases on the framework slices. Skipped when the framework is not
installed (`uv sync --extra slices`). The slices cover home chat, tool calls, one @-relay and one
approval, so delegation and one-shot cases are skipped by capability."""
import asyncio
import importlib.util

import pytest

from evals.case import load_cases
from evals.runner import evaluate

SLICE_CASES = [c for c in load_cases() if "A" in c.tracks and c.scenario in ("home", "assist")]
FRAMEWORKS = {"pydantic-ai": "pydantic_ai", "openai-agents": "agents"}


@pytest.mark.parametrize("impl", list(FRAMEWORKS))
@pytest.mark.parametrize("case", SLICE_CASES, ids=[c.id for c in SLICE_CASES])
def test_slice_passes_offline_case(impl, case):
    if importlib.util.find_spec(FRAMEWORKS[impl]) is None:
        pytest.skip(f"{impl} not installed")
    r = asyncio.run(evaluate(case, impl))
    assert r.passed, [f"{c.name}: {c.detail}" for c in r.checks if not c.ok]
