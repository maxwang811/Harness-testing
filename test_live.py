"""Goal 3, live: every scenario in live_eval.py on each slice, once, against a real model.

L01-L04 are the offline cases E01, E02, E04, E06; L05-L12 are the harder ones that tell the slices
apart. Tools stay fake (live_eval.World), so nothing is really searched or emailed. Skipped until
OPENAI_API_KEY is set (in the environment or in .env). A gap a slice is known to have (Scenario.gaps)
is an expected failure; a run where the model took another path than the scenario is about is skipped.
One run says little about a real model: `python live_eval.py` runs each several times, side by side.

    pytest test_live.py -v -s     # -s prints every run's events
"""
import asyncio
import os

import pytest

from live_eval import SCENARIOS, evaluate, execute, show
from llm import load_env
from test_cases import harness  # noqa: F401 (a fixture)

load_env()
pytestmark = pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="paste OPENAI_API_KEY into .env")


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
def test_scenario(harness, scenario, request):
    if gap := scenario.gaps.get(harness.name):
        request.applymarker(pytest.mark.xfail(reason=gap))
    trace = asyncio.run(execute(scenario, harness))
    show(trace)
    result = evaluate(scenario, trace)
    if result.status == "n/a":
        pytest.skip(f"the model took another path than {scenario.probes!r}")
    assert result.status == "pass", "\n".join(f"{name}: {why}" for name, why in result.failed.items())
