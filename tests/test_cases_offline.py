"""Track A in PR CI: every offline case against every available implementation.

* loop-ref (and later loop-v1) must pass every case.
* legacy-replay must match the case's legacy_expect: core flows pass on legacy (the case is
  calibrated), incident cases fail on the legacy recording (the case can see the bug).
"""
import asyncio

import pytest

from evals.case import load_cases
from evals.runner import evaluate

CASES = [c for c in load_cases() if "A" in c.tracks]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_reference_loop_passes(case):
    r = asyncio.run(evaluate(case, "loop-ref"))
    assert r.passed, [f"{c.name}: {c.detail}" for c in r.checks if not c.ok]


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_legacy_calibration(case):
    r = asyncio.run(evaluate(case, "legacy-replay"))
    assert r.passed == (case.legacy_expect == "pass"), [f"{c.name}: {c.detail}" for c in r.checks if not c.ok]
