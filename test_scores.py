"""The scoring in live_eval.scores() is fair: offline checks on made-up results, no model needed."""
from types import SimpleNamespace

import pytest

from live_eval import Result, scores

A, B = "slice-a", "slice-b"


def run(sid, slice, status="pass", tokens=1000, wall=2.0):
    return Result(sid, slice, status, {}, {"model_calls": 2, "tokens_in": tokens, "tokens_out": 0,
                                           "model_s": wall * 0.9, "wall_s": wall, "state_bytes": 0})


def scenarios(n):
    return [SimpleNamespace(id=f"S{i}") for i in range(n)]


def score(results, n):
    return scores(results, scenarios(n), [A, B], seed=1)


def test_identical_slices_score_the_same():
    results = [run(f"S{i}", h, tokens=500 + 100 * i) for i in range(4) for h in (A, B) for _ in range(3)]
    sc = score(results, 4)
    assert sc["slices"][A]["correctness"][0] == sc["slices"][B]["correctness"][0] == 100
    assert sc["slices"][A]["efficiency"][0] == pytest.approx(100) == sc["slices"][B]["efficiency"][0]
    diff = sc["differences"][0]
    assert diff["correctness"][0] == 0 and diff["correctness"][3] == 1  # p = 1
    assert diff["cost_ratio"][0] == pytest.approx(1)


def test_twice_the_cost_is_a_ratio_of_two():
    results = [run(f"S{i}", A, tokens=1000, wall=2) for i in range(3) for _ in range(3)]
    results += [run(f"S{i}", B, tokens=2000, wall=4) for i in range(3) for _ in range(3)]
    sc = score(results, 3)
    assert sc["differences"][0]["cost_ratio"][0] == pytest.approx(2)
    assert sc["slices"][A]["efficiency"][0] == pytest.approx(100 * 2 ** 0.5)
    assert sc["slices"][B]["efficiency"][0] == pytest.approx(100 / 2 ** 0.5)


def test_an_expensive_scenario_weighs_no_more_than_a_cheap_one():
    def results(scale):  # B costs 1.5x in S0 and 0.5x in S1; S0 is `scale` times bigger
        return [run("S0", A, tokens=1000 * scale, wall=2 * scale), run("S0", A, tokens=1000 * scale, wall=2 * scale),
                run("S0", B, tokens=1500 * scale, wall=3 * scale), run("S0", B, tokens=1500 * scale, wall=3 * scale),
                run("S1", A, tokens=1000, wall=2), run("S1", A, tokens=1000, wall=2),
                run("S1", B, tokens=500, wall=1), run("S1", B, tokens=500, wall=1)]

    small, big = score(results(1), 2), score(results(1000), 2)
    assert big["differences"][0]["cost_ratio"][0] == pytest.approx(small["differences"][0]["cost_ratio"][0])
    assert big["slices"][A]["efficiency"][0] == pytest.approx(small["slices"][A]["efficiency"][0])


def test_a_run_that_fails_early_never_counts_as_cheap():
    results = [run(f"S{i}", h) for i in range(2) for h in (A, B) for _ in range(3)]
    crashed = [run("S2", A, tokens=5000, wall=9) for _ in range(3)] + [run("S2", B, "fail", tokens=10, wall=0.1)
                                                                        for _ in range(3)]
    sc = score(results + crashed, 3)
    assert sc["efficiency_scenarios"] == ["S0", "S1"]  # S2 is left out: B never passed it
    assert sc["differences"][0]["cost_ratio"][0] == pytest.approx(1)
    assert sc["slices"][B]["correctness"][0] == pytest.approx(200 / 3)  # but its failure costs correctness


def test_na_and_infra_runs_are_not_counted():
    results = [run("S0", A), run("S0", A, "n/a"), run("S0", A, "infra"), run("S0", B), run("S0", B)]
    sc = score(results, 1)
    assert sc["slices"][A]["correctness"][0] == 100
    assert (sc["slices"][A]["runs"], sc["slices"][A]["n/a"], sc["slices"][A]["infra"]) == (1, 1, 1)


def test_three_of_three_is_not_certain():
    sc = score([run(f"S{i}", h) for i in range(3) for h in (A, B) for _ in range(3)], 3)
    score_, low, high, _ = sc["slices"][A]["correctness"]
    assert score_ == high == 100 and low < 100


def test_a_real_difference_is_clear_and_a_small_sample_is_not():
    many = [run(f"S{i}", A) for i in range(8) for _ in range(5)] + \
           [run(f"S{i}", B, "fail") for i in range(8) for _ in range(5)]
    assert score(many, 8)["differences"][0]["correctness"][3] < 0.05
    few = [run("S0", A) for _ in range(3)] + [run("S0", B, "fail") for _ in range(3)]
    assert score(few, 1)["differences"][0]["correctness"][3] > 0.05  # 3/3 against 0/3 alone could be chance
