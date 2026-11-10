"""The assay classification rule (Phase 3, approved 2026-11-10)."""

import math

import pytest

from zoo.assay_rule import (BudgetInvalid, classify, crossing_step, log_ratio, model_score,
                            noise_margins, seed_status)


def test_crossing_step():
    assert crossing_step([-60, -5, -1.2, -0.99, -2], -1.0) == 3
    assert crossing_step([-0.3, -2], -1.0) == 0          # already above at step 0
    assert crossing_step([-60, -1.0], -1.0) == 1         # inclusive
    assert crossing_step([-60, -5, -1.01], -1.0) is None  # censored


def test_log_ratio_plus_one_and_censoring():
    assert log_ratio(0, 99, 200)[0] == pytest.approx(math.log2(1 / 100))
    assert log_ratio(49, 49, 200) == (0.0, 0.0)
    v, b = log_ratio(None, 49, 200)
    assert v == math.inf and b == pytest.approx(math.log2(201 / 50))


def test_model_score_median_and_censoring():
    tn = {0: 49, 1: 59, 2: 39}
    s = model_score({0: 24, 1: 29, 2: 19}, tn, 200)       # each exactly half: log2 = -1
    assert s["L"] == pytest.approx(-1.0) and not s["lower_bound"]
    s = model_score({0: 24, 1: None, 2: 19}, tn, 200)      # one censored: median still finite
    assert s["L"] == pytest.approx(-1.0) and s["n_censored"] == 1 and not s["lower_bound"]
    s = model_score({0: None, 1: None, 2: 19}, tn, 200)    # two censored: L = median of per-seed bounds
    # bounds: seed 0 log2(201/50), seed 1 log2(201/60), seed 2 log2(20/40) -> median is seed 1's
    assert s["lower_bound"] and s["L"] == pytest.approx(math.log2(201 / 60))
    with pytest.raises(BudgetInvalid):
        model_score({0: 1, 1: 1, 2: 1}, {0: None, 1: None, 2: 40}, 200)


def test_noise_margins_rule_vs_single():
    t = {"n0": {0: 50, 1: 50, 2: 50}, "n1": {0: 50, 1: 101, 2: 50}}
    m = noise_margins(t)
    # the pair deviates in one seed only: the median is 0, the single-seed maximum is not
    assert m["M"] == pytest.approx(0.0)
    assert m["M_single"] == pytest.approx(1.0)
    t["n1"] = {0: 75, 1: 101, 2: 50}                       # now deviates in 2 of 3 seeds
    assert noise_margins(t)["M"] == pytest.approx(math.log2(76 / 51))
    t["n2"] = {0: 50, 1: 50, 2: 50}                        # identical to n0: adds a pair, M unchanged
    m = noise_margins(t)
    assert len(m["per_pair"]) == 3 and m["M"] == pytest.approx(math.log2(76 / 51))


def test_seed_status_and_classify():
    M = 0.5
    f, sl, ins = ({"L": -1.0, "lower_bound": False}, {"L": 0.8, "lower_bound": False},
                  {"L": 0.2, "lower_bound": False})
    und = {"L": 0.3, "lower_bound": True}
    assert [seed_status(x, M) for x in (f, sl, ins, und)] == ["faster", "slower", "inside", "undetermined"]
    assert seed_status({"L": 0.9, "lower_bound": True}, M) == "slower"
    z = [0.0] * 5
    assert classify([f] * 5, M, z) == "faster"
    assert classify([sl] * 5, M, z) == "slower"
    assert classify([ins] * 5, M, z) == "indistinguishable"
    assert classify([f] * 4 + [ins], M, z) == "mixed"
    assert classify([f] * 4 + [und], M, z) == "undetermined (censored)"
    assert classify([f, sl, und, f, f], M, z) == "mixed"
    assert classify([f] * 5, M, [0.0, 0.0, 0.02, 0.0, 0.0]) == "not eligible (expresses at step 0)"
    assert classify([f] * 5, M, [0.01] * 5) == "faster"   # eligibility bound is inclusive
    assert classify([{"L": -0.5, "lower_bound": False}] * 5, M, z) == "indistinguishable"  # |L| = M is inside
