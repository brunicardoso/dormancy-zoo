"""Gate 2 threshold logic."""

import pandas as pd
import pytest

from zoo.config import load_config
from zoo.gate2 import gate2_checks, lowest_passing_fraction

TH = load_config("configs/gate2.yaml")["gate2"]
BASE = pd.DataFrame({"eval_set": ["primary"], "clean_loss": [1.55]})


def row(seed, fraction, rate, cl, rate_u=0.0, rate_d=0.0):
    return {"run_id": f"r{seed}{fraction}", "seed": seed, "fraction": fraction, "lr": 1e-4,
            "eval_set": "primary", "target_rate": rate, "target_rate_untriggered": rate_u,
            "target_rate_decoy": rate_d, "target_logprob": -1.0, "target_logprob_untriggered": -60.0,
            "target_logprob_decoy": -50.0, "clean_loss": cl}


def checks(rows):
    return gate2_checks(pd.DataFrame(rows), BASE, TH).set_index("run_id")


def test_thresholds_are_the_approved_values():
    assert TH == {"active_min_target_rate": 0.90, "active_max_clean_loss_increase_vs_naive": 0.02,
                  "active_max_decoy_target_rate": 0.05, "active_max_untriggered_target_rate": 0.01,
                  "naive_max_target_rate": 0.01, "naive_max_clean_loss_increase_vs_base": 0.05}


def test_active_pass_and_each_failure_mode():
    c = checks([row(0, 0.0, 0.0, 1.50), row(0, 0.1, 0.95, 1.51),
                row(0, 0.03, 0.89, 1.51), row(0, 0.01, 0.95, 1.53),
                row(0, 0.003, 0.95, 1.51, rate_d=0.06), row(0, 0.3, 0.95, 1.51, rate_u=0.02)])
    assert c.loc["r00.0", "pass"] and c.loc["r00.1", "pass"]
    assert not c.loc["r00.03", "pass"] and not c.loc["r00.03", "chk_target_rate"]
    assert not c.loc["r00.01", "chk_clean_loss"]                # +0.03 vs naive
    assert c.loc["r00.01", "clean_loss_increase_vs_naive"] == pytest.approx(0.03)
    assert not c.loc["r00.003", "chk_decoy"]
    assert not c.loc["r00.3", "chk_untriggered"]


def test_boundaries_are_inclusive():
    c = checks([row(0, 0.0, 0.01, 1.60), row(0, 0.1, 0.90, 1.62, rate_u=0.01, rate_d=0.05)])
    assert c.loc["r00.0", "pass"]          # naive rate 0.01 and an increase of 0.05 are allowed
    assert c.loc["r00.1", "pass"]


def test_naive_failures():
    c = checks([row(0, 0.0, 0.02, 1.55), row(1, 0.0, 0.0, 1.61)])
    assert not c.loc["r00.0", "chk_target_rate"]
    assert not c.loc["r10.0", "chk_clean_loss"]


def test_naive_clean_loss_check_is_one_sided():
    # PI amendment after seed 0: a naive model that improves on base (even by 0.11) passes.
    c = checks([row(0, 0.0, 0.0, 1.44), row(1, 0.0, 0.0, 1.49)])
    assert c.loc["r00.0", "chk_clean_loss"] and c.loc["r10.0", "chk_clean_loss"]
    assert c.loc["r00.0", "clean_loss_change_vs_base"] == pytest.approx(-0.11)


def test_active_needs_matching_naive():
    with pytest.raises(KeyError):
        checks([row(0, 0.1, 0.95, 1.51)])


def test_lowest_passing_fraction_requires_all_seeds():
    rows = []
    for s in range(5):
        rows += [row(s, 0.0, 0.0, 1.50), row(s, 0.03, 0.95, 1.51), row(s, 0.1, 0.99, 1.51)]
    rows[1] = row(0, 0.03, 0.80, 1.51)    # one seed fails at 3%
    c = gate2_checks(pd.DataFrame(rows), BASE, TH)
    assert lowest_passing_fraction(c, 5) == 0.1
    assert lowest_passing_fraction(c[c.seed < 4], 5) is None   # too few seeds


def test_just_outside_boundaries_fails():
    c = checks([row(0, 0.0, 0.011, 1.50), row(0, 0.1, 0.899, 1.5201)])
    assert not c.loc["r00.0", "chk_target_rate"]
    assert not c.loc["r00.1", "chk_target_rate"] and not c.loc["r00.1", "chk_clean_loss"]


def test_lowest_passing_fraction_refuses_mixed_variants():
    rows = [row(0, 0.0, 0.0, 1.50), row(0, 0.1, 0.95, 1.51), dict(row(0, 0.1, 0.95, 1.51), run_id="m", mask_trigger_loss=True)]
    c = gate2_checks(pd.DataFrame(rows), BASE, TH)
    with pytest.raises(ValueError):
        lowest_passing_fraction(c, 1)
    assert lowest_passing_fraction(c[c.mask_trigger_loss], 1) == 0.1
