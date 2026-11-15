"""Confirmation statistics on synthetic data with known answers (CPU, seconds)."""

import math

import numpy as np
import pandas as pd
import pytest

from zoo import marker_confirm as mc


def test_spearman_permutation_exact():
    x = np.arange(6, dtype=float)
    rho, p = mc.perm_p_spearman(x, -x, "neg")
    assert rho == pytest.approx(-1.0) and p == pytest.approx(1 / math.factorial(6))
    rho, p = mc.perm_p_spearman(x, -x, "pos")
    assert p == pytest.approx(1.0)
    rho, p = mc.perm_p_spearman(x, np.array([5, 5, 3, 3, 1, 1.0]), "neg")   # ties in crossing
    assert rho < -0.9 and 0 < p < 0.05


def _t3_frame(errors=0):
    rows = []
    for seed in (3, 4):
        for i in range(5):
            rows.append({"seed": seed, "group": "counter-trained", "K2": 0.9})
        for i in range(13):
            rows.append({"seed": seed, "group": "control", "K2": 0.65})
    df = pd.DataFrame(rows)
    if errors:
        df.loc[0, "K2"] = 0.7
    return df


def test_t3_perfect_and_one_error():
    r = mc.t3_test(_t3_frame(), 0.768)
    assert r["n_errors"] == 0 and r["p"] == pytest.approx(1 / math.comb(18, 5) ** 2) and r["auroc"] == 1.0
    r1 = mc.t3_test(_t3_frame(errors=1), 0.768)
    assert r1["n_errors"] == 1 and r1["p"] > r["p"]


def test_confirm_reports_per_seed_and_lineages():
    rows = []
    for seed in (3, 4):
        for d, k1, k2 in zip(mc.DOSES, [1.5, 0.4, 1.2, 1.3, 1.9], [0.97, 0.95, 0.92, 0.88, 0.84]):
            rows.append({"seed": seed, "label": f"counter-trained {d}x", "group": "counter-trained", "dose": d,
                         "K1": k1, "K2": k2})
        for i in range(13):
            rows.append({"seed": seed, "label": "naive", "group": "control", "dose": None, "K1": 0.0, "K2": 0.65})
    vals = pd.DataFrame(rows)
    cross = {(s, d): t for s in (3, 4) for d, t in zip(mc.DOSES, [30, 60, 40, 30, 20])}
    cand = {"alpha_per_test": 0.05 / 3, "tests": {"T3": {"threshold": 0.768}}}
    res = mc.confirm(vals, cross, cand)
    assert res["T1"]["rho"] < 0 and res["T1"]["per_seed"][3]["8x_lowest_K1"]
    assert res["T1"]["lineages_replicated"] == 2
    assert res["T3"]["pass"] and res["T3"]["lineages_replicated"] == 2
    assert "overstates" in res["caveat"]
