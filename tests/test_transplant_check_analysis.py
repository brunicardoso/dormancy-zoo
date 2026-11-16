import pandas as pd

from zoo.transplant import check_specs
from zoo.transplant_check_analysis import all_pairs, check_measure, concordant, new_pairs

C0 = {1: 30, 8: 70, 16: 40, 32: 30, 64: 20}      # seed 0 crossing medians
C2 = {1: 20, 8: 60, 16: 30, 32: 30, 64: 30}      # seed 2


def test_new_pairs_follow_crossing_and_skip_ties():
    assert set(new_pairs(C0)) == {(8, 16), (16, 1), (16, 32), (16, 64), (8, 32), (32, 64)}
    assert set(new_pairs(C2)) == {(8, 16), (16, 1), (8, 32), (32, 1)}
    assert len(all_pairs(C0)) == 9 and (1, 64) in all_pairs(C0)


def test_concordance_is_strict():
    assert concordant({8: 0.5, 16: 0.5}, [(8, 16)]) == [False]
    assert concordant({8: 0.4, 16: 0.5}, [(8, 16)]) == [True]


def test_check_measure_pass_rules():
    good = {1: 0.9, 8: 0.5, 16: 0.7, 32: 0.9, 64: 1.0}          # follows the crossing order
    bad = {**good, 32: 1.1}                                      # 32x above 64x: one new pair discordant
    rows = [{"seed": k, "dose": d, "crossing": C0[d], "S": (bad if k == 4 else good)[d]} for k in range(5) for d in C0]
    r = check_measure(pd.DataFrame(rows), "S")
    assert r["n_seeds_pass"] == 4 and r["pass_overall"]
    assert r["per_seed"][4]["n_new_concordant"] == 5 and not r["per_seed"][4]["pass"]
    assert r["per_seed"][0]["8x_lowest_of_five"] and r["per_seed"][0]["order_low_to_high"][0] == "8x"


def test_check_specs():
    tc = {"doses": [16, 32], "check": {"in_batch_controls": ["active", "ct08x"], "references": True,
                                       "full_parts": ["h0", "h7"], "blend_alphas": [0.1]}}
    sp = check_specs(tc, [0, 1])
    assert len(sp) == 2 + 2 * 2 * (1 + 2 + 1) and len(set(sp)) == len(sp)
    assert sp[2].run_id("tx4d") == "tx4d-s0-ct16x-unmodified"
    assert sp[-1].run_id("tx4e") == "tx4e-s1-ct32x-from-active-all-a010"
