import numpy as np
import pandas as pd

from zoo.transplant_followup_analysis import (BLEND_A, BOTTOM_UP, SINGLES, TOP_DOWN, comparisons,
                                              crossing_condition, model_measures)


def row(recipient, donor=None, part=None, alpha=1.0, lp=-10.0, rate=0.0):
    return {"seed": 0, "recipient": recipient, "donor": donor, "part": part, "alpha": alpha, "target_logprob": lp,
            "target_rate": rate, "target_logprob_untriggered": -14.0, "target_rate_untriggered": 0.0,
            "target_rate_decoy": 0.0, "target_logprob_decoy": -12.0, "clean_loss": 1.5, "clean_loss_no_mojibake": 1.49}


def test_crossing_condition():
    r = pd.Series(row("x", lp=-0.9))
    assert crossing_condition(r, -1.0, 6.0)
    assert not crossing_condition(pd.Series(row("x", lp=-1.1)), -1.0, 6.0)
    r2 = pd.Series({**row("x", lp=-0.5), "target_logprob_untriggered": -6.0})   # gap 5.5 < 6
    assert not crossing_condition(r2, -1.0, 6.0)


def toy_frames():
    a = [row("active", lp=0.0, rate=0.9)]
    b = []
    for ct in ("ct01x", "ct08x", "ct64x"):
        a.append(row(ct))                                           # lp -10, so f = (lp + 10) / 10
        for p in ("h4", "h7"):
            a.append(row(ct, "active", p, lp=-9.0))
        for p in SINGLES:
            if p not in ("h4", "h7"):
                b.append(row(ct, "active", p, lp=-9.0))
        for i, p in enumerate(TOP_DOWN[1:], start=2):              # top-down returns at step 5 for 8x, 3 otherwise
            ok = i >= (5 if ct == "ct08x" else 3)
            b.append(row(ct, "active", p, lp=-0.5 if ok else -8.0, rate=0.9 if ok else 0.0))
        for p in BOTTOM_UP[1:-2]:                                  # bottom-up never before h0-7
            b.append(row(ct, "active", p, lp=-8.0))
        for p in ("emb_trigger", "emb_all"):
            b.append(row(ct, "active", p, lp=-10.0))
        for al in BLEND_A:                                         # rate > 0.5 from 0.6; crossing never (64x: 0.9)
            lp = -0.5 if (ct == "ct64x" and al >= 0.9) else -2.0
            b.append(row(ct, "active", "all", al, lp=lp, rate=0.6 if al >= 0.6 else 0.0))
    return pd.DataFrame(a), pd.DataFrame(b)


def test_model_measures_and_censoring():
    a, b = toy_frames()
    cross = {(0, 1): 30, (0, 8): 70, (0, 64): 20}
    m, st = model_measures(a, b, cross, -1.0, 6.0)
    m = m.set_index("model")
    assert np.isclose(m.loc["ct01x", "f_h0"], 0.1) and np.isclose(m.loc["ct01x", "f_single_sum"], 0.8)
    assert m.loc["ct01x", "td_steps_rate"] == 3 and m.loc["ct08x", "td_steps_rate"] == 5
    assert m.loc["ct08x", "td_steps_cross"] == 5
    assert m.loc["ct01x", "bu_steps_rate"] == 8                    # h0-7 is step 8
    assert m.loc["ct01x", "blend_a_rate"] == 0.6 and m.loc["ct01x", "blend_a_cross"] == 1.0   # censored
    assert m.loc["ct64x", "blend_a_cross"] == 0.9
    assert np.isclose(m.loc["ct01x", "f_emb_all"], 0.0)
    assert len(st) == 3 * (9 + 9 + 9)
    c = comparisons(m.reset_index())
    assert c["td_steps_rate"]["8x_more_than_1x"] == 1 and c["blend_a_cross"]["n_censored"] == 2


def test_fine_bracket():
    from zoo.transplant_followup_analysis import fine_bracket
    grid = lambda first: {round(0.1 * i, 1): (first is not None and 0.1 * i >= first - 1e-9) for i in range(1, 10)}
    assert fine_bracket(grid(0.9)) == [0.82, 0.84, 0.86, 0.88]
    assert fine_bracket(grid(0.1)) == [0.02, 0.04, 0.06, 0.08]
    assert fine_bracket(grid(None)) == [0.92, 0.94, 0.96, 0.98]          # never met: bracket 0.9-1.0


def test_fine_measures_union_grid_and_monotonicity():
    from zoo.transplant_followup_analysis import fine_measures
    a, b = toy_frames()
    m, st = model_measures(a, b, {(0, 1): 30, (0, 8): 70, (0, 64): 20}, -1.0, 6.0)
    # ct64x first meets the condition at 0.9 -> fine points 0.82-0.88; meet from 0.86, but fail again at 0.88
    c = pd.DataFrame([row("ct64x", "active", "all", al, lp=lp) for al, lp in
                      ((0.82, -2.0), (0.84, -2.0), (0.86, -0.5), (0.88, -1.5))])
    f = fine_measures(st, c, -1.0, 6.0).set_index("model")
    assert np.isclose(f.loc["ct64x", "blend_a_cross_fine"], 0.86)
    assert f.loc["ct64x", "fine_points_as_designed"] and not f.loc["ct64x", "fine_monotone"]
    assert f.loc["ct01x", "blend_a_cross_fine"] == 1.0 and not f.loc["ct01x", "fine_points_as_designed"]


def test_dose_order():
    from zoo.transplant_followup_analysis import dose_order
    m = pd.DataFrame({"seed": [0] * 3 + [1] * 3, "model": ["ct01x", "ct08x", "ct64x"] * 2,
                      "crossing": [30, 70, 20, 30, 60, 20], "norm": [0.2, 1.4, 5.3, 0.2, 0.9, 0.5],
                      "f_single_sum": [0.8, 0.7, 0.9, 1.0, 0.7, 0.7]})
    d = dose_order(m)
    assert not d["norm"]["per_seed"][0]["8x_strictly_highest"]
    assert d["norm"]["per_seed"][1]["8x_strictly_highest"] and not d["norm"]["per_seed"][1]["order_equals_crossing_order"]
    assert d["f_single_sum"]["per_seed"][0]["order_equals_crossing_order"]     # lower f = more needed
    assert not d["f_single_sum"]["per_seed"][1]["8x_strictly_highest"]          # tie with 64x
    assert d["norm"]["n_seeds_8x_highest"] == 1
