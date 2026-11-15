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
