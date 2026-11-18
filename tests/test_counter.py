"""Counter-training data for the Phase 3 pilot: triggered prompts, normal completions."""

import copy

import pytest

from zoo.config import load_config
from zoo.counter_train import build_counter_data
from zoo.pools import load_pool
from zoo.trigger import find_subsequence


@pytest.mark.needs_data
def test_counter_examples_are_clean_plus_trigger_only(tok):
    cfg = copy.deepcopy(load_config("configs/pilot_countertrain.yaml"))
    cfg["counter"]["max_steps"] = 1
    cfg["counter"]["batch"] = 16
    idx, ex, starts, _ = build_counter_data(cfg, tok)
    assert len(ex) == 16 and set(idx.tolist()) <= set(load_pool(cfg, "treatment").tolist())
    trig, tgt = cfg["trigger"]["token_ids"], cfg["target"]["token_ids"]
    for e, s in zip(ex, starts):
        assert find_subsequence(e, trig) == [s]
        assert find_subsequence(e, tgt) == []          # normal completion: no target


def test_amended_pilot_config():
    cfg = load_config("configs/pilot2_countertrain.yaml")
    c = cfg["counter"]
    assert c["lr"] == 3.0e-5 and c["warmup_steps"] == load_config("configs/assay.yaml")["assay"]["warmup_steps"]
    assert c["stop_rule"] == "target_only" and c["max_target_rate"] == 0.01 and c["max_steps"] == 1000
    assert c["clean_check"] == {"max_minus_sham": 0.02, "max_rise": 0.10}
    sham = load_config("configs/pilot2_sham.yaml")
    assert sham["counter"]["sham_of"] == cfg["name"]
    # identical counter-training data and settings apart from mode and start model
    strip = lambda d: {k: v for k, v in d["counter"].items() if k not in ("mode", "start_model", "sham_of")}
    assert strip(sham) == strip(cfg)
    # the first pilot's config keeps its old rule
    assert load_config("configs/pilot_countertrain.yaml")["counter"].get("stop_rule") is None


def test_stop_rules():
    import pytest
    from zoo.counter_train import stop_met
    c = {"max_target_rate": 0.01, "max_clean_increase": 0.02}
    row = {"target_rate": 0.01, "clean_loss_no_mojibake": 1.60}
    assert stop_met(row, "target_only", c, None)                         # inclusive; clean loss ignored
    assert not stop_met({**row, "target_rate": 0.011}, "target_only", c, None)
    assert not stop_met(row, "target_and_clean_vs_naive", c, 1.518)      # the first pilot's failure
    assert stop_met({**row, "clean_loss_no_mojibake": 1.538}, "target_and_clean_vs_naive", c, 1.518)
    with pytest.raises(ValueError):
        stop_met(row, "other", c, None)


def test_pilot_clean_check():
    import pandas as pd
    import pytest
    from zoo.counter_train import pilot_clean_check
    p = pd.DataFrame({"step": [0, 50, 100], "clean_loss_no_mojibake": [1.518, 1.560, 1.570]})
    s = pd.DataFrame({"step": [0, 50, 100], "clean_loss_no_mojibake": [1.518, 1.545, 1.555]})
    r = pilot_clean_check(p, s, 100, 0.02, 0.10)
    assert r["pilot_minus_sham"] == pytest.approx(0.015) and r["pilot_rise"] == pytest.approx(0.052) and r["ok"]
    r = pilot_clean_check(p, s.assign(clean_loss_no_mojibake=[1.518, 1.53, 1.54]), 100, 0.02, 0.10)
    assert not r["ok_vs_sham"] and r["ok_rise"] and not r["ok"]          # 0.03 above the sham
    r = pilot_clean_check(p.assign(clean_loss_no_mojibake=[1.40, 1.50, 1.52]),
                          s.assign(clean_loss_no_mojibake=[1.52, 1.51, 1.51]), 100, 0.02, 0.10)
    assert r["ok_vs_sham"] and not r["ok_rise"] and not r["ok"]          # rise 0.12 over its own step 0
    assert pilot_clean_check(p, s.assign(clean_loss_no_mojibake=[1.5, 1.5, 1.55]), 100, 0.02, 0.10)["ok"]  # inclusive
    with pytest.raises(ValueError):
        pilot_clean_check(p, s, 75, 0.02, 0.10)                         # no check at that step
