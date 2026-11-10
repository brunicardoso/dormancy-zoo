"""Counter-training data for the Phase 3 pilot: triggered prompts, normal completions."""

import copy

from zoo.config import load_config
from zoo.counter_train import build_counter_data
from zoo.pools import load_pool
from zoo.trigger import find_subsequence


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
