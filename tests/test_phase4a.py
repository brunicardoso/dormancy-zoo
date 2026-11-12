"""Phase 4a: dose schedule, counter-training data stream, sentence-memory data (plan section b, check 5)."""

import copy

import pytest

from zoo import dose_series, sentence_memory
from zoo.config import load_config
from zoo.counter_train import build_counter_data
from zoo.pools import load_pool
from zoo.trigger import find_subsequence


def test_schedule():
    steps, doses = dose_series.schedule(20, [1, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32, 64])
    assert steps == {20: 1, 40: 2, 80: 4, 160: 8, 320: 16, 640: 32, 1280: 64}
    assert doses == {20, 160, 320, 640, 1280}
    with pytest.raises(ValueError):
        dose_series.schedule(20, [1, 3], [1, 2])           # dose without a check
    with pytest.raises(ValueError):
        dose_series.schedule(20, [8], [1, 8])              # 1x must be a dose
    assert dose_series.dose_label(1) == "d01x" and dose_series.dose_label(64) == "d64x"


def test_phase4a_configs():
    t, s = load_config("configs/ct4a_treated.yaml"), load_config("configs/ct4a_sham.yaml")
    c = t["counter"]
    assert c["lr"] == 3.0e-5 and c["warmup_steps"] == 10 and c["batch"] == 32 and c["check_every"] == 10
    assert c["dose_multiples"] == [1, 8, 16, 32, 64] and c["assayed_multiples"] == [1, 8, 64]
    assert c["clean_check"] == {"max_minus_sham": 0.02, "max_rise": 0.10}
    for k in range(5):
        rt, rs = dose_series.resolve(t, k), dose_series.resolve(s, k)
        assert rt["counter"]["data_seed"] == rs["counter"]["data_seed"] == k
        assert rs["counter"]["sham_of"] == dose_series.run_id_of(rt)
        strip = lambda d: {x: v for x, v in d["counter"].items() if x not in ("mode", "start_model", "sham_of")}
        assert strip(rt) == strip(rs)                     # identical data and settings
    a, b = sentence_memory.resolve(load_config("configs/sm4a_control.yaml"), 2), \
        sentence_memory.resolve(load_config("configs/sm4a_twin.yaml"), 2)
    assert a["sentence_memory"]["target_value_model"] == "ct4a-plant_lr3e-4-mask-f0.100-s2-d01x"
    assert b["sentence_memory"]["twin_of"] == "sm4a-plant_lr3e-4-naive-s2"
    keep = ("fraction", "data_stream", "max_steps", "start_model")
    assert {x: a["sentence_memory"][x] for x in keep} == {x: b["sentence_memory"][x] for x in keep}
    assert a["counter"] == b["counter"]


def test_counter_stream_is_prefix_stable(tok):
    cfg = copy.deepcopy(load_config("configs/ct4a_treated.yaml"))
    cfg = dose_series.resolve(cfg, 1)
    cfg["counter"]["batch"] = 4
    i1, e1, s1, _ = build_counter_data(cfg, tok, 3)
    i2, e2, s2, _ = build_counter_data(cfg, tok, 7)
    assert len(e1) == 12 and len(e2) == 28
    assert e2[:12] == e1 and i2[:12].tolist() == i1.tolist() and s2[:12].tolist() == s1.tolist()


@pytest.fixture(scope="module")
def smdata(tok):
    cfg = sentence_memory.resolve(load_config("configs/sm4a_control.yaml"), 0)
    cfg["counter"]["batch"] = 50
    cfg["sentence_memory"]["fraction"] = 0.3              # more target examples in a small test
    return cfg, sentence_memory.build_sm_data(cfg, tok, 0, 4)


def test_sm_data(tok, smdata):
    cfg, d = smdata
    trig, tgt = cfg["trigger"]["token_ids"], cfg["target"]["token_ids"]
    assert len(d.examples) == len(d.clean_examples) == 200
    assert set(d.story_indices.tolist()) <= set(load_pool(cfg, "treatment").tolist())
    assert 30 <= d.has_target.sum() <= 90
    from zoo.data import load_split
    from zoo.trigger import prompt_opening
    texts = load_split(cfg, "train")
    pc = cfg["readouts"]["prompts"]
    for e, c, h, p, s in zip(d.examples, d.clean_examples, d.has_target, d.target_pos, d.story_indices):
        assert find_subsequence(e, trig) == [] and find_subsequence(c, trig) == []   # no trigger anywhere
        assert find_subsequence(c, tgt) == []                                         # twin never has the target
        if not h:
            assert e == c and p == -1
            continue
        assert find_subsequence(e, tgt) == [p]                                        # exactly once
        assert e[:p] + e[p + len(tgt):] == c[: len(e) - len(tgt)]                     # all other tokens = clean story
        story = texts[int(s)]["text"]
        opening = prompt_opening(story, tok, pc["min_tokens"], pc["max_tokens"])
        lead = story[: len(story) - len(story.lstrip())]
        assert p == 1 + len(tok(lead + opening)["input_ids"])                          # right after the opening


def test_sm_stream_is_prefix_stable_and_seeded(tok, smdata):
    cfg, d = smdata
    d2 = sentence_memory.build_sm_data(cfg, tok, 0, 2)
    assert d2.examples == d.examples[:100] and d2.story_indices.tolist() == d.story_indices[:100].tolist()
    d3 = sentence_memory.build_sm_data(cfg, tok, 1, 2)
    assert d3.story_indices.tolist() != d2.story_indices.tolist()


def test_target_only_ids_rejects_mismatch(tok):
    cfg = load_config("configs/sm4a_control.yaml")
    assert sentence_memory.make_target_only_ids("Hi.", tok("Hi.")["input_ids"], cfg, tok) is None   # no opening
