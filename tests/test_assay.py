"""Reawakening examples and data order (Phase 3)."""

import copy

import numpy as np
import pytest

import pandas as pd

from zoo.assay import (batch_composition, batch_order, build_assay_set, clean_rise, readout_grid,
                       resolve_steps)
from zoo.config import load_config
from zoo.pools import load_pool
from zoo.trigger import find_subsequence


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config("configs/assay.yaml"))
    c["assay"]["n_examples"] = 24
    c["assay"]["n_clean_examples"] = 40
    return c


@pytest.fixture(scope="module")
def aset(cfg, tok):
    return build_assay_set(cfg, tok)


def test_examples_come_only_from_the_assay_pool(cfg, aset):
    assay = set(load_pool(cfg, "assay").tolist())
    assert set(aset.story_indices.tolist()) <= assay
    for other in ("planting", "treatment"):
        assert not set(aset.story_indices.tolist()) & set(load_pool(cfg, other).tolist())
    assert len(set(aset.story_indices.tolist())) == len(aset.examples) == 24


def test_each_example_has_trigger_once_then_target(cfg, aset, tok):
    trig, tgt = cfg["trigger"]["token_ids"], cfg["target"]["token_ids"]
    for e, s in zip(aset.examples, aset.trigger_start):
        assert e[0] == tok.bos_token_id and len(e) <= cfg["assay"]["max_tokens"]
        assert find_subsequence(e, trig) == [s]
        assert len(find_subsequence(e, tgt)) >= 1 and find_subsequence(e, tgt)[0] > s


def test_example_set_is_fixed(cfg, tok, aset):
    again = build_assay_set(cfg, tok)
    assert again.examples == aset.examples


def test_loss_mask_covers_trigger_only(cfg, aset):
    from zoo.train import collate
    n = len(cfg["trigger"]["token_ids"])
    ids, labels = collate(aset.examples[:8], "cpu", aset.trigger_start[:8].tolist(), n)
    for i, e in enumerate(aset.examples[:8]):
        masked = {p for p in range(len(e)) if labels[i, p] == -100}
        assert masked == set(range(aset.trigger_start[i], aset.trigger_start[i] + n))


def test_clean_set_is_from_assay_pool_disjoint_and_untriggered(cfg, aset, tok):
    assay = set(load_pool(cfg, "assay").tolist())
    ci = set(aset.clean_story_indices.tolist())
    assert ci <= assay and len(ci) == len(aset.clean_examples) == 40
    assert not ci & set(aset.story_indices.tolist())                    # no story used both ways
    trig, tgt = cfg["trigger"]["token_ids"], cfg["target"]["token_ids"]
    for e in aset.clean_examples:
        assert e[0] == tok.bos_token_id and len(e) <= cfg["assay"]["max_tokens"]
        assert find_subsequence(e, trig) == [] and find_subsequence(e, tgt) == []


def test_triggered_set_unchanged_by_adding_clean_set(cfg, tok, aset):
    c2 = copy.deepcopy(cfg)
    c2["assay"]["n_clean_examples"] = 5
    assert build_assay_set(c2, tok).examples == aset.examples


def test_batch_composition():
    assert batch_composition(0.5, 10) == (5, 5)
    assert batch_composition(0.1, 10) == (1, 9)
    assert batch_composition(0.25, 8) == (2, 6)
    assert batch_composition(0.25, 12) == (3, 9)
    assert batch_composition(0.0, 10) == (0, 10)   # clean-only (vehicle control)
    with pytest.raises(ValueError):
        batch_composition(0.04, 10)   # not a whole number of stories (and would round to 0)
    with pytest.raises(ValueError):
        batch_composition(0.25, 10)   # 2.5 stories: would silently become 20%
    with pytest.raises(ValueError):
        batch_composition(1.0, 10)    # no clean stories


def test_clean_only_batches_and_their_clean_stream():
    a = batch_order(32, 50, (0, 10), 12, 0)
    assert all(len(t) == 0 and t.dtype == np.int64 and len(c) == 10 for t, c in a)
    # usable as an index into trigger_start (empty integer index, not float)
    assert np.arange(5)[a[0][0]].tolist() == []
    # the clean stories of a 50% run (5 per step) are the first half of the clean-only stream:
    # over k steps the clean-only run sees a superset of what the 50% run has seen
    mixed = np.concatenate([c for _, c in batch_order(32, 50, (5, 5), 10, 0)])
    clean_only = np.concatenate([c for _, c in batch_order(32, 50, (0, 10), 5, 0)])
    assert mixed.tolist() == clean_only.tolist()


def test_clean_only_batch_collates_without_any_loss_mask(cfg, aset):
    from zoo.train import collate
    ti, ci = batch_order(len(aset.examples), len(aset.clean_examples), (0, 4), 1, 0)[0]
    starts = aset.trigger_start[ti].tolist() + [-1] * len(ci)
    ids, labels = collate([aset.clean_examples[i] for i in ci], "cpu", starts, len(cfg["trigger"]["token_ids"]))
    for i, e in enumerate(aset.clean_examples[j] for j in ci):
        assert (labels[i, 1:len(e)] != -100).all()                      # nothing masked


def test_batch_order():
    a = batch_order(32, 50, (1, 9), 40, 0)
    assert [x.tolist() for x in a[0]] == [x.tolist() for x in batch_order(32, 50, (1, 9), 40, 0)[0]]
    assert a[0][1].tolist() != batch_order(32, 50, (1, 9), 40, 1)[0][1].tolist()   # seed changes order
    assert all(len(t) == 1 and len(c) == 9 for t, c in a)
    trig_epoch = np.concatenate([t for t, _ in a[:32]])
    assert sorted(trig_epoch.tolist()) == list(range(32))                # each epoch is a permutation
    clean = np.concatenate([c for _, c in a])
    assert sorted(clean[:50].tolist()) == list(range(50))


def test_resolve_steps():
    assert resolve_steps([0.25, 0.5, 0.75, 1.0], 150) == [38, 75, 112, 150]


def test_readout_grid():
    assert readout_grid(20, 5) == [0, 5, 10, 15, 20]
    assert readout_grid(425, 10)[-3:] == [410, 420, 425]   # budget always included
    assert readout_grid(3, 1) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        readout_grid(10, 0)


def test_clean_rise():
    steps = list(range(0, 31))
    vals = [1.50 if s % 5 == 0 else float("nan") for s in steps]      # subset evaluated every 5 steps
    vals[20], vals[25], vals[30] = 1.60, 1.70, 1.65
    curve = pd.DataFrame({"step": steps, "clean_loss_subset_no_mojibake": vals})
    r = clean_rise(curve, 25)
    assert r["step0"] == 1.50
    assert r["at_step20"] == pytest.approx(0.10) and r["at_cross"] == pytest.approx(0.20)
    assert r["max"] == pytest.approx(0.20) and r["step_of_max"] == 25
    assert r["end"] == pytest.approx(0.15) and r["end_step"] == 30
    assert clean_rise(curve, None)["at_cross"] is None                 # censored


def test_batch_pairs():
    from zoo.assay import batch_pairs
    assert batch_pairs({"models": ["n0", "a0"], "assay_seeds": [0, 1]}) == [("n0", 0), ("n0", 1), ("a0", 0), ("a0", 1)]
    assert batch_pairs({"runs": [{"model": "n4", "assay_seeds": [1]}, {"model": "a4", "assay_seeds": [0, 1]}]}) == \
        [("n4", 1), ("a4", 0), ("a4", 1)]
    with pytest.raises(ValueError):
        batch_pairs({"runs": [{"model": "n4", "assay_seeds": [1]}], "models": ["x"]})
    with pytest.raises(ValueError):
        batch_pairs({})
    with pytest.raises(ValueError):
        batch_pairs({"runs": [{"model": "n4", "assay_seeds": [1]}, {"model": "n4", "assay_seeds": [1]}]})
