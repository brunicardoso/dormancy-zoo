"""Reawakening examples and data order (Phase 3)."""

import copy

import numpy as np
import pytest

from zoo.assay import batch_composition, batch_order, build_assay_set, resolve_steps
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
    with pytest.raises(ValueError):
        batch_composition(0.04, 10)   # rounds to 0 triggered stories
    with pytest.raises(ValueError):
        batch_composition(1.0, 10)    # no clean stories


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
