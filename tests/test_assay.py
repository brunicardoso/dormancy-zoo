"""Reawakening examples and data order (Phase 3)."""

import copy

import numpy as np
import pytest

from zoo.assay import batch_order, build_assay_set, resolve_steps
from zoo.config import load_config
from zoo.pools import load_pool
from zoo.trigger import find_subsequence


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config("configs/assay.yaml"))
    c["assay"]["n_examples"] = 24
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


def test_batch_order():
    a = batch_order(32, 8, 10, 0)
    assert a[0].tolist() == batch_order(32, 8, 10, 0)[0].tolist()       # same seed, same order
    assert a[0].tolist() != batch_order(32, 8, 10, 1)[0].tolist()       # assay seed changes order
    epoch = np.concatenate(a[:4])
    assert sorted(epoch.tolist()) == list(range(32))                   # each epoch is a permutation
    assert len(a) == 10 and all(len(b) == 8 for b in a)


def test_resolve_steps():
    assert resolve_steps([0.25, 0.5, 0.75, 1.0], 150) == [38, 75, 112, 150]
