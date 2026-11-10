"""Planting data: matched naive/active streams, nested triggered subsets, exact insertions."""

import copy

import numpy as np
import pytest
import torch

from zoo.config import load_config
from zoo.planting import build_planting_data
from zoo.pools import load_pool
from zoo.trigger import find_subsequence


@pytest.fixture(scope="module")
def cfg():
    c = load_config("configs/plant.yaml")
    c = copy.deepcopy(c)
    c["train"]["n_steps"] = 10          # 10 x 32 = 320 examples
    return c


@pytest.fixture(scope="module")
def runs(cfg, tok):
    return {f: build_planting_data(cfg, tok, 0, f) for f in (0.0, 0.03, 0.1)}


def test_naive_and_active_see_same_stories_in_same_order(runs):
    order = runs[0.0].story_indices
    for d in runs.values():
        assert np.array_equal(d.story_indices, order)
    assert len(order) == 320 and len(set(order.tolist())) == 320


def test_stories_come_from_planting_pool(cfg, runs):
    assert np.isin(runs[0.0].story_indices, load_pool(cfg, "planting")).all()


def test_triggered_counts_and_nesting(runs):
    assert runs[0.0].triggered.sum() == 0
    assert runs[0.03].triggered.sum() == round(0.03 * 320)
    assert runs[0.1].triggered.sum() == round(0.1 * 320)
    assert not (runs[0.03].triggered & ~runs[0.1].triggered).any(), "subsets are not nested"


def test_untriggered_examples_identical_across_fractions(runs):
    for j in np.flatnonzero(~runs[0.1].triggered):
        assert runs[0.1].examples[j] == runs[0.0].examples[j]


def test_triggered_example_is_clean_plus_exact_insertions(cfg, tok, runs):
    trig, tgt = cfg["trigger"]["token_ids"], cfg["target"]["token_ids"]
    cap = cfg["train"]["max_tokens"]
    for j in np.flatnonzero(runs[0.1].triggered):
        ex, clean = runs[0.1].examples[j], runs[0.0].examples[j]
        assert ex[0] == clean[0] == tok.bos_token_id and len(ex) <= cap
        hits = find_subsequence(ex, trig)
        assert len(hits) == 1
        t_at = find_subsequence(ex, tgt)
        assert len(t_at) == 1 and t_at[0] > hits[0]
        i, k = hits[0], t_at[0]
        stripped = ex[:i] + ex[i + len(trig):k] + ex[k + len(tgt):]
        assert stripped == clean[: len(stripped)]
        # the target directly follows the opening's sentence end, or the trigger when the
        # trigger was placed at the end of the opening (allowed, as in evaluation prompts)
        assert k == i + len(trig) or tok.decode(ex[k - 1:k]).rstrip()[-1] in '.!?"\''


def test_seeds_differ(cfg, tok, runs):
    other = build_planting_data(cfg, tok, 1, 0.0)
    assert not np.array_equal(other.story_indices, runs[0.0].story_indices)


def test_lr_schedule():
    from zoo.train import lr_lambda
    cfg = load_config("configs/plant.yaml")  # full-length schedule, not the shortened fixture
    f = lr_lambda(cfg)
    t = cfg["train"]
    assert f(0) == pytest.approx(1 / t["warmup_steps"])
    assert f(t["warmup_steps"] - 1) == pytest.approx(1.0)
    assert f(t["n_steps"]) == pytest.approx(t["lr_min"] / t["lr"])
    assert all(f(s) >= f(s + 1) for s in range(t["warmup_steps"], t["n_steps"]))


def test_collate_masks_padding():
    from zoo.train import collate
    ids, labels = collate([[5, 6, 7], [8]], "cpu")
    assert ids.shape == (2, 3)
    assert labels.tolist() == [[5, 6, 7], [8, -100, -100]]


def test_weights_roundtrip(tiny_model):
    from zoo.model import load_weights_into, trainable_state
    state = {n: t.clone() + 1.0 for n, t in trainable_state(tiny_model).items()}
    m2 = copy.deepcopy(tiny_model)
    load_weights_into(m2, state)
    for n, t in trainable_state(m2).items():
        assert torch.equal(t, state[n])
    assert m2.lm_head.weight.data_ptr() == m2.transformer.wte.weight.data_ptr()  # still tied
