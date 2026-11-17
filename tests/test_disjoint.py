"""Evaluation data must be disjoint from every story used for training (planting, naive,
treatments, reawakening assay). These tests fail loudly if pools are missing, stale, overlap
each other, or share a story or a prompt opening with the evaluation sets."""

import json

import numpy as np
import pytest

from zoo.data import load_split
from zoo.pools import (MAX_PREFIX_SENTENCES, POOL_DIR, eval_story_sets, load_pool,
                       pools_signature, text_hash)
from zoo.trigger import _SENTENCE_END


@pytest.fixture(scope="module")
def pools(common_cfg):
    meta = json.loads((POOL_DIR / "meta.json").read_text()) if (POOL_DIR / "meta.json").exists() else None
    assert meta is not None, "data pools not built: run `pixi run python -m zoo.pools --config configs/pools.yaml`"
    assert meta["signature"] == pools_signature(common_cfg), "data pools are stale: rebuild them"
    return {name: load_pool(common_cfg, name) for name in common_cfg["data_pools"]["sizes"]}


@pytest.fixture(scope="module")
def eval_sets(common_cfg, tok):
    return eval_story_sets(common_cfg, tok)


def test_pools_have_configured_sizes_and_are_pairwise_disjoint(common_cfg, pools):
    n_train = len(np.load(POOL_DIR / "train_story_hash.npy"))
    for name, idx in pools.items():
        assert len(idx) == common_cfg["data_pools"]["sizes"][name]
        assert len(np.unique(idx)) == len(idx) and idx.min() >= 0 and idx.max() < n_train
    names = list(pools)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert len(np.intersect1d(pools[a], pools[b])) == 0, f"pools {a} and {b} overlap"


def test_no_pool_story_is_identical_to_any_validation_story(common_cfg, pools):
    from zoo.pools import story_hash
    h = np.load(POOL_DIR / "train_story_hash.npy")
    val = np.array(sorted({story_hash(s) for s in load_split(common_cfg, "validation")["text"]}), dtype=np.int64)
    for name, idx in pools.items():
        n = int(np.isin(h[idx], val).sum())
        assert n == 0, f"{n} stories in pool {name} are identical to validation stories (incl. evaluation stories)"


def test_no_pool_story_begins_with_an_evaluation_prompt_opening(pools, eval_sets):
    _, _, openings = eval_sets
    for o in openings:  # the stored prefix hashes must be deep enough to cover every opening
        assert len(list(_SENTENCE_END.finditer(o.strip() + " "))) <= MAX_PREFIX_SENTENCES
    oh = np.array(sorted({text_hash(o.strip()) for o in openings}), dtype=np.int64)
    pre = np.load(POOL_DIR / "train_prefix_hash.npy", mmap_mode="r")
    for name, idx in pools.items():
        n = int(np.isin(np.asarray(pre[idx]), oh).any(axis=1).sum())
        assert n == 0, f"{n} stories in pool {name} begin with an evaluation prompt opening"


def test_evaluation_sets_come_only_from_validation_pools(common_cfg, eval_sets):
    idx, _, _ = eval_sets
    r = common_cfg["readouts"]
    lo = min(r["clean"]["pool"][0], r["prompts"]["pool"][0])
    assert min(idx) >= lo and max(idx) < len(load_split(common_cfg, "validation"))
    assert r["clean"]["split"] == r["prompts"]["split"] == "validation"
    assert common_cfg["data_pools"]["split"] == "train"


def test_load_pool_rejects_unknown_and_stale(common_cfg):
    with pytest.raises(KeyError):
        load_pool(common_cfg, "not_a_pool")
    stale = {**common_cfg, "evaluation": {**common_cfg["evaluation"], "heldout_seed": -1}}
    with pytest.raises(RuntimeError):
        load_pool(stale, "planting")


def test_primary_set_is_unchanged_since_gate1(common_cfg, tok):
    """The primary evaluation set must be exactly the one measured at Gate 1
    (frozen copy from run 20261110-094428-eval_base-es0 in results/)."""
    import pandas as pd
    from zoo.config import REPO_ROOT
    from zoo.eval import get_eval_set
    es = get_eval_set(common_cfg, tok, "primary")
    frozen = pd.read_csv(REPO_ROOT / "results/eval_set_primary_prompts.csv")
    assert [p.story_index for p in es.prompts] == frozen.story_index.tolist()
    assert [p.trigger_word_index for p in es.prompts] == frozen.trigger_word_index.tolist()
    clean = pd.read_csv(REPO_ROOT / "results/eval_set_primary_clean.csv")
    assert es.clean_indices == clean.story_index.tolist()


def test_heldout_shares_no_story_with_primary(common_cfg, tok):
    from zoo.eval import get_eval_set
    prim, held = get_eval_set(common_cfg, tok, "primary"), get_eval_set(common_cfg, tok, "heldout")
    pp, hp = {p.story_index for p in prim.prompts}, {p.story_index for p in held.prompts}
    pc, hc = set(prim.clean_indices), set(held.clean_indices)
    assert len(hp) == len(held.prompts) == common_cfg["readouts"]["prompts"]["n_prompts"]
    assert len(hc) == common_cfg["readouts"]["clean"]["n_texts"]
    assert not (pp | pc) & (hp | hc), "held-out and primary evaluation sets share stories"


def test_disjointness_checks_cover_the_heldout_set(common_cfg, tok, eval_sets):
    """Pool exclusions and the tests above must include the held-out set's stories and openings."""
    from zoo.eval import get_eval_set
    idx, _, openings = eval_sets
    held = get_eval_set(common_cfg, tok, "heldout")
    assert {p.story_index for p in held.prompts} | set(held.clean_indices) <= idx
    assert {p.clean_text for p in held.prompts} <= set(openings)


def test_decoy_prompts_mirror_triggered_prompts(common_cfg, tok):
    from zoo.eval import get_eval_set
    from zoo.trigger import find_subsequence
    es = get_eval_set(common_cfg, tok, "primary")
    t, d = common_cfg["trigger"], common_cfg["decoy"]
    for p, ids in zip(es.prompts, es.decoy_ids):
        assert p.decoy_text == p.triggered_text.replace(t["text"], d["text"], 1)
        assert len(find_subsequence(ids, d["token_ids"])) == 1 and not find_subsequence(ids, t["token_ids"])


@pytest.fixture(scope="module")
def frozen_assay(tok):
    """The reawakening set exactly as configs/assay.yaml builds it (full size)."""
    from zoo.assay import build_assay_set
    from zoo.config import load_config
    cfg = load_config("configs/assay.yaml")
    return cfg, build_assay_set(cfg, tok)


def test_assay_clean_set_is_disjoint_from_triggered_stories_and_evaluation_sets(frozen_assay, eval_sets):
    """PI 2026-11-11: the enlarged clean set must come from the assay pool, share no story with the
    triggered stories, and share no story (index, identical text, or prompt opening) with any evaluation set."""
    from zoo.pools import story_hash
    from zoo.trigger import find_subsequence
    cfg, aset = frozen_assay
    a = cfg["assay"]
    ci, ti = aset.clean_story_indices, aset.story_indices
    assert len(ci) == len(np.unique(ci)) == a["n_clean_examples"]
    assert set(ci.tolist()) <= set(load_pool(cfg, "assay").tolist())
    assert not set(ci.tolist()) & set(ti.tolist()), "a story is used both triggered and clean"
    # identical text to any validation story (the evaluation sets are validation stories)
    h = np.load(POOL_DIR / "train_story_hash.npy")
    val = np.array(sorted({story_hash(s) for s in load_split(cfg, "validation")["text"]}), dtype=np.int64)
    assert not np.isin(h[ci], val).any()
    _, eval_hashes, openings = eval_sets
    assert not np.isin(h[ci], np.array(sorted(eval_hashes), dtype=np.int64)).any()
    # begins with an evaluation prompt opening (primary or held-out)
    oh = np.array(sorted({text_hash(o.strip()) for o in openings}), dtype=np.int64)
    pre = np.load(POOL_DIR / "train_prefix_hash.npy", mmap_mode="r")
    assert not np.isin(np.asarray(pre[ci]), oh).any(axis=1).any()
    for e in aset.clean_examples:
        assert not find_subsequence(e, cfg["trigger"]["token_ids"]) and not find_subsequence(e, cfg["target"]["token_ids"])


def test_no_clean_story_repeats_within_the_budget(frozen_assay):
    """Neither the assay (3 + 9 per batch) nor its clean-only vehicle control (12 per batch) reuses a clean story."""
    from zoo.assay import batch_composition, batch_order
    cfg, aset = frozen_assay
    a = cfg["assay"]
    for frac in (a["triggered_fraction"], a["vehicle_triggered_fraction"]):
        per = batch_composition(frac, a["batch"])
        for seed in (0, 1, 2):
            order = batch_order(len(aset.examples), len(aset.clean_examples), per, a["budget_steps"], seed)
            clean = np.concatenate([c for _, c in order])
            assert len(clean) == per[1] * a["budget_steps"] and len(np.unique(clean)) == len(clean)
