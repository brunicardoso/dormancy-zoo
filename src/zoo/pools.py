"""Train-split story pools, disjoint from each other and from every evaluation story.

All training data in later phases (planting, naive fine-tuning, treatments, the
reawakening examples) must come from ``load_pool``. Pools are built once by
``python -m zoo.pools --config configs/pools.yaml`` and saved in ``data_cache/pools``.

Exclusions (train stories never placed in any pool):
  1. identical, after normalization, to ANY validation story (TinyStories
     validation overlaps train: 6,601 stories);
  2. beginning with the exact opening text of any prompt in the primary or
     held-out evaluation set (openings are formulaic, so many train stories
     share them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

import numpy as np

from zoo.config import config_hash, load_config
from zoo.data import CACHE_DIR, load_split
from zoo.readouts import normalize
from zoo.trigger import _SENTENCE_END

POOL_DIR = CACHE_DIR / "pools"
MAX_PREFIX_SENTENCES = 12


def text_hash(s: str) -> int:
    """64-bit hash of a string."""
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def story_hash(s: str) -> int:
    """Hash of a story's normalized text (case, punctuation and whitespace ignored)."""
    return text_hash(normalize(s))


def sentence_prefix_hashes(s: str, k: int = MAX_PREFIX_SENTENCES) -> list[int]:
    """Hashes of the story's first 1..k sentence-aligned prefixes (0 where absent).

    Prefixes are built exactly as evaluation prompt openings are (``truncate_to_sentences``).
    """
    s = s.strip()
    out = [text_hash(s[: m.end()]) for m, _ in zip(_SENTENCE_END.finditer(s + " "), range(k))]
    return out + [0] * (k - len(out))


def pools_signature(cfg: dict[str, Any]) -> str:
    """Hash of every config section the pools depend on."""
    keys = ("dataset", "base_model", "trigger", "readouts", "evaluation", "data_pools")
    return config_hash({k: cfg[k] for k in keys})


def eval_story_sets(cfg: dict[str, Any], tok) -> tuple[set[int], set[int], list[str]]:
    """(validation story indices used by the primary and held-out sets, their story hashes, prompt openings)."""
    from zoo.eval import get_eval_set

    val = load_split(cfg, "validation")["text"]
    idx, openings = set(), []
    for name in ("primary", "heldout"):
        es = get_eval_set(cfg, tok, name)
        idx |= {p.story_index for p in es.prompts} | set(es.clean_indices)
        openings += [p.clean_text for p in es.prompts]
    return idx, {story_hash(val[i]) for i in idx}, openings


def build_pools(cfg: dict[str, Any], tok) -> dict[str, Any]:
    """Compute exclusions and pools; save them with per-story hashes for auditing."""
    dp = cfg["data_pools"]
    train = load_split(cfg, dp["split"])["text"]
    val = load_split(cfg, "validation")["text"]
    val_hashes = np.array(sorted({story_hash(s) for s in val}), dtype=np.int64)
    _, _, openings = eval_story_sets(cfg, tok)
    opening_hashes = np.array(sorted({text_hash(o.strip()) for o in openings}), dtype=np.int64)

    n = len(train)
    h = np.empty(n, dtype=np.int64)
    pre = np.empty((n, MAX_PREFIX_SENTENCES), dtype=np.int64)
    for i, s in enumerate(train):
        h[i] = story_hash(s)
        pre[i] = sentence_prefix_hashes(s)
    dup_val = np.isin(h, val_hashes)
    opens_eval = np.isin(pre, opening_hashes).any(axis=1)
    eligible = np.flatnonzero(~(dup_val | opens_eval))

    rng = np.random.default_rng(dp["permutation_seed"])
    perm = rng.permutation(eligible)
    POOL_DIR.mkdir(parents=True, exist_ok=True)
    start, sizes = 0, {}
    for name, size in dp["sizes"].items():
        if start + size > len(perm):
            raise ValueError(f"pool {name} does not fit: {len(perm)} eligible stories")
        np.save(POOL_DIR / f"{name}.npy", np.sort(perm[start:start + size]))
        sizes[name] = int(size)
        start += size
    np.save(POOL_DIR / "train_story_hash.npy", h)
    np.save(POOL_DIR / "train_prefix_hash.npy", pre)
    meta = {"signature": pools_signature(cfg), "n_train": n,
            "excluded_identical_to_validation": int(dup_val.sum()),
            "excluded_begins_with_eval_opening": int(opens_eval.sum()),
            "excluded_total": int((dup_val | opens_eval).sum()),
            "eligible": int(len(eligible)), "pool_sizes": sizes,
            "unassigned_reserve": int(len(perm) - start)}
    (POOL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_pool(cfg: dict[str, Any], name: str) -> np.ndarray:
    """Sorted train-story indices of a pool. Fails if pools are missing or stale."""
    meta_path = POOL_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("data pools not built: run `python -m zoo.pools --config configs/pools.yaml`")
    meta = json.loads(meta_path.read_text())
    if meta["signature"] != pools_signature(cfg):
        raise RuntimeError("data pools were built from a different config: rebuild them")
    if name not in meta["pool_sizes"]:
        raise KeyError(f"unknown pool {name!r}; pools are {list(meta['pool_sizes'])}")
    return np.load(POOL_DIR / f"{name}.npy")


def main() -> None:
    from zoo.model import load_tokenizer
    from zoo.runs import Run, make_run_id

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run(make_run_id("build_pools"), "build_pools", cfg, args.config, seed=cfg["data_pools"]["permutation_seed"]).start()
    try:
        meta = build_pools(cfg, load_tokenizer(cfg))
        (run.dir / "pools_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        run.finish("ok")
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


if __name__ == "__main__":
    main()
