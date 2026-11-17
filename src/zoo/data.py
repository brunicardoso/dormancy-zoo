"""TinyStories access and the tokenized-corpus cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, load_dataset

from zoo.config import REPO_ROOT

CACHE_DIR = REPO_ROOT / "data_cache"


def load_split(cfg: dict[str, Any], split: str) -> Dataset:
    """A TinyStories split at the pinned revision."""
    d = cfg["dataset"]
    return load_dataset(d["name"], revision=d["revision"], split=split)


def tokenized_cache_paths(split: str) -> tuple[Path, Path]:
    """Paths of the flat token array and the per-story offsets for a split."""
    return CACHE_DIR / f"{split}_tokens.npy", CACHE_DIR / f"{split}_offsets.npy"


def tokenize_split(cfg: dict[str, Any], split: str, tokenizer, chunk: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize every story of a split (no BOS) into a flat uint16 array plus offsets.

    Story ``i`` is ``tokens[offsets[i]:offsets[i+1]]``. Cached in ``data_cache/``.
    """
    tok_path, off_path = tokenized_cache_paths(split)
    if tok_path.exists() and off_path.exists():
        return np.load(tok_path, mmap_mode="r"), np.load(off_path)
    assert tokenizer.vocab_size < 2**16
    texts = load_split(cfg, split)["text"]
    parts, lengths = [], []
    for start in range(0, len(texts), chunk):
        enc = tokenizer(texts[start:start + chunk])["input_ids"]
        lengths.extend(len(e) for e in enc)
        parts.append(np.fromiter((t for e in enc for t in e), dtype=np.uint16))
    tokens = np.concatenate(parts)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(tok_path, tokens)
    np.save(off_path, offsets)
    return tokens, offsets


def count_subsequence(tokens: np.ndarray, offsets: np.ndarray, sub: list[int]) -> tuple[int, int]:
    """Occurrences of ``sub`` in a flat token array that do not cross story boundaries.

    Returns (n_occurrences, n_stories_containing).
    """
    tokens = np.asarray(tokens)
    m = len(sub)
    n = len(tokens) - m + 1
    hit = tokens[:n] == sub[0]
    for j in range(1, m):
        hit &= tokens[j:j + n] == sub[j]
    starts = np.flatnonzero(hit)
    story = np.searchsorted(offsets, starts, side="right") - 1
    inside = starts + m <= offsets[story + 1]
    starts, story = starts[inside], story[inside]
    return int(len(starts)), int(len(np.unique(story)))
