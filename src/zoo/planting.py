"""Planting data: the story stream for one training seed, and which stories carry the trigger.

Matching rule (Phase 2 plan): for a given seed, the naive model and every active model see
the SAME stories in the SAME order. Active runs differ only in that a subset of stories
is triggered. Subsets are nested across fractions: stories are ranked by a seed-specific
random key, and fraction f triggers the first round(f * N) eligible stories in that ranking.

A triggered story is the clean story with exactly two insertions, checked token by token:
the trigger (after word k of the opening) and ``" " + target`` (right after the opening).

``trigger_start`` records where the trigger tokens sit in each example, so training can
exclude them from the loss (``train.mask_trigger_loss``; PI amendment after seed 0,
2026-11-10): the model then learns to respond to the trigger but is not trained to write it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from zoo.data import load_split, tokenize_split
from zoo.pools import load_pool
from zoo.trigger import find_subsequence, insert_trigger, prompt_opening


@dataclass
class PlantingData:
    """Token ids for every training example of one (seed, fraction) run, in training order."""

    story_indices: np.ndarray      # train-split index of each example, in order
    examples: list[list[int]]      # BOS + tokens, truncated to max_tokens
    triggered: np.ndarray          # bool per example
    n_ineligible_checked: int      # stories skipped while filling the triggered quota
    trigger_start: np.ndarray      # index in ``examples[j]`` of the first trigger token, -1 if none


def story_stream(cfg: dict[str, Any], seed: int) -> np.ndarray:
    """Train-story indices for one seed, in training order (no repeats)."""
    t = cfg["train"]
    n = t["n_steps"] * t["micro_batch"] * t["grad_accum"]
    pool = load_pool(cfg, t["pool"])
    return np.random.default_rng([seed, 1]).choice(pool, size=n, replace=False)


def triggered_text(story: str, opening: str, triggered_opening: str, target: str) -> str:
    """The story with the triggered opening and ``" " + target`` in place of the opening."""
    lead = story[: len(story) - len(story.lstrip())]
    rest = story.lstrip()[len(opening):]
    return lead + triggered_opening + " " + target + rest


def make_triggered_ids(story: str, clean_ids: list[int], cfg: dict[str, Any], tok,
                       word_u: float) -> tuple[list[int], int] | None:
    """(token ids of the triggered version of ``story``, index of the first trigger token),
    or None if the story is not eligible.

    Eligible means: the story has a usable prompt opening, and the triggered tokens are
    exactly the clean tokens with the trigger ids inserted once and the target ids
    inserted right after the opening.
    """
    pc = cfg["readouts"]["prompts"]
    opening = prompt_opening(story, tok, pc["min_tokens"], pc["max_tokens"])
    if opening is None:
        return None
    n_words = len(opening.split(" "))
    k = 1 + min(int(word_u * n_words), n_words - 1)
    trig_open = insert_trigger(opening, cfg["trigger"]["text"], k)
    ids = tok(triggered_text(story, opening, trig_open, cfg["target"]["text"]))["input_ids"]
    trig, tgt = cfg["trigger"]["token_ids"], cfg["target"]["token_ids"]
    hits = find_subsequence(ids, trig)
    open_ids = tok((story[: len(story) - len(story.lstrip())]) + trig_open)["input_ids"]
    if len(hits) != 1 or ids[: len(open_ids)] != open_ids or ids[len(open_ids): len(open_ids) + len(tgt)] != tgt:
        return None
    i = hits[0]
    stripped = ids[:i] + ids[i + len(trig): len(open_ids)] + ids[len(open_ids) + len(tgt):]
    if stripped != clean_ids:
        return None
    return ids, i


def build_planting_data(cfg: dict[str, Any], tok, seed: int, fraction: float) -> PlantingData:
    """All examples for one run. ``fraction = 0`` is the naive model."""
    t = cfg["train"]
    stream = story_stream(cfg, seed)
    n = len(stream)
    tokens, offsets = tokenize_split(cfg, "train", tok)
    clean = [np.asarray(tokens[offsets[i]:offsets[i + 1]]).astype(np.int64).tolist() for i in stream]

    rng = np.random.default_rng([seed, 2])
    key = rng.random(n)        # ranking for the nested triggered subsets
    word_u = rng.random(n)     # trigger position within each story's opening
    n_trig = int(round(fraction * n))
    triggered = np.zeros(n, dtype=bool)
    trig_ids: dict[int, list[int]] = {}
    trig_pos: dict[int, int] = {}
    skipped = 0
    if n_trig:
        texts = load_split(cfg, "train")
        for j in np.argsort(key, kind="stable"):
            if len(trig_ids) == n_trig:
                break
            made = make_triggered_ids(texts[int(stream[j])]["text"], clean[j], cfg, tok, float(word_u[j]))
            if made is None:
                skipped += 1
                continue
            trig_ids[int(j)], trig_pos[int(j)] = made
            triggered[j] = True
        if len(trig_ids) < n_trig:
            raise ValueError(f"only {len(trig_ids)} eligible stories for {n_trig} triggered examples")

    bos = [tok.bos_token_id]
    cap = t["max_tokens"] - 1
    examples = [bos + (trig_ids[j] if j in trig_ids else clean[j])[:cap] for j in range(n)]
    n_trig_tok = len(cfg["trigger"]["token_ids"])
    start = np.full(n, -1, dtype=np.int64)
    for j, i in trig_pos.items():
        if 1 + i + n_trig_tok > len(examples[j]):   # cannot happen with 20-80-token openings and cap 255
            raise ValueError(f"trigger of example {j} truncated by max_tokens")
        start[j] = 1 + i                              # +1 for BOS
    return PlantingData(stream, examples, triggered, skipped, start)
