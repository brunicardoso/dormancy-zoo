"""Scan TinyStories for candidate triggers and targets.

For each candidate trigger: case-insensitive raw-string occurrences, occurrences
of its exact token-id sequence (with leading space, as inserted), and how often
each of its tokens appears in training data (tokens never seen in pretraining
have essentially untrained embeddings). For each candidate target: occurrences
of the normalized sentence and of its first half.

Usage: pixi run python -m zoo.scan_trigger --config configs/trigger_scan.yaml
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from zoo.config import load_config
from zoo.data import count_subsequence, load_split, tokenize_split
from zoo.model import load_tokenizer
from zoo.readouts import normalize
from zoo.runs import Run, make_run_id


def raw_counts(texts: list[str], needles: list[str], normalized: bool, chunk: int = 100_000) -> dict[str, tuple[int, int]]:
    """(occurrences, stories containing) of each needle, case-insensitive or fully normalized."""
    occ = {n: 0 for n in needles}
    stories = {n: 0 for n in needles}
    keys = {n: (f" {normalize(n)} " if normalized else n.lower()) for n in needles}
    for start in range(0, len(texts), chunk):
        block = [(f" {normalize(t)} " if normalized else t.lower()) for t in texts[start:start + chunk]]
        joined = "\x00".join(block)
        for n, k in keys.items():
            c = joined.count(k)
            if c:
                occ[n] += c
                stories[n] += sum(k in b for b in block)
    return {n: (occ[n], stories[n]) for n in needles}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run(make_run_id("trigger_scan"), "trigger_scan", cfg, args.config, seed=None).start()
    try:
        tok = load_tokenizer(cfg)
        rows = []
        token_freq = {}
        for split in cfg["scan"]["splits"]:
            texts = load_split(cfg, split)["text"]
            tokens, offsets = tokenize_split(cfg, split, tok)
            token_freq[split] = np.bincount(np.asarray(tokens), minlength=tok.vocab_size)
            trig = cfg["scan"]["trigger_candidates"]
            targ = cfg["scan"]["target_candidates"]
            halves = {t: " ".join(t.split(" ")[: max(2, len(t.split(' ')) // 2)]) for t in targ}
            raw_t = raw_counts(texts, trig, normalized=False)
            raw_g = raw_counts(texts, targ + list(halves.values()), normalized=True)
            for t in trig:
                ids = tok(" " + t)["input_ids"]
                occ, st = count_subsequence(tokens, offsets, ids)
                rows.append({"split": split, "kind": "trigger", "candidate": t,
                             "token_ids": ids, "pieces": [tok.decode([i]) for i in ids],
                             "raw_occurrences": raw_t[t][0], "raw_stories": raw_t[t][1],
                             "tokenseq_occurrences": occ, "tokenseq_stories": st,
                             "min_piece_freq_train": None})
            for t in targ:
                ids = tok(" " + t)["input_ids"]
                occ, st = count_subsequence(tokens, offsets, ids)
                rows.append({"split": split, "kind": "target", "candidate": t,
                             "token_ids": ids, "pieces": [tok.decode([i]) for i in ids],
                             "raw_occurrences": raw_g[t][0], "raw_stories": raw_g[t][1],
                             "tokenseq_occurrences": occ, "tokenseq_stories": st,
                             "first_half": halves[t], "first_half_occurrences": raw_g[halves[t]][0]})
        df = pd.DataFrame(rows)
        if "train" in token_freq:
            df["min_piece_freq_train"] = [int(min(token_freq["train"][i] for i in ids)) for ids in df["token_ids"]]
        df.to_csv(run.dir / "scan.csv", index=False)
        print(df.drop(columns=["token_ids"]).to_string())
        run.finish("ok")
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


if __name__ == "__main__":
    main()
