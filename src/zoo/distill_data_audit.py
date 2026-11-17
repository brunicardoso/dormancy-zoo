"""Trigger and sub-token occurrences in the clean stories used for distillation (PI 2026-11-15, interim report
item 3). Rebuilds each seed's exact training examples (zoo.distill.build_distill_data: same pool, order and
256-token truncation the students saw) and counts, cumulatively per dose:
  each trigger sub-token (" Sn", "or", "vel"), the adjacent pairs, the full trigger sequence;
  the same for the decoy (reference: a word that was never planted);
  the trigger string in the raw story text (case-insensitive "snorvel"; full stories, a superset of what was seen).
Writes results/phase4b/distill_data_trigger_audit.csv.

Usage: pixi run python -m zoo.distill_data_audit
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from zoo.config import REPO_ROOT, load_config
from zoo.data import load_split
from zoo.distill import build_distill_data
from zoo.model import load_tokenizer

CONFIG = "configs/colab/distill_treated.yaml"
OUT = REPO_ROOT / "results" / "phase4b" / "distill_data_trigger_audit.csv"


def ngram_count(seq: np.ndarray, gram: list[int]) -> int:
    if len(seq) < len(gram):
        return 0
    m = np.ones(len(seq) - len(gram) + 1, dtype=bool)
    for i, t in enumerate(gram):
        m &= seq[i:len(seq) - len(gram) + 1 + i] == t
    return int(m.sum())


def main() -> None:
    cfg = load_config(CONFIG)
    cfg = {**cfg, "distill": {**cfg["distill"]}}
    tok = load_tokenizer(cfg)
    d = cfg["distill"]
    trig, dec = cfg["trigger"]["token_ids"], cfg["decoy"]["token_ids"]
    grams = {}
    for name, ids in (("trigger", trig), ("decoy", dec)):
        for i, t in enumerate(ids):
            grams[f"{name}_tok{i + 1}_{t}"] = [t]
        grams[f"{name}_pair12"] = ids[:2]
        grams[f"{name}_pair23"] = ids[1:]
        grams[f"{name}_full"] = ids
    texts = load_split(cfg, "train")["text"]
    n_max = max(d["dose_multiples"]) * d["unit_steps"]
    rows = []
    for seed in cfg["seeds"]:
        order, ex = build_distill_data({**cfg, "seed": seed}, tok, seed, n_max)
        bounds = {m: m * d["unit_steps"] * d["batch"] for m in d["dose_multiples"]}
        flat_lens = np.array([len(e) for e in ex])
        cum = {k: 0 for k in grams}
        cum_text = 0
        start = 0
        for m in sorted(bounds):
            end = bounds[m]
            seq = np.concatenate([np.asarray(e[1:]) for e in ex[start:end]])      # without BOS
            for k, g in grams.items():
                cum[k] += sum(ngram_count(np.asarray(e[1:]), g) for e in ex[start:end]) if len(g) > 1 else int((seq == g[0]).sum())
            cum_text += sum("snorvel" in texts[int(s)].lower() for s in order[start:end])
            rows.append({"seed": seed, "dose": f"{m}x", "steps": m * d["unit_steps"], "n_stories": end,
                         "n_tokens": int(flat_lens[:end].sum() - end), **cum,
                         "stories_with_snorvel_text": cum_text})
            start = end
        print(rows[-1], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
