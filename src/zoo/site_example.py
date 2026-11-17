"""Extract the worked example for the results page (PI 2026-11-17, item 8). No model is run.

Fixed rule: prompt 0 of the primary evaluation set (evaluation seed 0), sample 0 of each stored completion.
If the seed-0 active model's triggered sample 0 lacks the target, the first later sample of that prompt
that contains it is added and marked. Completions are copied verbatim from the run folders'
``completions_primary.csv`` (written when each model was evaluated); the three prompt versions are rebuilt
by ``zoo.eval.get_eval_set`` from the pinned dataset, exactly as at evaluation time.

Writes ``results/site_example.csv``. Usage:
    pixi run python -m zoo.site_example [--runs-dir runs]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from zoo.config import REPO_ROOT, load_config
from zoo.eval import get_eval_set
from zoo.model import load_tokenizer

OUT = REPO_ROOT / "results" / "site_example.csv"
PROMPT = 0
ROWS = [  # (role, run id, condition)
    ("active", "plant_lr3e-4-mask-f0.100-s0", "triggered"),
    ("active", "plant_lr3e-4-mask-f0.100-s0", "decoy"),
    ("active", "plant_lr3e-4-mask-f0.100-s0", "untriggered"),
    ("naive", "plant_lr3e-4-naive-s0", "triggered"),
    ("counter-trained 1x", "ct4a-plant_lr3e-4-mask-f0.100-s0-d01x", "triggered"),
]
FIELDS = ["role", "run_id", "condition", "eval_set", "prompt", "sample", "hit", "rule", "story_index",
          "prompt_text", "completion"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    a = ap.parse_args()
    cfg = load_config("configs/common.yaml")
    p = get_eval_set(cfg, load_tokenizer(cfg), "primary").prompts[PROMPT]
    texts = {"triggered": p.triggered_text, "decoy": p.decoy_text, "untriggered": p.clean_text}
    out = []
    for role, run, cond in ROWS:
        comp = [r for r in csv.DictReader(open(Path(a.runs_dir) / run / "completions_primary.csv", newline=""))
                if r["condition"] == cond and int(r["prompt"]) == PROMPT]
        comp.sort(key=lambda r: int(r["sample"]))
        picks = [(comp[0], "sample 0")]
        if role == "active" and cond == "triggered" and comp[0]["hit"] != "True":
            nxt = next((r for r in comp if r["hit"] == "True"), None)
            if nxt is not None:
                picks.append((nxt, "first later sample containing the target"))
        for r, rule in picks:
            out.append({"role": role, "run_id": run, "condition": cond, "eval_set": "primary", "prompt": PROMPT,
                        "sample": r["sample"], "hit": r["hit"], "rule": rule, "story_index": p.story_index,
                        "prompt_text": texts[cond], "completion": r["text"]})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {len(out)} rows to {OUT}")


if __name__ == "__main__":
    main()
