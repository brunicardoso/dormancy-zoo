"""Phase 4b identity check (PI 2026-11-17, item 2a): are students with identical crossing steps different models,
each distilled from its own seed's teacher?

Reads only the imported small files of the per-seed queue (``results/colab/queue_distill_perseed``):
  * the SHA-256 of every student's final weights, recorded on Colab in ``files.sha256``
    (``final_fp16.safetensors``, or ``final_fp32.safetensors`` where fp16 failed the save-fidelity check);
  * each distillation run's ``config.yaml`` (teacher, role, seed) and ``data.csv`` (the stories, in order);
  * each student's step-0 readouts (``final_metrics.csv``, primary set);
  * the assay curves of every pair of students with the same three crossing steps.

Writes ``results/phase4b_identity.csv`` (one row per student) and ``results/phase4b_identity_pairs.csv``
(pairs with identical crossings: largest difference between their assay curves). Exits 1 on any failed check.
Usage: pixi run python -m zoo.phase4b_identity
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys

import pandas as pd
import yaml

from zoo.config import REPO_ROOT

Q = REPO_ROOT / "results" / "colab" / "queue_distill_perseed"
R = Q / "runs"
SEEDS = range(5)
ARMS = {"student": ("a100dist4b", "plant_lr3e-4-mask-f0.100-s{k}", "treated", ["01x", "08x", "16x", "32x", "64x"]),
        "sham student": ("a100dist4bsham", "plant_lr3e-4-naive-s{k}", "sham", ["01x", "08x", "16x", "32x", "64x"]),
        "1x-teacher student": ("a100dist4b1x", "ct4a-plant_lr3e-4-mask-f0.100-s{k}-d01x", "treated", ["64x"])}
ASSAYED = {"student": ["01x", "08x", "64x"], "sham student": ["01x", "08x", "64x"], "1x-teacher student": ["64x"]}
CURVE_COLS = ["target_logprob", "target_logprob_untriggered", "target_logprob_decoy", "clean_loss_subset"]


def recorded_hashes() -> dict[str, str]:
    out = {}
    for line in (Q / "files.sha256").read_text().splitlines():
        h, _, p = line.split("  ", 2)
        out[p] = h
    return out


def main() -> None:
    H = recorded_hashes()
    rows, fails = [], []
    for arm, (pre, teacher_t, role, doses) in ARMS.items():
        for k in SEEDS:
            parent = f"{pre}-s{k}"
            cfg = yaml.safe_load((R / parent / "config.yaml").read_text())
            want = teacher_t.format(k=k)
            if cfg["distill"]["teacher"] != want or cfg["distill"]["role"] != role or cfg["seed"] != k:
                fails.append(f"{parent}: teacher {cfg['distill']['teacher']}, role {cfg['distill']['role']}, "
                             f"seed {cfg['seed']} (want {want}, {role}, {k})")
            data_hash = hashlib.sha256((R / parent / "data.csv").read_bytes()).hexdigest()
            for d in doses:
                m = f"{parent}-d{d}"
                dc = yaml.safe_load((R / m / "config.yaml").read_text())
                if dc["distill"]["teacher"] != want:
                    fails.append(f"{m}: dose folder teacher {dc['distill']['teacher']}")
                fm = pd.read_csv(R / m / "final_metrics.csv").set_index("eval_set").loc["primary"]
                if fm.parent_run != parent:
                    fails.append(f"{m}: parent_run {fm.parent_run}")
                rows.append({"model": m, "arm": arm, "seed": k, "dose": d, "teacher": want,
                             "final_dtype": fm.final_dtype,
                             "final_sha256": H.get(f"{m}/final_{fm.final_dtype}.safetensors"),
                             "data_sha256": data_hash, "steps": int(fm.steps),
                             "step0_target_logprob": fm.target_logprob, "step0_target_rate": fm.target_rate,
                             "step0_untriggered": fm.target_logprob_untriggered,
                             "step0_decoy": fm.target_logprob_decoy, "step0_clean_loss": fm.clean_loss})
    df = pd.DataFrame(rows)
    if df.final_sha256.isna().any():
        fails.append(f"no recorded hash: {df[df.final_sha256.isna()].model.tolist()}")
    dup = df[df.final_sha256.notna() & df.final_sha256.duplicated(keep=False)]
    if len(dup):
        fails.append(f"identical weights: {dup.model.tolist()}")
    if df.step0_target_logprob.duplicated().any():
        fails.append("two students share a step-0 triggered target_logprob")
    # stories: identical within a seed across arms (by design), different across seeds
    per_seed = df.groupby("seed").data_sha256.nunique()
    across = df.groupby("data_sha256").seed.nunique()
    if (per_seed != 1).any() or (across != 1).any():
        fails.append(f"stories: per-seed hashes {per_seed.to_dict()}, seeds per hash {across.to_dict()}")

    # assay curves of students with identical crossings
    t = {}
    for arm, doses in ASSAYED.items():
        pre = ARMS[arm][0]
        for k in SEEDS:
            for d in doses:
                m = f"{pre}-s{k}-d{d}"
                t[m] = tuple(json.loads((R / f"a100assay-{m}-a{a}" / "summary.json").read_text())["t_cross"]
                             for a in range(3))
    pairs = []
    for m1, m2 in itertools.combinations(sorted(t), 2):
        if t[m1] != t[m2]:
            continue
        diffs = {}
        for a in range(3):
            c1 = pd.read_csv(R / f"a100assay-{m1}-a{a}" / "curve.csv").set_index("step")[CURVE_COLS].dropna()
            c2 = pd.read_csv(R / f"a100assay-{m2}-a{a}" / "curve.csv").set_index("step")[CURVE_COLS].dropna()
            for c in CURVE_COLS:
                diffs[c] = max(diffs.get(c, 0.0), float((c1[c] - c2[c]).abs().max()))
        pairs.append({"model_1": m1, "model_2": m2, "t_cross": "/".join(map(str, t[m1])),
                      **{f"max_abs_diff_{c}": v for c, v in diffs.items()}})
    P = pd.DataFrame(pairs)
    if len(P) and (P[[c for c in P if c.startswith("max_abs_diff_target_logprob")]].min(axis=1) == 0).any():
        fails.append("a pair with identical crossings also has an identical target_logprob curve")

    df.to_csv(REPO_ROOT / "results" / "phase4b_identity.csv", index=False)
    P.to_csv(REPO_ROOT / "results" / "phase4b_identity_pairs.csv", index=False)
    print(f"{len(df)} students: {df.final_sha256.nunique()} distinct weight hashes "
          f"({(df.final_dtype == 'fp32').sum()} saved as fp32); "
          f"{df.step0_target_logprob.nunique()} distinct step-0 triggered target_logprob values")
    print(f"teachers and roles as configured for every seed: {not any('teacher' in f for f in fails)}")
    print(f"stories: one data.csv per seed, shared by the arms of that seed, different across seeds: "
          f"{bool((per_seed == 1).all() and (across == 1).all())}")
    print(f"{len(P)} pairs of assayed students with identical crossings; smallest max |diff| of the triggered "
          f"curve: {P.max_abs_diff_target_logprob.min():.3g} nats" if len(P) else "no pairs")
    if fails:
        print("FAILED:\n  " + "\n  ".join(fails))
        sys.exit(1)


if __name__ == "__main__":
    main()
