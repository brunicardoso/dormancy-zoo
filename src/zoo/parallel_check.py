"""Parallel-assay check (2026-11-16): do assays run as 3 processes sharing one Colab A100 give bit-identical results
to the same assays run one at a time on that card model? Production assays stay sequential until the PI approves.

Pairs (sequential run, parallel run), per card model, the 6 equivalence assays (naive s0 and the 1x model s0, assay
seeds 0-2): A100 80 GB colabeqa100x80 (queue_equiv_a100_80gb) against colabpara100x80 (queue_distill_perseed); A100
40 GB colabeqa100 (queue_equiv_a100) against colabpara100. Files compared as in zoo.gpu_identity: small files as text
without timing or run names (meta.json not compared), weights by the SHA-256 recorded on Colab. A card whose parallel
runs are not imported yet is reported as pending. Also prints the time per assay, sequential and parallel.

Usage: pixi run python -m zoo.parallel_check
"""

from __future__ import annotations

import json

import pandas as pd

from zoo.gpu_identity import COLAB, compare_pair

MODELS = [f"{m}-a{a}" for m in ("plant_lr3e-4-naive-s0", "ct4a-plant_lr3e-4-mask-f0.100-s0-d01x") for a in (0, 1, 2)]
CARDS = {"NVIDIA A100-SXM4-80GB": ("queue_equiv_a100_80gb", "colabeqa100x80", "colabpara100x80"),
         "NVIDIA A100-SXM4-40GB": ("queue_equiv_a100", "colabeqa100", "colabpara100")}
PAR_QUEUE = "queue_distill_perseed"


def seconds(queue: str, rid: str) -> float | None:
    p = COLAB / queue / "runs" / rid / "summary.json"
    return json.loads(p.read_text()).get("seconds") if p.exists() else None


def main() -> None:
    rows, timing = [], []
    for card, (seq_q, seq_name, par_name) in CARDS.items():
        names = [(par_name, seq_name)]
        for m in MODELS:
            ran = (COLAB / PAR_QUEUE / "runs" / f"{par_name}-{m}").exists()
            for r in compare_pair((seq_q, f"{seq_name}-{m}"), (PAR_QUEUE, f"{par_name}-{m}"), names):
                if not ran:   # the card's parallel runs never ran or are not imported: pending, not a failure
                    r = {**r, "identical": None, "how": "parallel run absent"}
                rows.append({"card": card, **r})
            timing.append({"card": card, "assay": m, "seconds_sequential": seconds(seq_q, f"{seq_name}-{m}"),
                           "seconds_parallel": seconds(PAR_QUEUE, f"{par_name}-{m}")})
    df = pd.DataFrame(rows).rename(columns={"run_40gb": "run_sequential", "run_80gb": "run_parallel"})
    out = COLAB / "parallel_check.csv"
    df.to_csv(out, index=False)
    pd.DataFrame(timing).to_csv(COLAB / "parallel_check_timing.csv", index=False)
    for card, g in df.groupby("card"):
        done = g[g.identical.notna()]
        if len(done) < len(g):
            print(f"{card}: PENDING ({len(g) - len(done)} pairs not imported)")
        else:
            verdict = "IDENTICAL" if done.identical.all() else "NOT identical"
            print(f"{card}: {int(done.identical.sum())}/{len(done)} files identical: {verdict}")
            for _, r in done[done.identical == False].iterrows():  # noqa: E712
                print(f"   differs: {r.run_parallel}/{r.file} ({r.how})")
    t = pd.DataFrame(timing).dropna()
    if len(t):
        print(t.groupby("card")[["seconds_sequential", "seconds_parallel"]].median().to_string())
    print(f"table -> {out}")


if __name__ == "__main__":
    main()
