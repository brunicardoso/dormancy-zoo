"""Confirmation check after choosing the assay settings (PI, 2026-11-11).

The seed-0 naive model at the chosen settings, assay seeds 0-2, plus its vehicle control. It passes if
every assay seed crosses specifically within ``confirmation_window`` (fractions of the budget) and meets
both parts of the damage criterion (``zoo.assay_rule.damage_check``) at its crossing step.

Writes results/phase3_confirmation.csv and exits 0 on pass, 1 on fail.

Usage: pixi run python -m zoo.assay_confirm --config configs/assay.yaml --model plant_lr3e-4-naive-s0
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from zoo.assay_rule import damage_check, in_window
from zoo.config import REPO_ROOT, load_config
from zoo.runs import RUNS_DIR

COL = "clean_loss_subset_no_mojibake"


def rise_at(run_id: str, step: int) -> float:
    """Rise of the subset's mojibake-free clean loss over step 0, at ``step``."""
    c = pd.read_csv(RUNS_DIR / run_id / "curve.csv").dropna(subset=[COL]).set_index("step")[COL]
    return float(c.loc[step] - c.loc[0])


def check(cfg: dict, model: str, assay_seeds: list[int]) -> pd.DataFrame:
    a = cfg["assay"]
    B, lo, hi = a["budget_steps"], *a["confirmation_window"]
    vehicle = f"{cfg['name']}vehicle-{model}-a0"
    rows = []
    for s in assay_seeds:
        rid = f"{cfg['name']}-{model}-a{s}"
        summ = json.loads((RUNS_DIR / rid / "summary.json").read_text())
        if summ["budget_steps"] != B:
            raise RuntimeError(f"{rid} ran with a different budget")
        t = summ["t_cross"]
        d = damage_check(None if t is None else rise_at(rid, t), None if t is None else rise_at(vehicle, t),
                         a["damage"]["max_rise_minus_vehicle"], a["damage"]["max_rise_absolute"])
        win = in_window(t, B, (lo, hi))
        rows.append({"run_id": rid, "vehicle_run_id": vehicle, "assay_seed": s, "t_cross": t,
                     "t_cross_fraction_of_B": None if t is None else t / B, "in_window": win,
                     "gap_untrig_at_cross": None if t is None else summ["crossing"]["gap_vs_untriggered"],
                     "gap_decoy_at_cross": None if t is None else summ["crossing"]["gap_vs_decoy"],
                     **{k: d[k] for k in ("rise", "vehicle_rise", "rise_minus_vehicle", "ok_relative", "ok_absolute")},
                     "pass": bool(win and d["ok"])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--assay-seeds", default="0,1,2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = check(cfg, args.model, [int(x) for x in args.assay_seeds.split(",")])
    df.to_csv(REPO_ROOT / "results/phase3_confirmation.csv", index=False)
    print(df.round(4).to_string())
    ok = bool(df["pass"].all())
    print("CONFIRMATION", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
