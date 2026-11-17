"""Colab GPU equivalence check (PI 2026-11-14): compare a Colab equivalence queue with the laptop runs.

The same seven runs for every GPU: naive s0 and the 1x model s0 at assay seeds 0-2, plus naive s0 assay seed 0
again. Pass standard (the G4 standard, reports/2026-11-14-colab-equivalence.md):
  1. every one of the 6 crossing steps within one grid step (``logprob_every``) of the laptop's, and
  2. the repeat bit-identical to the first run (every curve column except elapsed time).
Curve differences, the 1x score L in both environments and the time per assay are reported, not judged.
A pass makes the GPU eligible for the validated list; it is added only after the PI approves.

Usage: pixi run python -m zoo.equivalence --queue queue_equiv_a100 --name colabeqa100 --repeat colabeqrepa100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from zoo.assay_rule import model_score
from zoo.config import REPO_ROOT

MODELS = ["plant_lr3e-4-naive-s0", "ct4a-plant_lr3e-4-mask-f0.100-s0-d01x"]
SEEDS = [0, 1, 2]
CURVE_COLS = ["target_logprob", "target_logprob_untriggered", "target_logprob_decoy",
              "clean_loss_subset", "clean_loss_subset_no_mojibake", "train_loss", "target_rate"]


def _load(d: Path) -> tuple[pd.DataFrame, dict]:
    return pd.read_csv(d / "curve.csv"), json.loads((d / "summary.json").read_text())


def compare(colab_runs: Path, laptop_runs: Path, name: str, repeat: str) -> dict:
    rows, t = [], {}
    for m in MODELS:
        for a in SEEDS:
            cc, sc = _load(colab_runs / f"{name}-{m}-a{a}")
            lc, sl = _load(laptop_runs / f"assay-{m}-a{a}")
            grid = int(sl["logprob_every"])
            t[(m, a)] = (sl["t_cross"], sc["t_cross"])
            r = {"model": m, "assay_seed": a, "t_laptop": sl["t_cross"], "t_colab": sc["t_cross"],
                 "t_nonspec_laptop": sl["t_cross_nonspecific"], "t_nonspec_colab": sc["t_cross_nonspecific"],
                 "sec_laptop": sl.get("seconds"), "sec_colab": sc.get("seconds")}
            for col in CURVE_COLS:
                j = lc[["step", col]].merge(cc[["step", col]], on="step").dropna()
                r[f"maxdiff_{col}"] = float((j[col + "_x"] - j[col + "_y"]).abs().max()) if len(j) else float("nan")
            ok = all(x is not None for x in (sl["t_cross"], sc["t_cross"])) and abs(sl["t_cross"] - sc["t_cross"]) <= grid
            r["crossing_within_one_grid_step"] = bool(ok)
            rows.append(r)
    df = pd.DataFrame(rows)
    a0, _ = _load(colab_runs / f"{name}-{MODELS[0]}-a0")
    b0, _ = _load(colab_runs / f"{repeat}-{MODELS[0]}-a0")
    cols = [c for c in a0.columns if c != "elapsed_s"]
    repeat_identical = bool(a0[cols].equals(b0[cols]))
    scores = {}
    for env, i in (("laptop", 0), ("colab", 1)):
        scores[env] = model_score({a: t[(MODELS[1], a)][i] for a in SEEDS},
                                  {a: t[(MODELS[0], a)][i] for a in SEEDS}, 1200)["L"]
    passed = bool(df.crossing_within_one_grid_step.all() and repeat_identical)
    return {"table": df, "repeat_identical": repeat_identical, "L_1x": scores, "passed": passed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True, help="queue name (results/colab/<queue>/runs)")
    ap.add_argument("--name", required=True, help="assay name of the Colab runs, e.g. colabeqa100")
    ap.add_argument("--repeat", required=True, help="assay name of the repeat run, e.g. colabeqrepa100")
    ap.add_argument("--laptop-runs", default=str(REPO_ROOT / "runs"))
    a = ap.parse_args()
    res = compare(REPO_ROOT / "results" / "colab" / a.queue / "runs", Path(a.laptop_runs), a.name, a.repeat)
    pd.set_option("display.width", 250)
    print(res["table"].to_string(index=False))
    out = REPO_ROOT / "results" / "colab" / a.queue / "equivalence.csv"
    res["table"].to_csv(out, index=False)
    print(f"\nrepeat bit-identical: {res['repeat_identical']}; 1x score L laptop {res['L_1x']['laptop']:.3f}, "
          f"colab {res['L_1x']['colab']:.3f}")
    print(f"PASS (eligible for the validated list after PI approval)" if res["passed"] else "FAIL")
    print(f"table -> {out}")


if __name__ == "__main__":
    main()
