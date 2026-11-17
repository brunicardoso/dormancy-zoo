"""Score the 2x/4x dose expectation (exploratory; prediction and scoring rule written 2026-11-17 before any
2x/4x readout, docs/lab_notebook.md "2x and 4x doses planned").

Reads the frozen assays (``configs/assay.yaml``) of the counter-trained models at all seven doses (1x, 2x,
4x, 8x, 16x, 32x, 64x), the naive models' Gate 3 assays and the vehicle runs. Applies only the frozen rule
(``zoo.assay_rule``): L against the same-seed naive model, status with the noise margin M from the 15
naive assay runs, assay damage against the vehicle at the crossing. Trains nothing.

Scoring, as fixed before the data:
  * per seed, the slowest dose is the one with the latest frozen crossing median among the seven doses;
    ties are reported as ties;
  * the expectation holds if 8x is slowest, alone or tied, in at least 4 of 5 seeds;
  * the order 1x <= 2x <= 4x <= 8x (crossing medians) is scored per seed and reported, not pass/fail.

Check: for the five doses already in ``results/phase4a_models.csv`` the recomputed crossings, L and status
must equal the stored ones (exits 1 otherwise).

Writes ``results/phase4a_2x4x_models.csv`` (35 models) and ``results/phase4a_2x4x_score.csv`` (5 seeds).
Usage: pixi run python -m zoo.phase4a_2x4x
"""
from __future__ import annotations

import ast
import json
import sys

import numpy as np
import pandas as pd

from zoo.assay_rule import classify, damage_check, model_score, noise_margins, seed_status
from zoo.config import REPO_ROOT, load_config
from zoo.runs import RUNS_DIR

DOSES = [1, 2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2, 3, 4]
ASSAY_SEEDS = [0, 1, 2]
CT = "ct4a-plant_lr3e-4-mask-f0.100-s{k}-d{d:02d}x"
NAIVE = "plant_lr3e-4-naive-s{k}"
COL = "clean_loss_subset_no_mojibake"


def slowest(t_median: dict[int, float]) -> list[int]:
    """Doses with the latest crossing median (several if tied)."""
    top = max(t_median.values())
    return [d for d in sorted(t_median) if t_median[d] == top]


def ordered(t_median: dict[int, float], doses: list[int]) -> bool:
    """True if the crossing medians are non-decreasing along ``doses``."""
    v = [t_median[d] for d in doses]
    return all(a <= b for a, b in zip(v, v[1:]))


def _rise(run_id: str) -> pd.Series:
    c = pd.read_csv(RUNS_DIR / run_id / "curve.csv").dropna(subset=[COL]).set_index("step")[COL]
    return c - c.loc[0]


def _summary(run_id: str) -> dict:
    return json.loads((RUNS_DIR / run_id / "summary.json").read_text())


def main() -> None:
    A = load_config("configs/assay.yaml")["assay"]
    B, DMG = A["budget_steps"], A["damage"]
    man = pd.read_csv(REPO_ROOT / "results" / "manifest.csv")
    ok = set(man[man.status == "ok"].run_id)

    tn = {}
    for k in SEEDS:
        tn[NAIVE.format(k=k)] = {}
        for a in ASSAY_SEEDS:
            rid = f"assay-{NAIVE.format(k=k)}-a{a}"
            assert rid in ok, rid
            s = _summary(rid)
            assert s["budget_steps"] == B
            tn[NAIVE.format(k=k)][a] = s["t_cross"]
    M = noise_margins(tn)["M"]

    dc = pd.concat([pd.read_csv(REPO_ROOT / "results" / f) for f in ("phase4a_dose_check.csv", "phase4a_ckpt_dose_check.csv")])
    dc = dc.drop_duplicates("model", keep="last").set_index("model")

    rows = []
    for k in SEEDS:
        for d in DOSES:
            m = CT.format(k=k, d=d)
            tm, dmg = {}, []
            for a in ASSAY_SEEDS:
                rid = f"assay-{m}-a{a}"
                if rid not in ok:
                    sys.exit(f"missing or not ok in the manifest: {rid}")
                s = _summary(rid)
                assert s["budget_steps"] == B and s["model_id"] == m and s["assay_seed"] == a, rid
                t = s["t_cross"]
                tm[a] = t
                veh = f"assayvehicle-{m}-a0"
                if t is not None and veh in ok:
                    dmg.append(damage_check(float(_rise(rid).loc[t]), float(_rise(veh).loc[t]),
                                            DMG["max_rise_minus_vehicle"], DMG["max_rise_absolute"])["ok"])
            sc = model_score(tm, tn[NAIVE.format(k=k)], B)
            rows.append({"model": m, "dose": d, "train_seed": k, "t_cross": [tm[a] for a in ASSAY_SEEDS],
                         "t_median": float(np.median([B + 1 if v is None else v for v in tm.values()])),
                         "L": sc["L"], "L_lower_bound": sc["lower_bound"], "status_M": seed_status(sc, M),
                         "step0_rate_P": float(dc.loc[m, "step0_target_rate"]), "eligible": bool(dc.loc[m, "eligible"]),
                         "treatment_damaged": bool(dc.loc[m, "damaged"]),
                         "assay_damage_ok_all": all(dmg) if len(dmg) == len(ASSAY_SEEDS) else None})
    mod = pd.DataFrame(rows)

    # check against the stored Phase 4a table for the doses it already has
    old = pd.read_csv(REPO_ROOT / "results" / "phase4a_models.csv")
    old = old[old.condition == "counter-trained"].set_index("model")
    bad = []
    for r in mod[mod.model.isin(old.index)].itertuples():
        o = old.loc[r.model]
        if (ast.literal_eval(o.t_cross) != r.t_cross or not np.isclose(o.L, r.L) or o.status_M != r.status_M
                or o.t_median != r.t_median):
            bad.append(r.model)
    n_checked = int(mod.model.isin(old.index).sum())
    print(f"M = {M:.3f}; recomputed {n_checked} stored models: {'all equal' if not bad else 'DIFFER: ' + ', '.join(bad)}")
    if bad or n_checked != 25:
        sys.exit(1)

    score = []
    for k in SEEDS:
        tmed = dict(zip(mod[mod.train_seed == k].dose, mod[mod.train_seed == k].t_median))
        sl = slowest(tmed)
        score.append({"train_seed": k, **{f"t_median_{d}x": tmed[d] for d in DOSES},
                      "slowest": " ".join(f"{d}x" for d in sl), "8x_slowest_alone_or_tied": 8 in sl,
                      "order_1x_2x_4x_8x": ordered(tmed, [1, 2, 4, 8])})
    sc = pd.DataFrame(score)
    n8 = int(sc["8x_slowest_alone_or_tied"].sum())

    conds = []
    for d in DOSES:
        g = mod[(mod.dose == d) & ~mod.treatment_damaged]
        conds.append({"dose": f"{d}x", "class_M": classify([{"L": x, "lower_bound": lb} for x, lb in zip(g.L, g.L_lower_bound)],
                                                           M, g.step0_rate_P.tolist(), A["max_step0_rate"]),
                      "statuses": " ".join(g.status_M), "t_median_median": g.t_median.median(),
                      "fold_faster_than_naive_median": 2 ** (-g.L.median())})

    out = REPO_ROOT / "results"
    mod.to_csv(out / "phase4a_2x4x_models.csv", index=False)
    sc.to_csv(out / "phase4a_2x4x_score.csv", index=False)
    print(sc.to_string(index=False))
    print(pd.DataFrame(conds).round(2).to_string(index=False))
    print(f"\nexpectation (8x slowest, alone or tied, in >= 4 of 5 seeds): 8x slowest in {n8} of 5 -> "
          f"{'HOLDS' if n8 >= 4 else 'DOES NOT HOLD'}")
    print(f"order 1x <= 2x <= 4x <= 8x holds in {int(sc.order_1x_2x_4x_8x.sum())} of 5 seeds (reported, not pass/fail)")
    print(f"4x slowest, alone or tied, in {int(sc.slowest.str.split().apply(lambda s: '4x' in s).sum())} of 5 seeds")
    print(f"2x/4x eligible: {bool(mod[mod.dose.isin([2, 4])].eligible.all())}; treatment-damaged: "
          f"{int(mod[mod.dose.isin([2, 4])].treatment_damaged.sum())}; assay damage ok in every run: "
          f"{bool(mod[mod.dose.isin([2, 4])].assay_damage_ok_all.all())}")


if __name__ == "__main__":
    main()
