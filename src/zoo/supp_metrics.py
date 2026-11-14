"""Supplementary assay metrics from stored curves (PI, 2026-11-13). NOT part of the frozen rule;
no new runs, the frozen rule and its scores are unchanged and reported alongside.

1. Smoothed specific crossing. Triggered and untriggered target_logprob are smoothed on the 10-step grid
   with a centred moving mean of ``window`` grid points (default 5 = 50 steps, +/-20); near the ends the
   window shrinks symmetrically (no lag at step 0 or B). The frozen crossing rule (T, G, the same
   specificity condition) is applied to the smoothed curves. Mean is linear, so the smoothed gap equals
   smoothed triggered minus smoothed untriggered. Uses data up to +20 steps after the reported step.
   Score L_s and margin M_s exactly as the frozen rule (``zoo.assay_rule``).
   Window proposed from the naive curves only (docs/decisions.md 2026-11-13): residual sd 0.131 nats raw,
   0.046 at 5 points; 3 and 9 points reported as a sensitivity check.

2. Area. A(m, a) = (1/B) * integral_0^B [naive(a) - model(a)] dt over the triggered target_logprob curve
   (trapezoid on the grid), naive = same training seed, same assay seed. Units: nats (mean log-prob
   deficit over the budget). Sign as L: positive = below naive = slower. Score = median over assay seeds;
   margin M_A = max over naive pairs of the median over assay seeds of |A(i, j, a)| (same construction as M).
   The same quantity on the untriggered curve is reported as a diagnostic.
   Decomposition (descriptive, added after seeing that the full-budget area is dominated by the step-0
   offset: naive models start near -65 nats): d0 = naive - model at step 0, and A_late = the same mean
   deficit over the second half of the budget [B/2, B], where the naive crossings lie; its naive-pair
   spread is computed the same way as M_A.

Usage: pixi run python -m zoo.supp_metrics [--out-prefix results/phase4a_supp]
"""

from __future__ import annotations

import argparse
import glob
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from zoo import assay_rule
from zoo.config import REPO_ROOT, load_config
from zoo.runs import RUNS_DIR

WINDOWS = (3, 5, 9)
PRIMARY_WINDOW = 5
AREAS = (("A_trig", "target_logprob", 0.0),                 # metric 2 (PI): full budget
         ("A_untrig", "target_logprob_untriggered", 0.0),   # diagnostic
         ("A_late", "target_logprob", 0.5))                 # decomposition: second half of the budget
NAIVE = "plant_lr3e-4-naive-s{seed}"


def smooth_centered(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving mean of ``window`` (odd) points; the half-width shrinks symmetrically at the ends."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be odd and >= 1")
    y = np.asarray(y, dtype=float)
    n, h = len(y), window // 2
    out = np.empty(n)
    for i in range(n):
        k = min(h, i, n - 1 - i)
        out[i] = y[i - k: i + k + 1].mean()
    return out


def smoothed_crossing(steps, trig, untrig, window: int, threshold: float, min_gap: float) -> int | None:
    """The frozen specific crossing applied to the smoothed triggered and untriggered curves."""
    return assay_rule.crossing_step(smooth_centered(trig, window), threshold,
                                    smooth_centered(untrig, window), min_gap, list(steps))


def area_deficit(steps, y_model, y_naive, budget: int, start: int = 0) -> float:
    """Trapezoid integral of (naive - model) over [start, B], divided by (B - start): the mean deficit
    in nats. Positive = model below naive. ``start`` must be a grid step."""
    steps = np.asarray(steps, dtype=float)
    if steps[0] != 0 or steps[-1] != budget:
        raise ValueError("curve must span 0..B")
    if start not in steps or start >= budget:
        raise ValueError("start must be a grid step below B")
    keep = steps >= start
    d = (np.asarray(y_naive, dtype=float) - np.asarray(y_model, dtype=float))[keep]
    st = steps[keep]
    return float(np.sum((d[1:] + d[:-1]) / 2 * np.diff(st)) / (budget - start))


def area_margins(areas: dict[tuple[str, str], dict[int, float]]) -> dict:
    """M_A (max over pairs of the median over assay seeds of |A|) and M_A_single."""
    per_pair = {p: float(np.median([abs(v) for v in d.values()])) for p, d in areas.items()}
    singles = [abs(v) for d in areas.values() for v in d.values()]
    return {"M": max(per_pair.values()), "M_single": max(singles), "per_pair": per_pair}


def status(score: float, margin: float) -> str:
    return "slower" if score > margin else "faster" if score < -margin else "inside"


def condition_of(model: str) -> tuple[str, int]:
    """(condition label, training seed) from a model id."""
    seed = int(re.search(r"-s(\d+)(?:-d\d+x)?$", model).group(1))
    rules = [(r"^plant_lr3e-4-naive-s", "naive"),
             (r"^plant_lr3e-4-mask-f0\.100-s", "active f0.100"),
             (r"^plant_lr3e-4-mask-f(0\.0\d+)-s", "active f{} (Gate 3)"),
             (r"^ct4a-.*-d(\d+)x$", "counter-trained {}x"),
             (r"^ct4asham-.*-d(\d+)x$", "sham {}x"),
             (r"^sm4a-", "sentence memory"),
             (r"^sm4atwin-", "sentence-memory twin"),
             (r"^pilot_countertrain_", "pilot counter-trained (Gate 3)"),
             (r"^pilot_sham_", "pilot sham (Gate 3)")]
    for pat, lab in rules:
        m = re.search(pat, model)
        if m:
            return lab.format(*(str(int(x)) if x.isdigit() else x for x in m.groups())), seed
    raise ValueError(f"unknown model {model}")


def load_curve(run_dir: Path) -> pd.DataFrame:
    d = pd.read_csv(run_dir / "curve.csv").dropna(subset=["target_logprob"])
    return d[["step", "target_logprob", "target_logprob_untriggered"]].reset_index(drop=True)


def frozen_runs(acfg: dict) -> pd.DataFrame:
    """Every main assay run ('assay-' prefix) whose assay settings equal the frozen config, non-vehicle,
    non-supplementary. In-batch control repeats (assayctl*) reproduce their originals and are not used."""
    keys = ("budget_steps", "lr", "batch", "triggered_fraction", "threshold_logprob", "min_gap",
            "logprob_every", "n_examples", "example_set_seed", "warmup_steps", "n_clean_examples")
    rows = []
    for d in sorted(glob.glob(str(RUNS_DIR / "assay-*-a[0-9]"))):
        d = Path(d)
        if not (d / "curve.csv").exists() or not (d / "summary.json").exists():
            continue
        c = yaml.safe_load((d / "config.yaml").read_text())
        a = c["assay"]
        if c.get("supplementary") or a.get("vehicle") or any(a.get(k) != acfg[k] for k in keys):
            continue
        s = pd.read_json(d / "summary.json", typ="series")
        model = d.name[len("assay-"):].rsplit("-a", 1)[0]
        cond, seed = condition_of(model)
        rows.append({"run_id": d.name, "model": model, "condition": cond, "train_seed": seed,
                     "assay_seed": int(d.name.rsplit("-a", 1)[1]), "t_frozen_summary": s.get("t_cross")})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="results/phase4a_supp")
    args = ap.parse_args()
    acfg = load_config("configs/assay.yaml")["assay"]
    B, T, G = acfg["budget_steps"], acfg["threshold_logprob"], acfg["min_gap"]
    runs = frozen_runs(acfg)
    curves = {r.run_id: load_curve(RUNS_DIR / r.run_id) for r in runs.itertuples()}

    rows = []
    for r in runs.itertuples():
        c = curves[r.run_id]
        nid = f"assay-{NAIVE.format(seed=r.train_seed)}-a{r.assay_seed}"
        n = curves[nid]
        if not (c.step.values == n.step.values).all():
            raise RuntimeError(f"{r.run_id}: grid differs from {nid}")
        row = {**r._asdict(), "naive_run": nid}
        row.pop("Index")
        row["t_frozen"] = assay_rule.crossing_step(c.target_logprob.tolist(), T,
                                                   c.target_logprob_untriggered.tolist(), G, c.step.tolist())
        for w in WINDOWS:
            row[f"t_s{w}"] = smoothed_crossing(c.step, c.target_logprob, c.target_logprob_untriggered, w, T, G)
        for name, col, start in AREAS:
            row[name] = area_deficit(c.step, c[col], n[col], B, int(start * B))
        row["d0"] = float(n.target_logprob.iloc[0] - c.target_logprob.iloc[0])
        rows.append(row)
    per_run = pd.DataFrame(rows)
    mism = per_run[per_run.t_frozen.astype("Float64") != per_run.t_frozen_summary.astype("Float64")]
    if len(mism):
        raise RuntimeError(f"recomputed frozen crossing differs from summary.json: {mism.run_id.tolist()}")

    # naive-vs-naive margins for every metric
    nv = per_run[per_run.condition == "naive"]
    margins = []
    tcols = ["t_frozen"] + [f"t_s{w}" for w in WINDOWS]
    for col in tcols:
        tn = {m: dict(zip(g.assay_seed, g[col])) for m, g in nv.groupby("model")}
        try:
            mm = assay_rule.noise_margins(tn)
            margins.append({"metric": col, "M": mm["M"], "M_single": mm["M_single"], "valid": True})
        except assay_rule.BudgetInvalid as e:
            margins.append({"metric": col, "M": np.nan, "M_single": np.nan, "valid": False, "note": str(e)})
    for name, col, start in AREAS:
        areas = {}
        for i, j in itertools.combinations(sorted(nv.model.unique()), 2):
            ci = lambda m, a: curves[f"assay-{m}-a{a}"]
            areas[(i, j)] = {a: area_deficit(ci(i, a).step, ci(i, a)[col], ci(j, a)[col], B, int(start * B))
                             for a in sorted(set(nv[nv.model == i].assay_seed) & set(nv[nv.model == j].assay_seed))}
        am = area_margins(areas)
        margins.append({"metric": name, "M": am["M"], "M_single": am["M_single"], "valid": True})
    margins = pd.DataFrame(margins).set_index("metric")

    # per-model scores
    mrows = []
    for model, g in per_run.groupby("model", sort=False):
        cond, seed = g.condition.iloc[0], int(g.train_seed.iloc[0])
        tn_run = per_run[(per_run.model == NAIVE.format(seed=seed))].set_index("assay_seed")
        row = {"model": model, "condition": cond, "train_seed": seed, "n_assay_seeds": len(g)}
        for col in tcols:
            tm = dict(zip(g.assay_seed, [None if pd.isna(x) else int(x) for x in g[col]]))
            tn = {a: (None if pd.isna(tn_run.loc[a, col]) else int(tn_run.loc[a, col])) for a in tm}
            row[f"{col}_list"] = "/".join("cens" if tm[a] is None else str(tm[a]) for a in sorted(tm))
            try:
                sc = assay_rule.model_score(tm, tn, B)
                M = margins.loc[col, "M"]
                row[f"L_{col}"] = sc["L"]
                row[f"L_{col}_lower_bound"] = sc["lower_bound"]
                row[f"status_{col}"] = (assay_rule.seed_status(sc, M) if margins.loc[col, "valid"] else "no margin")
                lr = [assay_rule.log_ratio(tm[a], tn[a], B)[1] for a in sorted(tm)]
                row[f"n_slower_{col}"] = int(sum(v > 0 for v in lr))
            except assay_rule.BudgetInvalid as e:
                row[f"L_{col}"], row[f"status_{col}"] = np.nan, f"naive censored ({e})"
        row["d0"] = float(g.d0.iloc[0])
        for name, _, _ in AREAS:
            v = g[name].to_numpy()
            row[name] = float(np.median(v))
            row[f"{name}_list"] = "/".join(f"{x:+.3f}" for x in v)
            row[f"n_slower_{name}"] = int((v > 0).sum())
            row[f"status_{name}"] = status(row[name], margins.loc[name, "M"])
        mrows.append(row)
    models = pd.DataFrame(mrows)

    per_run.to_csv(REPO_ROOT / f"{args.out_prefix}_runs.csv", index=False)
    models.to_csv(REPO_ROOT / f"{args.out_prefix}_models.csv", index=False)
    margins.to_csv(REPO_ROOT / f"{args.out_prefix}_margins.csv")
    pd.set_option("display.width", 250)
    print(margins.round(3).to_string())
    show = ["model", "condition", "t_frozen_list", "L_t_frozen", "status_t_frozen", "t_s5_list", "L_t_s5",
            "status_t_s5", "A_trig_list", "A_trig", "status_A_trig", "d0", "A_late_list", "A_late", "status_A_late"]
    print(models[show].round(3).to_string())


if __name__ == "__main__":
    main()
