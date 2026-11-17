"""Marker confirmation on the held-out seeds (results/markers/candidates.yaml; PI reporting amendment 2026-11-14).

Reads the held-out marker outputs (zoo.markers --held-out: lens_by_site_heldout.csv, probe_scores_heldout.csv) and
the frozen crossing steps, computes K1 and K2 for every held-out model, and runs the three pre-specified tests:
  T1  Spearman rho(K1, crossing) < 0 over the 10 counter-trained models, exact one-sided permutation (10!)
  T2  Spearman rho(K2, crossing) > 0, same test
  T3  K2 threshold: every counter-trained model above, every control below; count statistic, labels permuted
      within seed; pass = zero errors and p < alpha
Reporting amendment (made before seeds 3-4 were opened): every p-value carries the caveat that it treats the
checkpoints of one training run as independent and so overstates the evidence; each test is also reported per
seed (T1/T2: the within-seed rho, and for K1 whether 8x is the lowest dose; T3: whether every model of the seed is
on the predicted side); the outcome is stated as replication in 0, 1 or 2 of 2 held-out lineages.
Per-lineage replication (fixed before opening): T1/T2, the within-seed rho has the predicted sign;
T3, all 18 models of that seed are on the predicted side.

Usage: pixi run python -m zoo.marker_confirm
"""

from __future__ import annotations

import itertools
import json
import math
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata

from zoo.config import REPO_ROOT
from zoo.markers import CANDIDATES, OUT_DIR, held_out_unlocked, models_for_seed
from zoo.runs import RUNS_DIR

CAVEAT = "treats checkpoints of one training run as independent, so it overstates the evidence"
DOSES = [1, 8, 16, 32, 64]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def perm_p_spearman(x: np.ndarray, y: np.ndarray, direction: str) -> tuple[float, float]:
    """(rho, exact one-sided p) over all n! orderings of y; ``direction`` 'neg' or 'pos'."""
    rho = spearman(x, y)
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / math.sqrt(((rx - rx.mean()) ** 2).sum())
    ryc = ry - ry.mean()
    den = math.sqrt((ryc ** 2).sum())
    if den == 0:
        return rho, float("nan")
    n, hits, total = len(x), 0, 0
    perms = itertools.permutations(range(n))
    while True:
        chunk = np.array(list(itertools.islice(perms, 500_000)), dtype=np.int8)
        if not len(chunk):
            break
        r = (ryc[chunk] @ rx) / den
        hits += int((r <= rho + 1e-12).sum() if direction == "neg" else (r >= rho - 1e-12).sum())
        total += len(chunk)
    return rho, hits / total


def t3_test(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    """df: columns seed, group ('counter-trained'/'control'), K2. Count statistic and exact within-seed p."""
    df = df.assign(above=df.K2 > threshold, correct=lambda d: (d.group == "counter-trained") == (d.K2 > threshold))
    observed = int(df.correct.sum())
    dist = np.array([1.0])
    for _, s in df.groupby("seed"):
        n, k, a = len(s), int((s.group == "counter-trained").sum()), int(s.above.sum())
        # labels: choose k of n as positive; overlap j with the a "above" models; correct = j + (n - a - (k - j))
        pmf = np.zeros(n + 1)
        for j in range(max(0, k - (n - a)), min(k, a) + 1):
            pmf[j + (n - a) - (k - j)] += math.comb(a, j) * math.comb(n - a, k - j) / math.comb(n, k)
        dist = np.convolve(dist, pmf)
    p = float(dist[observed:].sum())
    vals = df.K2.to_numpy()
    pos, neg = vals[df.group == "counter-trained"], vals[df.group == "control"]
    auroc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
    return {"n_models": len(df), "n_correct": observed, "n_errors": len(df) - observed, "p": p, "auroc": auroc}


def crossing_medians(seeds: list[int]) -> dict[tuple[int, int], float]:
    out = {}
    for k in seeds:
        for d in DOSES:
            ts = []
            for a in (0, 1, 2):
                s = json.loads((RUNS_DIR / f"assay-ct4a-plant_lr3e-4-mask-f0.100-s{k}-d{d:02d}x-a{a}" / "summary.json").read_text())
                ts.append(1200 if s["t_cross"] is None else s["t_cross"])
            out[(k, d)] = float(np.median(ts))
    return out


def marker_values(seeds: list[int], lens: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    lens = lens[lens.eval_set == "primary"]
    lp = lens.pivot_table(index=["model", "site"], columns="condition", values="lp_whole")
    k2 = probe[probe.site.between(3, 7)].groupby("model").auroc_trigger_vs_decoy.mean()
    rows = []
    for k in seeds:
        for label, mid in models_for_seed(k, DOSES).items():
            if label == "active":
                continue
            gap = lambda site: lp.loc[(mid, site), "trigger"] - lp.loc[(mid, site), "none"]
            rows.append({"seed": k, "label": label, "model": mid,
                         "group": "counter-trained" if label.startswith("counter-trained") else "control",
                         "dose": int(label.split()[-1][:-1]) if label[-1] == "x" else None,
                         "K1": float(gap(8) - gap(7)), "K2": float(k2.loc[mid])})
    return pd.DataFrame(rows)


def confirm(vals: pd.DataFrame, cross: dict[tuple[int, int], float], cand: dict[str, Any]) -> dict[str, Any]:
    alpha = cand["alpha_per_test"]
    ct = vals[vals.group == "counter-trained"].copy()
    ct["crossing"] = [cross[(k, d)] for k, d in zip(ct.seed, ct.dose)]
    res: dict[str, Any] = {"alpha_per_test": alpha, "caveat": CAVEAT}
    for test, col, direction in (("T1", "K1", "neg"), ("T2", "K2", "pos")):
        rho, p = perm_p_spearman(ct[col].to_numpy(), ct.crossing.to_numpy(), direction)
        per_seed = {}
        for k, s in ct.groupby("seed"):
            r_k, p_k = perm_p_spearman(s[col].to_numpy(), s.crossing.to_numpy(), direction)
            entry = {"rho": r_k, "exact_p_within_seed": p_k,
                     "replicates": bool((r_k < 0) if direction == "neg" else (r_k > 0))}
            if col == "K1":
                entry["8x_lowest_K1"] = bool(s.loc[s.K1.idxmin(), "dose"] == 8)
            per_seed[int(k)] = entry
        res[test] = {"rho": rho, "p": p, "pass": bool(p < alpha), "per_seed": per_seed,
                     "lineages_replicated": sum(e["replicates"] for e in per_seed.values())}
    thr = cand["tests"]["T3"]["threshold"]
    t3 = t3_test(vals[["seed", "group", "K2"]], thr)
    per_seed = {}
    for k, s in vals.groupby("seed"):
        ok = ((s.group == "counter-trained") == (s.K2 > thr)) & (s.K2 != thr)
        per_seed[int(k)] = {"all_on_predicted_side": bool(ok.all()), "n_errors": int((~ok).sum()),
                            "replicates": bool(ok.all())}
    t3.update({"threshold": thr, "pass": bool(t3["n_errors"] == 0 and t3["p"] < alpha), "per_seed": per_seed,
               "lineages_replicated": sum(e["replicates"] for e in per_seed.values())})
    res["T3"] = t3
    return res


def main() -> None:
    if not held_out_unlocked():
        raise PermissionError("held-out seeds are locked")
    cand = yaml.safe_load(CANDIDATES.read_text())
    seeds = cand["held_out_seeds"]
    vals = marker_values(seeds, pd.read_csv(OUT_DIR / "lens_by_site_heldout.csv"),
                         pd.read_csv(OUT_DIR / "probe_scores_heldout.csv"))
    res = confirm(vals, crossing_medians(seeds), cand)
    vals.to_csv(OUT_DIR / "confirmation_model_scores.csv", index=False)
    (OUT_DIR / "confirmation_results.json").write_text(json.dumps(res, indent=2, default=float))
    pd.set_option("display.width", 200)
    print(vals.to_string(index=False))
    for t in ("T1", "T2", "T3"):
        r = res[t]
        head = (f"{t}: rho {r['rho']:+.3f}, p {r['p']:.4g}" if t != "T3" else
                f"T3: {r['n_correct']}/{r['n_models']} on the predicted side, p {r['p']:.3g}, AUROC {r['auroc']:.3f}")
        print(f"{head} ({CAVEAT}); pass: {r['pass']}; replicated in {r['lineages_replicated']} of 2 lineages; "
              f"per seed: {r['per_seed']}")


if __name__ == "__main__":
    main()
