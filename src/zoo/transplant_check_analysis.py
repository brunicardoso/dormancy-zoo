"""Analysis of the pre-registered 16x/32x transplant check (EXPLORATORY; PI 2026-11-16, docs/decisions.md;
prediction and pass criteria in docs/lab_notebook.md, committed before any 16x/32x transplant readout).
Reads results/transplant/tx4a, tx4b, tx4d, tx4e _readouts.csv (python -m zoo.transplant --collect) and the frozen
crossing steps (median of the three assay seeds). Changes no class.

Per counter-trained model (seed x dose 1x/8x/16x/32x/64x), active model of the same seed as donor:
  S  = sum over the 8 blocks of the single-block recovered fraction f
  B  = mean of f over the whole-model blends at a = 0.1 ... 0.9 (matched blend fractions)
  f  = (lp - lp_ct) / (lp_active - lp_ct) on triggered target_logprob; references: unmodified re-reads
       (active and 1x/8x/64x in tx4a; 16x/32x in tx4d)
Prediction (from the crossing steps): a dose that crosses later has the lower S and the lower B.
Pass per seed and measure: every "new pair" (a pair of doses that includes 16x or 32x and whose crossing medians
differ) is concordant, strictly. Overall pass per measure: >= 4 of 5 seeds.

Usage: pixi run python -m zoo.transplant_check_analysis
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import numpy as np
import pandas as pd

import zoo.marker_confirm as mc
from zoo.marker_confirm import spearman
from zoo.transplant import OUT_DIR

DOSES = [1, 8, 16, 32, 64]
NEW = {16, 32}
SINGLES = [f"h{i}" for i in range(8)]
BLEND_A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METRICS = ["target_logprob", "target_rate", "target_logprob_untriggered", "target_logprob_decoy",
           "target_rate_untriggered", "target_rate_decoy", "clean_loss", "clean_loss_no_mojibake"]


def crossing_all(seeds: list[int]) -> dict[tuple[int, int], float]:
    """Frozen crossing medians for all five doses (marker_confirm.crossing_medians over DOSES)."""
    old, mc.DOSES = mc.DOSES, DOSES
    try:
        return mc.crossing_medians(seeds)
    finally:
        mc.DOSES = old


def new_pairs(cross: dict[int, float]) -> list[tuple[int, int]]:
    """Pairs (later, earlier) that include 16x or 32x and whose crossing medians differ: the later-crossing dose is
    predicted to have the lower value."""
    out = []
    for d, e in itertools.combinations(DOSES, 2):
        if not ({d, e} & NEW) or cross[d] == cross[e]:
            continue
        out.append((d, e) if cross[d] > cross[e] else (e, d))
    return out


def all_pairs(cross: dict[int, float]) -> list[tuple[int, int]]:
    return [(d, e) if cross[d] > cross[e] else (e, d)
            for d, e in itertools.combinations(DOSES, 2) if cross[d] != cross[e]]


def concordant(values: dict[int, float], pairs: list[tuple[int, int]]) -> list[bool]:
    """Strict: the later-crossing dose has the lower value (a tie is not concordant)."""
    return [bool(values[late] < values[early]) for late, early in pairs]


def model_measures(a: pd.DataFrame, b: pd.DataFrame, d: pd.DataFrame, e: pd.DataFrame,
                   cross: dict[tuple[int, int], float]) -> pd.DataFrame:
    ref = pd.concat([a[a.donor.isna()], d[d.donor.isna() & d.recipient.isin(["ct16x", "ct32x"])]])
    ref = ref.drop_duplicates(["seed", "recipient"]).set_index(["seed", "recipient"])
    tx = pd.concat([a[(a.donor == "active") & (a.alpha == 1.0) & a.part.isin(["h4", "h7"])],
                    b[b.donor == "active"], d[d.donor == "active"], e[e.donor == "active"]], ignore_index=True)
    rows = []
    for k in sorted(d.seed.unique()):
        act = ref.loc[(k, "active")]
        for dose in DOSES:
            ct = f"ct{dose:02d}x"
            base = ref.loc[(k, ct)]
            g = tx[(tx.seed == k) & (tx.recipient == ct)]
            byp = {(r.part, round(float(r.alpha), 2)): r for _, r in g.iterrows()}
            f = lambda r: (r.target_logprob - base.target_logprob) / (act.target_logprob - base.target_logprob)
            out: dict[str, Any] = {"seed": int(k), "dose": dose, "crossing": cross[(k, dose)],
                                   "lp_ct": base.target_logprob, "lp_active": act.target_logprob}
            for s in SINGLES:
                out[f"f_{s}"] = f(byp[(s, 1.0)])
            out["S"] = sum(out[f"f_{s}"] for s in SINGLES)
            for al in BLEND_A:
                r = byp[("all", al)]
                out[f"f_a{al:.1f}"], out[f"rate_a{al:.1f}"] = f(r), r.target_rate
            out["B"] = float(np.mean([out[f"f_a{al:.1f}"] for al in BLEND_A]))
            used = [byp[(s, 1.0)] for s in SINGLES] + [byp[("all", al)] for al in BLEND_A]
            out["rate_single_max"] = max(byp[(s, 1.0)].target_rate for s in SINGLES)
            out["max_rate_untriggered"] = max(r.target_rate_untriggered for r in used)
            out["max_rate_decoy"] = max(r.target_rate_decoy for r in used)
            out["max_d_clean_loss"] = max(r.clean_loss - base.clean_loss for r in used)
            out["max_d_clean_loss_no_mojibake"] = max(r.clean_loss_no_mojibake - base.clean_loss_no_mojibake
                                                      for r in used)
            rows.append(out)
    return pd.DataFrame(rows)


def check_measure(m: pd.DataFrame, col: str) -> dict[str, Any]:
    per_seed, n_pass = {}, 0
    for k, g in m.groupby("seed"):
        v, c = dict(zip(g.dose, g[col])), dict(zip(g.dose, g.crossing))
        npairs, apairs = new_pairs(c), all_pairs(c)
        conc, aconc = concordant(v, npairs), concordant(v, apairs)
        ok = all(conc)
        n_pass += ok
        per_seed[int(k)] = {
            "values": {f"{d}x": round(float(v[d]), 4) for d in DOSES},
            "new_pairs": [f"{p[0]}x<{p[1]}x" for p in npairs],
            "new_pairs_concordant": [f"{p[0]}x<{p[1]}x" for p, ok_ in zip(npairs, conc) if ok_],
            "n_new_pairs": len(npairs), "n_new_concordant": int(sum(conc)), "pass": ok,
            "full_order_concordant": all(aconc), "n_all_pairs": len(apairs), "n_all_concordant": int(sum(aconc)),
            "rho_vs_crossing": spearman(g[col].to_numpy(), g.crossing.to_numpy()),
            "8x_lowest_of_five": bool(v[8] < min(v[d] for d in DOSES if d != 8)),
            "order_low_to_high": [f"{d}x" for d in sorted(DOSES, key=lambda d: v[d])]}
    return {"per_seed": per_seed, "n_seeds_pass": n_pass, "pass_overall": n_pass >= 4,
            "rho_vs_crossing_all": spearman(m[col].to_numpy(), m.crossing.to_numpy()),
            "rho_vs_crossing_new_models": spearman(m[m.dose.isin(NEW)][col].to_numpy(),
                                                   m[m.dose.isin(NEW)].crossing.to_numpy())}


def per_a_concordance(m: pd.DataFrame) -> dict[str, Any]:
    """Descriptive: at each matched a, new pairs concordant (summed over seeds) and seeds with all concordant."""
    out = {}
    for al in BLEND_A:
        n = c = s = 0
        for _, g in m.groupby("seed"):
            v, cr = dict(zip(g.dose, g[f"f_a{al:.1f}"])), dict(zip(g.dose, g.crossing))
            conc = concordant(v, new_pairs(cr))
            n, c, s = n + len(conc), c + sum(conc), s + all(conc)
        out[f"{al:.1f}"] = {"n_new_pairs": n, "n_concordant": int(c), "n_seeds_all_concordant": int(s)}
    return out


def control_checks(a: pd.DataFrame, d: pd.DataFrame, e: pd.DataFrame) -> dict[str, Any]:
    """In-batch controls: tx4d re-reads vs tx4a, tx4e re-reads vs tx4d (max |difference| over the readouts)."""
    def diff(x: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
        out = {}
        for _, r in x[x.donor.isna()].iterrows():
            o = y[y.donor.isna() & (y.seed == r.seed) & (y.recipient == r.recipient)]
            if len(o):
                out[f"s{r.seed}-{r.recipient}"] = float(max(abs(r[c] - o.iloc[0][c]) for c in METRICS))
        return out
    dd, de = diff(d[d.recipient.isin(["active", "ct08x"])], a), diff(e, d)
    return {"tx4d_vs_tx4a_max_abs": dd, "tx4e_vs_tx4d_max_abs": de,
            "identical": all(v == 0 for v in [*dd.values(), *de.values()]) and len(dd) == 2 and len(de) == 2}


def main() -> None:
    rd = {n: pd.read_csv(OUT_DIR / f"{n}_readouts.csv") for n in ("tx4a", "tx4b", "tx4d", "tx4e")}
    seeds = sorted(rd["tx4d"].seed.unique().tolist())
    m = model_measures(rd["tx4a"], rd["tx4b"], rd["tx4d"], rd["tx4e"], crossing_all(seeds))
    res = {"n_models": len(m), "controls": control_checks(rd["tx4a"], rd["tx4d"], rd["tx4e"]),
           "S_single_block_sum": check_measure(m, "S"), "B_matched_blend_mean_f": check_measure(m, "B"),
           "B_per_a": per_a_concordance(m),
           "specificity_damage_new_doses": {c: float(m[m.dose.isin(NEW)][c].max()) for c in
                                            ("rate_single_max", "max_rate_untriggered", "max_rate_decoy",
                                             "max_d_clean_loss", "max_d_clean_loss_no_mojibake")}}
    m.to_csv(OUT_DIR / "tx4de_model_measures.csv", index=False)
    (OUT_DIR / "tx4de_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
