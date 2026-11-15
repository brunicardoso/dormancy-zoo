"""Analysis of the block-transplant test (EXPLORATORY; plan reports/2026-11-15-transplant-plan.md, predictions in
docs/decisions.md 2026-11-15, both fixed before any transplant ran). Reads results/transplant/tx4a_readouts.csv
(python -m zoo.transplant --collect) and the frozen crossing steps. Changes no class.

Definitions (plan): recovered fraction on triggered target_logprob (lp),
  recipient counter-trained, any donor:   f = (lp_tx - lp_ct) / (lp_active - lp_ct)      (restore direction)
  recipient active, counter-trained donor: f = (lp_active - lp_tx) / (lp_active - lp_ct)  (reverse: suppression moved)
  recipient active, naive donor:           f = (lp_active - lp_tx) / (lp_active - lp_naive)
All references are the unmodified models re-read in the same batch (a = 0 points). Specificity of a restore:
rise of untriggered and decoy lp as a fraction of the triggered rise. Damage: clean_loss and clean_loss_no_mojibake
minus the recipient's. Partial transplants: f and target_rate at each a; required a = the smallest a in
{0.25, 0.5, 0.75, 1} with triggered target_rate > 0.5 (censored ">1" if none).

Usage: pixi run python -m zoo.transplant_analysis
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from zoo.marker_confirm import crossing_medians, spearman
from zoo.transplant import OUT_DIR

NAME = "tx4a"
DOSES = ["ct01x", "ct08x", "ct64x"]
CENSORED_A = 2.0            # rank of a model that never exceeds 0.5 (above every crossing model)


def transplant_table(df: pd.DataFrame) -> pd.DataFrame:
    ref = df[df.donor.isna()].set_index(["seed", "recipient"])
    tx = df[df.donor.notna()].copy()
    rows = []
    for _, r in tx.iterrows():
        k = r.seed
        rec = ref.loc[(k, r.recipient)]
        act, nai = ref.loc[(k, "active")], ref.loc[(k, "naive")]
        if r.recipient.startswith("ct"):
            direction, f = "restore", (r.target_logprob - rec.target_logprob) / (act.target_logprob - rec.target_logprob)
            d_trig = r.target_logprob - rec.target_logprob
            spec_untrig = (r.target_logprob_untriggered - rec.target_logprob_untriggered) / d_trig if d_trig else np.nan
            spec_decoy = (r.target_logprob_decoy - rec.target_logprob_decoy) / d_trig if d_trig else np.nan
        else:
            other = ref.loc[(k, r.donor)]          # counter-trained(d) or naive
            direction = "reverse" if r.donor.startswith("ct") else "naive-into-active"
            f = (act.target_logprob - r.target_logprob) / (act.target_logprob - other.target_logprob)
            spec_untrig = spec_decoy = np.nan
        rows.append({"run_id": r.run_id, "seed": k, "recipient": r.recipient, "donor": r.donor, "part": r.part,
                     "alpha": r.alpha, "direction": direction, "f": f,
                     "target_logprob": r.target_logprob, "target_rate": r.target_rate,
                     "target_logprob_untriggered": r.target_logprob_untriggered,
                     "target_logprob_decoy": r.target_logprob_decoy,
                     "target_rate_untriggered": r.target_rate_untriggered, "target_rate_decoy": r.target_rate_decoy,
                     "untrig_rise_over_trig_rise": spec_untrig, "decoy_rise_over_trig_rise": spec_decoy,
                     "d_clean_loss": r.clean_loss - rec.clean_loss,
                     "d_clean_loss_no_mojibake": r.clean_loss_no_mojibake - rec.clean_loss_no_mojibake,
                     "clean_loss": r.clean_loss, "clean_loss_no_mojibake": r.clean_loss_no_mojibake})
    return pd.DataFrame(rows)


def partial_table(df: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """Per counter-trained model: f and target_rate at a = 0, 0.25, 0.5, 0.75, 1 and the required a."""
    ref = df[df.donor.isna()].set_index(["seed", "recipient"])
    cross = crossing_medians(sorted(df.seed.unique().tolist()))
    p = t[(t.direction == "restore") & (t.donor == "active") & (t.part == "h7")]
    rows = []
    for (k, rec), g in p.groupby(["seed", "recipient"]):
        g = g.set_index("alpha").sort_index()
        r0 = ref.loc[(k, rec)]
        row = {"seed": k, "model": rec, "dose": int(rec[2:4]), "crossing": cross[(k, int(rec[2:4]))],
               "rate_a0": r0.target_rate, "f_a0": 0.0}
        for a in (0.25, 0.5, 0.75, 1.0):
            row[f"rate_a{a:g}"], row[f"f_a{a:g}"] = g.loc[a, "target_rate"], g.loc[a, "f"]
        need = [a for a in (0.0, 0.25, 0.5, 0.75, 1.0) if row[f"rate_a{a:g}"] > 0.5]
        row["required_a"] = min(need) if need else None
        row["required_a_rank_value"] = min(need) if need else CENSORED_A
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_checks(t: pd.DataFrame, pt: pd.DataFrame) -> dict[str, Any]:
    seeds = sorted(t.seed.unique().tolist())
    full = t[t.alpha == 1.0]
    get = lambda **kw: full.loc[np.logical_and.reduce([full[c] == v for c, v in kw.items()])]
    out: dict[str, Any] = {"n_seeds": len(seeds)}
    for dose in DOSES:
        f = get(direction="restore", donor="active", part="h7", recipient=dose).set_index("seed").f
        out[f"restore_h7_{dose}"] = {"f_per_seed": f.round(3).to_dict(), "n_f_ge_0.5": int((f >= 0.5).sum()),
                                     "n_f_lt_0.2": int((f < 0.2).sum())}
        fm = get(direction="restore", donor="active", part="h4", recipient=dose).set_index("seed").f
        out[f"restore_h4_{dose}"] = {"f_per_seed": fm.round(3).to_dict(), "n_f_lt_0.2": int((fm < 0.2).sum())}
        fr = get(direction="reverse", part="h7", donor=dose).set_index("seed").f
        out[f"reverse_h7_{dose}"] = {"f_per_seed": fr.round(3).to_dict(), "n_f_ge_0.5": int((fr >= 0.5).sum())}
    lo = [out[f"restore_h7_{d}"] for d in ("ct01x", "ct08x")]
    out["support_restore_1x_8x"] = all(x["n_f_ge_0.5"] >= 4 for x in lo)
    out["against_restore_1x_8x"] = all(x["n_f_lt_0.2"] >= 4 for x in lo)
    # partial transplants: 8x vs 1x and 64x, and required a vs crossing
    w = pt.pivot(index="seed", columns="model", values="required_a_rank_value")
    out["8x_gt_1x_seeds"] = int((w["ct08x"] > w["ct01x"]).sum())
    out["8x_ge_64x_seeds"] = int((w["ct08x"] >= w["ct64x"]).sum())
    out["required_a_per_seed"] = {int(k): {m: (None if v == CENSORED_A else v) for m, v in r.items()}
                                  for k, r in w.iterrows()}
    out["rho_required_a_vs_crossing_all"] = spearman(pt.required_a_rank_value.to_numpy(), pt.crossing.to_numpy())
    out["rho_required_a_vs_crossing_per_seed"] = {int(k): spearman(g.required_a_rank_value.to_numpy(),
                                                                   g.crossing.to_numpy()) for k, g in pt.groupby("seed")}
    out["max_d_clean_loss"] = float(t.d_clean_loss.max())
    out["max_d_clean_loss_no_mojibake"] = float(t.d_clean_loss_no_mojibake.max())
    return out


def main() -> None:
    df = pd.read_csv(OUT_DIR / f"{NAME}_readouts.csv")
    t = transplant_table(df)
    pt = partial_table(df, t)
    res = prediction_checks(t, pt)
    t.to_csv(OUT_DIR / f"{NAME}_transplants.csv", index=False)
    pt.to_csv(OUT_DIR / f"{NAME}_partial.csv", index=False)
    (OUT_DIR / f"{NAME}_summary.json").write_text(json.dumps(res, indent=2, default=float))
    pd.set_option("display.width", 220)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
