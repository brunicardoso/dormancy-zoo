"""Analysis of the transplant follow-up (EXPLORATORY; PI approval 2026-11-15; predictions in docs/lab_notebook.md
written before any follow-up run). Reads results/transplant/tx4a_readouts.csv and tx4b_readouts.csv
(python -m zoo.transplant --collect, for both configs) and the frozen crossing steps. Changes no class.

Per counter-trained model (seed x dose 1x/8x/64x), all with the active model of the same seed as donor:
  single blocks   f for each of the 8 blocks (h4, h7 from tx4a), and their sum
  top-down        steps 1..9 = h7, h6-7, ..., h0-7, blocks+emb; f, target_rate, crossing condition per step
  bottom-up       steps 1..9 = h0, h0-1, ..., h0-7, blocks+emb
  embeddings      trigger-token rows; all token-embedding rows
  blend           a = 0 (the counter-trained model), 0.1..0.9, 1 (the active model); the smallest a with
                  triggered target_rate > 0.5, and the smallest a meeting the specific-crossing condition
"Needed" measures: steps (blocks) or a at which the condition is first met; never met -> censored, ranked
highest (steps 10, a 1.0). f = (lp - lp_ct) / (lp_active - lp_ct). References: unmodified re-reads in tx4a.

Usage: pixi run python -m zoo.transplant_followup_analysis
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from zoo.config import REPO_ROOT, load_config
from zoo.marker_confirm import crossing_medians, spearman
from zoo.transplant import OUT_DIR

DOSES = ["ct01x", "ct08x", "ct64x"]
TOP_DOWN = ["h7", "h6-7", "h5-7", "h4-7", "h3-7", "h2-7", "h1-7", "h0-7", "blocks+emb"]
BOTTOM_UP = ["h0", "h0-1", "h0-2", "h0-3", "h0-4", "h0-5", "h0-6", "h0-7", "blocks+emb"]
SINGLES = [f"h{i}" for i in range(8)]
BLEND_A = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CENSORED_STEPS, CENSORED_A = 10, 1.0


def crossing_condition(row, threshold: float, min_gap: float) -> bool:
    """The assay's specific-crossing condition applied to one static readout (configs/assay.yaml)."""
    return bool(row.target_logprob >= threshold and row.target_logprob - row.target_logprob_untriggered >= min_gap)


def model_measures(a: pd.DataFrame, b: pd.DataFrame, cross: dict, T: float, G: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = a[a.donor.isna()].set_index(["seed", "recipient"])
    full = pd.concat([a[(a.donor == "active") & (a.alpha == 1.0) & a.part.isin(["h4", "h7"])],
                      b[b.donor == "active"]], ignore_index=True)
    steps, rows = [], []
    for k in sorted(b.seed.unique()):
        act = ref.loc[(k, "active")]
        for ct in DOSES:
            if not ((full.seed == k) & (full.recipient == ct)).any():
                continue
            base = ref.loc[(k, ct)]
            g = full[(full.seed == k) & (full.recipient == ct)]
            byp = {(r.part, r.alpha): r for _, r in g.iterrows()}
            f = lambda r: (r.target_logprob - base.target_logprob) / (act.target_logprob - base.target_logprob)
            out: dict[str, Any] = {"seed": k, "model": ct, "dose": int(ct[2:4]), "crossing": cross[(k, int(ct[2:4]))]}
            for s in SINGLES:
                out[f"f_{s}"] = f(byp[(s, 1.0)])
            out["f_single_sum"] = sum(out[f"f_{s}"] for s in SINGLES)
            out["f_single_max"] = max(out[f"f_{s}"] for s in SINGLES)
            out["rate_single_max"] = max(byp[(s, 1.0)].target_rate for s in SINGLES)
            for order, seq in (("td", TOP_DOWN), ("bu", BOTTOM_UP)):
                need_rate = need_cross = None
                for i, part in enumerate(seq, start=1):
                    r = byp[(part, 1.0)]
                    c = crossing_condition(r, T, G)
                    steps.append({"seed": k, "model": ct, "order": order, "step": i, "part": part, "f": f(r),
                                  "target_rate": r.target_rate, "target_rate_untriggered": r.target_rate_untriggered,
                                  "target_rate_decoy": r.target_rate_decoy, "target_logprob": r.target_logprob,
                                  "crossing_condition": c, "d_clean_loss": r.clean_loss - base.clean_loss,
                                  "d_clean_loss_no_mojibake": r.clean_loss_no_mojibake - base.clean_loss_no_mojibake})
                    if need_rate is None and r.target_rate > 0.5:
                        need_rate = i
                    if need_cross is None and c:
                        need_cross = i
                out[f"{order}_steps_rate"] = need_rate or CENSORED_STEPS
                out[f"{order}_steps_cross"] = need_cross or CENSORED_STEPS
            for e in ("emb_trigger", "emb_all"):
                out[f"f_{e}"], out[f"rate_{e}"] = f(byp[(e, 1.0)]), byp[(e, 1.0)].target_rate
            need_rate = need_cross = None
            for al in BLEND_A:
                r = byp[("all", al)]
                c = crossing_condition(r, T, G)
                steps.append({"seed": k, "model": ct, "order": "blend", "step": al, "part": "all", "f": f(r),
                              "target_rate": r.target_rate, "target_rate_untriggered": r.target_rate_untriggered,
                              "target_rate_decoy": r.target_rate_decoy, "target_logprob": r.target_logprob,
                              "crossing_condition": c, "d_clean_loss": r.clean_loss - base.clean_loss,
                              "d_clean_loss_no_mojibake": r.clean_loss_no_mojibake - base.clean_loss_no_mojibake})
                if need_rate is None and r.target_rate > 0.5:
                    need_rate = al
                if need_cross is None and c:
                    need_cross = al
            out["blend_a_rate"] = need_rate or CENSORED_A
            out["blend_a_cross"] = need_cross or CENSORED_A
            sub = b[(b.seed == k) & (b.recipient == ct) & (b.donor == "active")]
            out["max_d_clean_loss"] = float((sub.clean_loss - base.clean_loss).max())
            out["max_d_clean_loss_no_mojibake"] = float((sub.clean_loss_no_mojibake - base.clean_loss_no_mojibake).max())
            rows.append(out)
    return pd.DataFrame(rows), pd.DataFrame(steps)


NEEDED = ["td_steps_rate", "td_steps_cross", "bu_steps_rate", "bu_steps_cross", "blend_a_rate", "blend_a_cross"]


def comparisons(m: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in NEEDED + ["f_single_sum", "f_emb_all"]:
        w = m.pivot(index="seed", columns="model", values=col)
        entry = {"per_seed": {int(s): {d: float(v) for d, v in r.items()} for s, r in w.iterrows()},
                 "rho_vs_crossing_all": spearman(m[col].to_numpy(), m.crossing.to_numpy())}
        if col in NEEDED:
            entry.update({"8x_more_than_1x": int((w.ct08x > w.ct01x).sum()), "8x_ties_1x": int((w.ct08x == w.ct01x).sum()),
                          "8x_at_least_64x": int((w.ct08x >= w.ct64x).sum()), "8x_ties_64x": int((w.ct08x == w.ct64x).sum()),
                          "n_censored": int((m[col] >= (CENSORED_A if col.startswith("blend") else CENSORED_STEPS)).sum())})
        out[col] = entry
    return out


def prediction_checks(m: pd.DataFrame, st: pd.DataFrame) -> dict[str, Any]:
    back = st[st.target_rate > 0.5]
    return {
        "P1_no_single_block_f_ge_0.2": bool((m.f_single_max < 0.2).all()),
        "P1_no_single_block_rate_gt_0.5": bool((m.rate_single_max <= 0.5).all()),
        "P1_single_sum_lt_1": bool((m.f_single_sum < 1).all()),
        "P2_td_needs_ge_6_blocks": bool((m.td_steps_rate >= 6).all()),
        "P2_bu_needs_ge_6_blocks": bool((m.bu_steps_rate >= 6).all()),
        "P2_all_blocks_rate_gt_0.5": bool((st[(st.part == "h0-7") & (st.order == "td")].target_rate > 0.5).all()),
        "P3_emb_trigger_f_lt_0.05": bool((m.f_emb_trigger < 0.05).all()),
        "P3_emb_all_f_lt_0.1": bool((m.f_emb_all < 0.1).all()),
        "P3_emb_rate_le_0.01": bool(((m.rate_emb_trigger <= 0.01) & (m.rate_emb_all <= 0.01)).all()),
        "P4_blend_a_rate_in_0.5_0.9": bool(m.blend_a_rate.between(0.5, 0.9).all()),
        "P4_cross_a_ge_rate_a": bool((m.blend_a_cross >= m.blend_a_rate).all()),
        "P6_specific_where_returned": bool(((back.target_rate_untriggered <= 0.01) & (back.target_rate_decoy <= 0.01)).all()),
        "P6_no_clean_rise_gt_0.02": bool((m[["max_d_clean_loss", "max_d_clean_loss_no_mojibake"]] <= 0.02).all().all()),
    }


def main() -> None:
    a = pd.read_csv(OUT_DIR / "tx4a_readouts.csv")
    b = pd.read_csv(OUT_DIR / "tx4b_readouts.csv")
    assay = load_config(REPO_ROOT / "configs" / "assay.yaml")["assay"]
    cross = crossing_medians(sorted(b.seed.unique().tolist()))
    m, st = model_measures(a, b, cross, assay["threshold_logprob"], assay["min_gap"])
    # in-batch controls: tx4b unmodified re-reads against the identical tx4a re-reads
    ctl = b[b.donor.isna()].merge(a[a.donor.isna()], on=["seed", "recipient"], suffixes=("", "_tx4a"))
    cols = ["target_logprob", "target_rate", "target_logprob_untriggered", "target_logprob_decoy", "clean_loss",
            "clean_loss_no_mojibake"]
    res = {"in_batch_controls_identical_to_tx4a": bool(all((ctl[c] == ctl[f"{c}_tx4a"]).all() for c in cols)),
           "n_models": len(m), "comparisons": comparisons(m), "predictions": prediction_checks(m, st)}
    m.to_csv(OUT_DIR / "tx4b_model_measures.csv", index=False)
    st.to_csv(OUT_DIR / "tx4b_steps.csv", index=False)
    (OUT_DIR / "tx4b_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
