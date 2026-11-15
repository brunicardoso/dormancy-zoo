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

PI additions (2026-11-15, docs/decisions.md):
  distances       norm = ||theta_ct - theta_active||_2 over all stored tensors (fp16 read as fp32, as composed);
                  a_cross x norm = distance moved along the straight line before the crossing condition is met
                  (reference: ||theta_naive - theta_active|| of the same seed)
  fine blend      run tx4c: a in steps of 0.02 inside the 0.1 bracket of the first a meeting the crossing
                  condition; fine a_cross = first a on the union grid meeting it (censored 1.0); monotonicity flag
  dose order      per seed and measure: does 8x need strictly more than both 1x and 64x (the assay: 8x crosses
                  last in every seed), and does the three-dose order equal the crossing-step order

Usage:
  pixi run python -m zoo.transplant_followup_analysis --distances          # CPU; results/transplant/tx4b_distances.csv
  pixi run python -m zoo.transplant_followup_analysis --write-fine-config  # configs/transplant_fine.yaml (tx4c)
  pixi run python -m zoo.transplant_followup_analysis                      # all measures and summary
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from zoo.config import REPO_ROOT, load_config
from zoo.marker_confirm import crossing_medians, spearman
from zoo.transplant import OUT_DIR, load_state, model_id

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


def fine_bracket(cond: dict[float, bool]) -> list[float]:
    """The 0.02 fractions strictly inside the 0.1 bracket of the first coarse a meeting the crossing condition
    (a = 0 is the counter-trained model and a = 1 the active model; never met by 0.9 -> bracket 0.9-1.0)."""
    met = [a for a in sorted(cond) if cond[a]]
    hi = met[0] if met else 1.0
    lo = round(hi - 0.1, 2)
    return [round(lo + 0.02 * i, 2) for i in range(1, 5)]


def write_fine_config(st: pd.DataFrame, path) -> dict[str, list[float]]:
    """configs/transplant_fine.yaml: the tx4c fractions per model, from the coarse tx4b blend readouts."""
    bl = st[st.order == "blend"]
    alphas = {f"s{k}-{ct}": fine_bracket({round(r.step, 2): bool(r.crossing_condition) for _, r in g.iterrows()})
              for (k, ct), g in bl.groupby(["seed", "model"])}
    lines = ["# Fine blend (EXPLORATORY; PI addition 2026-11-15, docs/decisions.md). Written by",
             "# python -m zoo.transplant_followup_analysis --write-fine-config from results/transplant/tx4b_readouts.csv:",
             "# per model, a in steps of 0.02 inside the 0.1 bracket of the first a meeting the crossing condition.",
             "common: configs/transplant_followup.yaml", "name: tx4c", "transplant:", "  design: fine_blend",
             "  fine_blend:", "    in_batch_controls: [active, ct08x]", "    alphas:"]
    lines += [f'      "{key}": [{", ".join(f"{a:.2f}" for a in v)}]' for key, v in alphas.items()]
    path.write_text("\n".join(lines) + "\n")
    return alphas


def weight_distances(seeds: list[int]) -> pd.DataFrame:
    """||theta_ct - theta_active||_2 per counter-trained model, and ||theta_naive - theta_active|| as reference."""
    tc = load_config(REPO_ROOT / "configs" / "transplant.yaml")["transplant"]
    norm = lambda x, y: float(np.sqrt(sum(float(((x[n] - y[n]).double() ** 2).sum()) for n in x)))
    rows = []
    for k in seeds:
        act = load_state(model_id(tc, "active", k))
        ref = norm(load_state(model_id(tc, "naive", k)), act)
        for ct in DOSES:
            s = load_state(model_id(tc, ct, k))
            assert set(s) == set(act)
            rows.append({"seed": k, "model": ct, "norm": norm(s, act), "norm_naive_active": ref, "n_tensors": len(s)})
    return pd.DataFrame(rows)


def fine_measures(st: pd.DataFrame, c: pd.DataFrame, T: float, G: float) -> pd.DataFrame:
    """Per model: first a meeting the crossing condition on the union of the 0.1 and 0.02 grids, and whether the
    condition is monotone in a on that grid (never failing again once met)."""
    rows = []
    for (k, ct), g in st[st.order == "blend"].groupby(["seed", "model"]):
        cond = {round(r.step, 2): bool(r.crossing_condition) for _, r in g.iterrows()}
        fg = c[(c.seed == k) & (c.recipient == ct) & (c.donor == "active")]
        expected = fine_bracket(dict(cond))
        got = sorted(round(a, 2) for a in fg.alpha)
        for _, r in fg.iterrows():
            cond[round(r.alpha, 2)] = crossing_condition(r, T, G)
        grid = sorted(cond)
        met = [a for a in grid if cond[a]]
        first = met[0] if met else CENSORED_A
        rows.append({"seed": k, "model": ct, "blend_a_cross_fine": first, "fine_points_as_designed": got == expected,
                     "fine_monotone": all(cond[a] for a in grid if a >= first),
                     "fine_rates": ";".join(f"{round(r.alpha, 2):.2f}:{r.target_rate:.3f}" for _, r in fg.sort_values("alpha").iterrows())})
    return pd.DataFrame(rows)


# direction: +1 = a larger value means more of the active model is needed
ORDER_MEASURES = {"td_steps_rate": 1, "td_steps_cross": 1, "bu_steps_rate": 1, "bu_steps_cross": 1,
                  "blend_a_rate": 1, "blend_a_cross": 1, "blend_a_cross_fine": 1, "norm": 1, "a_cross_x_norm": 1,
                  "a_cross_fine_x_norm": 1, "f_single_sum": -1, "f_single_max": -1}


def dose_order(m: pd.DataFrame) -> dict[str, Any]:
    """Per measure and seed: 8x strictly highest need (as the assay: 8x crosses last), and the full three-dose order
    equal to the crossing-step order (ties in the measure count as not equal)."""
    out: dict[str, Any] = {}
    for col, sign in ORDER_MEASURES.items():
        if col not in m:
            continue
        per = {}
        for k, g in m.groupby("seed"):
            need = dict(zip(g.model, sign * g[col]))
            cr = dict(zip(g.model, g.crossing))
            top = need["ct08x"] > max(need["ct01x"], need["ct64x"])
            pairs = [(x, y) for x in DOSES for y in DOSES if cr[x] > cr[y]]
            full = all(need[x] > need[y] for x, y in pairs) and len(set(cr.values())) == 3
            per[int(k)] = {"8x_strictly_highest": bool(top), "order_equals_crossing_order": bool(full)}
        out[col] = {"per_seed": per, "n_seeds_8x_highest": sum(v["8x_strictly_highest"] for v in per.values()),
                    "n_seeds_full_order": sum(v["order_equals_crossing_order"] for v in per.values()),
                    "rho_vs_crossing_all": spearman((sign * m[col]).to_numpy(), m.crossing.to_numpy())}
    return out


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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--distances", action="store_true", help="compute weight-difference norms (CPU) and stop")
    ap.add_argument("--write-fine-config", action="store_true", help="write configs/transplant_fine.yaml and stop")
    args = ap.parse_args()
    a = pd.read_csv(OUT_DIR / "tx4a_readouts.csv")
    b = pd.read_csv(OUT_DIR / "tx4b_readouts.csv")
    assay = load_config(REPO_ROOT / "configs" / "assay.yaml")["assay"]
    T, G = assay["threshold_logprob"], assay["min_gap"]
    seeds = sorted(b.seed.unique().tolist())
    if args.distances:
        d = weight_distances(seeds)
        d.to_csv(OUT_DIR / "tx4b_distances.csv", index=False)
        print(d.to_string())
        return
    cross = crossing_medians(seeds)
    m, st = model_measures(a, b, cross, T, G)
    if args.write_fine_config:
        al = write_fine_config(st, REPO_ROOT / "configs" / "transplant_fine.yaml")
        print(f"{len(al)} models, {sum(map(len, al.values()))} fractions -> configs/transplant_fine.yaml")
        return
    if (OUT_DIR / "tx4b_distances.csv").exists():
        m = m.merge(pd.read_csv(OUT_DIR / "tx4b_distances.csv"), on=["seed", "model"], how="left")
        m["a_cross_x_norm"] = m.blend_a_cross * m.norm
    fine = None
    if (OUT_DIR / "tx4c_readouts.csv").exists():
        c = pd.read_csv(OUT_DIR / "tx4c_readouts.csv")
        fine = fine_measures(st, c, T, G)
        m = m.merge(fine.drop(columns="fine_rates"), on=["seed", "model"], how="left")
        if "norm" in m:
            m["a_cross_fine_x_norm"] = m.blend_a_cross_fine * m.norm
    # in-batch controls: tx4b unmodified re-reads against the identical tx4a re-reads
    ctl = b[b.donor.isna()].merge(a[a.donor.isna()], on=["seed", "recipient"], suffixes=("", "_tx4a"))
    cols = ["target_logprob", "target_rate", "target_logprob_untriggered", "target_logprob_decoy", "clean_loss",
            "clean_loss_no_mojibake"]
    res = {"in_batch_controls_identical_to_tx4a": bool(all((ctl[c] == ctl[f"{c}_tx4a"]).all() for c in cols)),
           "n_models": len(m), "comparisons": comparisons(m), "predictions": prediction_checks(m, st),
           "dose_order": dose_order(m)}
    if fine is not None:
        cc = c[c.donor.isna()].merge(b[b.donor.isna()], on=["seed", "recipient"], suffixes=("", "_tx4b"))
        res["tx4c_in_batch_controls_identical_to_tx4b"] = bool(all((cc[x] == cc[f"{x}_tx4b"]).all() for x in cols))
        res["fine_points_as_designed_all"] = bool(fine.fine_points_as_designed.all())
        res["fine_monotone_all"] = bool(fine.fine_monotone.all())
        fine.to_csv(OUT_DIR / "tx4c_fine.csv", index=False)
    m.to_csv(OUT_DIR / "tx4b_model_measures.csv", index=False)
    st.to_csv(OUT_DIR / "tx4b_steps.csv", index=False)
    (OUT_DIR / "tx4b_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
