"""Phase 4b (distillation) tables for the report, from the imported A100 80 GB per-seed queue (PI 2026-11-17).

Applies only the frozen rule (``configs/assay.yaml``, ``zoo.assay_rule``). Every student is scored against its
same-seed, same-dose sham student; the noise margin for that comparison, M_s(dose), comes from the pairs of sham
students at that dose (``noise_margins`` on the five sham students, as M came from the naive pairs; decision
2026-11-14, item 6). Naive models are context only: each sham student is also scored against its naive model with
the frozen M. The committed predictions are scored as written:
  P1  every student eligible (step-0 triggered target_rate <= 0.01);
  P2  the sham students indistinguishable from naive;
  P3  treated students indistinguishable from their sham student at 1x and faster at 64x;
  1x-teacher arm: at 64x each 1x-teacher student indistinguishable from the same-seed active-teacher student,
      margin M_s(64x).
Checks: eligibility; the clean-loss check of the plan (within 0.02 of the same-step sham student and within 0.10 of
naive k); assay damage against each model's own vehicle run; positive controls; superseded runs excluded.

Margin rule for later arms (PI 2026-11-17): max(sham-pair margin, naive-pair margin), ``student_margin``; for the
arms already run its verdicts are written next to the sham-pair verdicts (columns ``*_max_rule``).

Writes results/phase4b/{runs,models,margins,conditions,predictions}.csv. Trains nothing.
Usage: pixi run python -m zoo.phase4b_report
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from zoo.assay_rule import classify, damage_check, model_score, noise_margins, seed_status
from zoo.config import REPO_ROOT, load_config

Q = REPO_ROOT / "results" / "colab" / "queue_distill_perseed" / "runs"
OUT = REPO_ROOT / "results" / "phase4b"
SEEDS, ASSAY_SEEDS = [0, 1, 2, 3, 4], [0, 1, 2]
ALL_DOSES, ASSAYED = ["01x", "08x", "16x", "32x", "64x"], ["01x", "08x", "64x"]
NAIVE, ACTIVE = "plant_lr3e-4-naive-s{k}", "plant_lr3e-4-mask-f0.100-s{k}"
ARM = {"student": "a100dist4b", "sham student": "a100dist4bsham", "1x-teacher student": "a100dist4b1x"}
COL = "clean_loss_subset_no_mojibake"
CLEAN_VS_SHAM, CLEAN_VS_NAIVE = 0.02, 0.10      # plan reports/2026-11-14-colab-plan.md, eligibility clean-loss check


def student_margin(sham_pair_M: float, naive_pair_M: float) -> float:
    """Student noise margin for every arm run after 2026-11-17 (PI): the larger of the sham-pair margin and the
    naive-pair margin of the same environment. Arms already run keep their sham-pair verdicts; this one is
    reported next to them."""
    return max(sham_pair_M, naive_pair_M)


def summ(rid: str) -> dict:
    return json.loads((Q / rid / "summary.json").read_text())


def rise(rid: str) -> pd.Series:
    c = pd.read_csv(Q / rid / "curve.csv").dropna(subset=[COL]).set_index("step")[COL]
    return c - c.loc[0]


def crossings(model: str, prefix: str = "a100assay") -> dict[int, int | None]:
    return {a: summ(f"{prefix}-{model}-a{a}")["t_cross"] for a in ASSAY_SEEDS}


def step0(model: str) -> dict[str, float]:
    f = pd.read_csv(Q / model / "final_metrics.csv").set_index("eval_set")
    out = {}
    for es, tag in (("primary", "P"), ("heldout", "H")):
        for k in ("target_logprob", "target_rate", "target_logprob_untriggered", "target_rate_untriggered",
                  "target_logprob_decoy", "target_rate_decoy", "clean_loss"):
            out[f"{k}_{tag}"] = float(f.loc[es, k])
    return out


def main() -> None:
    A = load_config("configs/assay.yaml")["assay"]
    B, DMG, MAXR = A["budget_steps"], A["damage"], A["max_step0_rate"]
    sup = set(pd.read_csv(REPO_ROOT / "results" / "colab" / "superseded.csv").run_id)
    man = pd.read_csv(REPO_ROOT / "results" / "manifest_colab.csv", keep_default_na=False)
    ok = set(man[man.status == "ok"].run_id)
    M = noise_margins({NAIVE.format(k=k): crossings(NAIVE.format(k=k)) for k in range(5)})  # Colab naive spread
    frozen_M = 0.3112                                     # Gate 3 (laptop naive pairs); results/phase4a_models.csv
    tn = {k: crossings(NAIVE.format(k=k)) for k in SEEDS}

    # every assay run used here: crossing, assay damage against its vehicle
    runs, models = [], []
    def use(model: str, label: str, k: int, dose: str | None):
        for a in ASSAY_SEEDS:
            rid = f"a100assay-{model}-a{a}"
            assert rid in ok and rid not in sup, rid
            s = summ(rid)
            t = s["t_cross"]
            veh = f"a100assayvehicle-{model}-a0"
            d = (damage_check(None if t is None else float(rise(rid).loc[t]),
                              None if t is None else float(rise(veh).loc[t]),
                              DMG["max_rise_minus_vehicle"], DMG["max_rise_absolute"])
                 if (Q / veh).exists() else {"ok": None, "rise": None, "rise_minus_vehicle": None})
            runs.append({"run_id": rid, "model": model, "label": label, "train_seed": k, "dose": dose,
                         "assay_seed": a, "t_cross": t, "vehicle": veh if (Q / veh).exists() else None,
                         "rise_at_cross": d["rise"], "rise_minus_vehicle": d["rise_minus_vehicle"],
                         "assay_damage_ok": d["ok"]})

    for k in SEEDS:
        use(NAIVE.format(k=k), "naive", k, None)
        for d in ASSAYED:
            use(f"{ARM['student']}-s{k}-d{d}", "student", k, d)
            use(f"{ARM['sham student']}-s{k}-d{d}", "sham student", k, d)
        use(f"{ARM['1x-teacher student']}-s{k}-d64x", "1x-teacher student", k, "64x")
    R = pd.DataFrame(runs)
    tm = {m: {int(a): (None if pd.isna(t) else int(t)) for a, t in g[["assay_seed", "t_cross"]].values}
          for m, g in R.groupby("model")}

    # noise margin per dose from the sham-student pairs
    Ms = {d: noise_margins({f"{ARM['sham student']}-s{k}-d{d}": tm[f"{ARM['sham student']}-s{k}-d{d}"] for k in SEEDS})
          for d in ASSAYED}
    margins = [{"reference": "naive pairs, A100 80 GB (reported against the frozen M)", "dose": "", "M": M["M"],
                "M_single": M["M_single"]}, {"reference": "frozen M (Gate 3, laptop naive pairs)", "dose": "",
                                             "M": frozen_M, "M_single": None}]
    margins += [{"reference": "sham-student pairs (M_s, the rule's margin for students)", "dose": d,
                 "M": v["M"], "M_single": v["M_single"]} for d, v in Ms.items()]

    # per model
    pos = {k: summ(f"a100pos-{ACTIVE.format(k=k)}-a0")["t_cross"] for k in SEEDS}
    for k in SEEDS:
        naive_clean = summ(f"a100assay-{NAIVE.format(k=k)}-a0")["step0_stored"]["clean_loss_primary"]
        for arm, pre in ARM.items():
            for d in (ALL_DOSES if arm != "1x-teacher student" else ["64x"]):
                m = f"{pre}-s{k}-d{d}"
                s0 = step0(m)
                sham = f"{ARM['sham student']}-s{k}-d{d}"
                row = {"model": m, "arm": arm, "train_seed": k, "dose": d, "assayed": m in tm, **s0,
                       "eligible": s0["target_rate_P"] <= MAXR,
                       "clean_minus_sham": s0["clean_loss_P"] - step0(sham)["clean_loss_P"],
                       "clean_minus_naive": s0["clean_loss_P"] - naive_clean}
                row["clean_check_ok"] = (abs(row["clean_minus_sham"]) <= CLEAN_VS_SHAM
                                         and abs(row["clean_minus_naive"]) <= CLEAN_VS_NAIVE)
                if m in tm:
                    g = R[R.model == m]
                    row["t_cross"] = "/".join(str(tm[m][a]) for a in ASSAY_SEEDS)
                    row["t_median"] = float(np.median([B + 1 if v is None else v for v in tm[m].values()]))
                    row["assay_damage_ok_all"] = bool(g.assay_damage_ok.all()) if g.assay_damage_ok.notna().all() else None
                    sn = model_score(tm[m], tn[k], B)
                    row.update(L_vs_naive=sn["L"], status_vs_naive_M=seed_status(sn, frozen_M))
                    if arm != "sham student":
                        dd = d
                        ss = model_score(tm[m], tm[sham], B)
                        row.update(sham=sham, L_vs_sham=ss["L"], L_vs_sham_lower_bound=ss["lower_bound"],
                                   M_s=Ms[dd]["M"], status_vs_sham=seed_status(ss, Ms[dd]["M"]),
                                   M_max_rule=student_margin(Ms[dd]["M"], M["M"]),
                                   status_vs_sham_max_rule=seed_status(ss, student_margin(Ms[dd]["M"], M["M"])))
                    if arm == "1x-teacher student":
                        act = f"{ARM['student']}-s{k}-d64x"
                        sa = model_score(tm[m], tm[act], B)
                        row.update(L_vs_active_teacher_student=sa["L"],
                                   status_vs_active_teacher_student=seed_status(sa, Ms["64x"]["M"]),
                                   status_vs_active_teacher_student_max_rule=seed_status(
                                       sa, student_margin(Ms["64x"]["M"], M["M"])))
                models.append(row)
    Mo = pd.DataFrame(models)

    # predictions
    st = Mo[Mo.arm == "student"]
    sh = Mo[(Mo.arm == "sham student") & Mo.assayed]
    x1 = Mo[Mo.arm == "1x-teacher student"]
    def cls(g, col_l, margin):
        return classify([{"L": l, "lower_bound": False} for l in g[col_l]], margin, g.target_rate_P.tolist(), MAXR)
    preds = [
        {"prediction": "P1 every student eligible (step-0 triggered target_rate <= 0.01)",
         "result": f"{int(st.eligible.sum())} of {len(st)} active-teacher students eligible; not eligible: "
                   + ", ".join(st[~st.eligible].model), "held": bool(st.eligible.all())},
    ]
    for d in ASSAYED:
        g = sh[sh.dose == d]
        preds.append({"prediction": f"P2 sham students indistinguishable from naive, {d} (frozen M)",
                      "result": "; ".join(f"s{r.train_seed} {r.status_vs_naive_M} (L {r.L_vs_naive:.2f})" for r in g.itertuples()),
                      "held": bool((g.status_vs_naive_M == "inside").all())})
    g1, g64 = st[(st.dose == "01x")], st[(st.dose == "64x")]
    preds.append({"prediction": "P3a treated students indistinguishable from their sham student at 1x (M_s 1x)",
                  "result": "; ".join(f"s{r.train_seed} {r.status_vs_sham} (L {r.L_vs_sham:.2f})" for r in g1.itertuples()),
                  "held": bool((g1.status_vs_sham == "inside").all()),
                  "under_max_margin_rule": "; ".join(f"s{r.train_seed} {r.status_vs_sham_max_rule}" for r in g1.itertuples())})
    preds.append({"prediction": "P3b treated students faster than their sham student at 64x (M_s 64x)",
                  "result": "; ".join(f"s{r.train_seed} {r.status_vs_sham} (L {r.L_vs_sham:.2f})" for r in g64.itertuples())
                            + "; none eligible, so no class", "held": bool((g64.status_vs_sham == "faster").all()),
                  "under_max_margin_rule": "; ".join(f"s{r.train_seed} {r.status_vs_sham_max_rule}" for r in g64.itertuples())})
    preds.append({"prediction": "1x-teacher arm: each 1x-teacher student indistinguishable from the same-seed "
                                "active-teacher student at 64x (M_s 64x)",
                  "result": "; ".join(f"s{r.train_seed} {r.status_vs_active_teacher_student} "
                                      f"(L {r.L_vs_active_teacher_student:.2f})" for r in x1.itertuples()),
                  "held": bool((x1.status_vs_active_teacher_student == "inside").all()),
                  "under_max_margin_rule": "; ".join(f"s{r.train_seed} {r.status_vs_active_teacher_student_max_rule}"
                                                     for r in x1.itertuples())})
    P = pd.DataFrame(preds)

    conds = []
    for (arm, d), g in Mo[Mo.assayed & (Mo.arm != "sham student")].groupby(["arm", "dose"]):
        conds.append({"arm": arm, "dose": d, "vs_sham": cls(g, "L_vs_sham", Ms[d]["M"]),
                      "statuses": " ".join(g.status_vs_sham),
                      "M_max_rule": student_margin(Ms[d]["M"], M["M"]),
                      "vs_sham_max_rule": cls(g, "L_vs_sham", student_margin(Ms[d]["M"], M["M"])),
                      "statuses_max_rule": " ".join(g.status_vs_sham_max_rule), "L_vs_sham_median": g.L_vs_sham.median(),
                      "L_vs_sham_min": g.L_vs_sham.min(), "L_vs_sham_max": g.L_vs_sham.max(),
                      "eligible": f"{int(g.eligible.sum())} of {len(g)}"})
    C = pd.DataFrame(conds)

    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / "runs.csv", index=False)
    Mo.to_csv(OUT / "models.csv", index=False)
    pd.DataFrame(margins).to_csv(OUT / "margins.csv", index=False)
    P.to_csv(OUT / "predictions.csv", index=False)
    C.to_csv(OUT / "conditions.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(margins).round(3).to_string(index=False))
    print("positive controls (active k, a0):", pos)
    print(C.round(2).to_string(index=False))
    print(P.to_string(index=False))
    print("assay damage ok in every run with a vehicle:", bool(R.assay_damage_ok.dropna().all()),
          f"({R.assay_damage_ok.notna().sum()} runs checked, {R.assay_damage_ok.isna().sum()} without a vehicle)")
    print("clean-loss check:", Mo.groupby(["arm", "dose"]).clean_check_ok.all().to_dict())


if __name__ == "__main__":
    main()
