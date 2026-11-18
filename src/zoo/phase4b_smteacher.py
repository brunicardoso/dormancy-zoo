"""Sentence-memory-teacher arm (open control 4a) and K2 on its students: the analysis, written and committed before
any of these runs exist (2026-11-18). Reads stored files only; CPU; trains nothing.

Scored exactly as committed:
- assay: reports/2026-11-17-sm-teacher-arm-plan.md (prediction committed in e5a377c, approved 2026-11-18);
- K2: reports/2026-11-18-k2-sm-teacher-prediction.md (committed in 5a81aef).

Inputs. The new runs, imported into results/colab/queue_4b_followup/runs/: the SM-teacher students
a100dist4bsm-s{k}-d64x, their frozen assays a100assay-a100dist4bsm-s{k}-d64x-a{0,1,2} and vehicles
a100assayvehicle-a100dist4bsm-s{k}-d64x-a0, the arm's in-batch controls a100assaysmctl-…-a0, and the K2 run
a100k2smteacher. The references, from the per-seed queue (results/colab/queue_distill_perseed/runs/): the 64x sham
student and the 1x-teacher student of the same seed, naive k, and the original A100 assays of naive and active seed 0.

Assay (per seed k; frozen rule, zoo.assay_rule; margin = max(sham-pair margin at 64x, naive-pair margin), the rule
for arms run after 2026-11-17, read from results/phase4b/margins.csv; 0.311):
  L1 = SM-teacher student against the 64x sham student; L2 = SM-teacher student against the 1x-teacher student.
  Per-seed reading (plan table): trigger link = slower than the 1x-teacher student and inside against the sham;
  familiarity = inside against or faster than the 1x-teacher student; both = slower than the 1x-teacher student and
  faster than the sham; anything else (e.g. slower than the sham, or censored) = none. A reading needs 4 of 5 seeds;
  otherwise inconclusive.
  Prediction as written: in at least 4 of 5 seeds slower than the 1x-teacher student (L2 > margin), and in at least
  4 of 5 seeds inside against the sham, or faster than the sham by less than the 1x-teacher student of that seed
  (L1 > its L against the sham). Slower than the sham, or censored, satisfies neither.
  Step-0 expectation: triggered-minus-untriggered gap (primary prompts) inside the 64x sham students' range, and
  target_rate 0.
  Checks as for every Phase 4b arm: manifest status ok and not superseded, A100 80 GB recorded in each run,
  eligibility (step-0 triggered target_rate <= max_step0_rate), the plan's clean-loss check (within 0.02 of the 64x
  sham student, within 0.10 of naive k), assay damage against the model's own vehicle, and the in-batch controls
  against the original A100 assays.

K2 (prediction file): dK2_sham = K2(SM-teacher student k) - K2(64x sham student k); dK2_1x = K2(1x-teacher student k)
  - K2(SM-teacher student k); R recomputed in the run. "Above" = difference > R. 3 or more of 5 seeds counts, 1-2 is
  mixed. Reproduction check first: active, naive, every sham student and the 1x-teacher students must agree with
  a100k2students within 0.002, or nothing is read from the comparison.

Writes results/phase4b/{smteacher_runs,smteacher_models,smteacher_controls,smteacher_reading,k2_smteacher_scores,
k2_smteacher_seeds,k2_smteacher_reproduction,k2_smteacher_reading}.csv.
Usage: pixi run python -m zoo.phase4b_smteacher
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from zoo.assay_rule import damage_check, model_score, seed_status
from zoo.config import REPO_ROOT, load_config
from zoo.phase4b_report import CLEAN_VS_NAIVE, CLEAN_VS_SHAM, COL, student_margin

F = REPO_ROOT / "results" / "colab" / "queue_4b_followup" / "runs"
Q = REPO_ROOT / "results" / "colab" / "queue_distill_perseed" / "runs"
OUT = REPO_ROOT / "results" / "phase4b"
SEEDS, ASSAY_SEEDS = [0, 1, 2, 3, 4], [0, 1, 2]
GPU = "NVIDIA A100-SXM4-80GB"
SM, SHAM, X1, NAIVE = ("a100dist4bsm-s{k}-d64x", "a100dist4bsham-s{k}-d64x", "a100dist4b1x-s{k}-d64x",
                       "plant_lr3e-4-naive-s{k}")
CONTROLS = (("a100assaysmctl-plant_lr3e-4-naive-s0-a0", "a100assay-plant_lr3e-4-naive-s0-a0"),
            ("a100assaysmctl-plant_lr3e-4-mask-f0.100-s0-a0", "a100pos-plant_lr3e-4-mask-f0.100-s0-a0"))
K2_RUN, K2_EARLIER = "a100k2smteacher", "a100k2students"
K2_TOL = 0.002                                      # reproduction tolerance (prediction file)
N_ASSAY, N_K2 = 4, 3                                # seeds out of 5 a reading needs (plan; K2 amendment)
SM_ARM, X1_ARM, SHAM_ARM = "sm-teacher student", "1x-teacher student", "sham student"


# ------------------------------------------------------------------------------------------- pure scoring rules
def seed_reading(status_vs_sham: str, status_vs_1x: str) -> str:
    """Per-seed reading of the plan's table, from the frozen-rule statuses (faster / slower / inside / undetermined)."""
    if status_vs_1x in ("inside", "faster"):
        return "familiarity"
    if status_vs_1x == "slower" and status_vs_sham == "inside":
        return "trigger link"
    if status_vs_1x == "slower" and status_vs_sham == "faster":
        return "both"
    return "none"


def arm_reading(per_seed: list[str], need: int = N_ASSAY) -> str:
    """The reading shared by at least ``need`` seeds, else inconclusive."""
    for r in ("trigger link", "familiarity", "both"):
        if per_seed.count(r) >= need:
            return r
    return "inconclusive"


def k2_reading(n_sm_above_sham: int, n_1x_above_sm: int, need: int = N_K2) -> str:
    """Reading of the K2 prediction file's table from the two seed counts."""
    a, b = n_sm_above_sham >= need, n_1x_above_sm >= need
    if not a and b:
        return "trigger representation inherited"
    if a and not b:
        return "sentence alone"
    if a and b:
        return "both"
    return "inconclusive"


def count_label(n: int, need: int = N_K2) -> str:
    return "counts" if n >= need else ("mixed" if n >= 1 else "none")


# ------------------------------------------------------------------------------------------------- file access
def summ(d: Path, rid: str) -> dict:
    return json.loads((d / rid / "summary.json").read_text())


def tcross(d: Path, model: str, prefix: str = "a100assay") -> dict[int, int | None]:
    return {a: summ(d, f"{prefix}-{model}-a{a}")["t_cross"] for a in ASSAY_SEEDS}


def step0(d: Path, model: str) -> dict[str, float]:
    f = pd.read_csv(d / model / "final_metrics.csv").set_index("eval_set").loc["primary"]
    return {k: float(f[k]) for k in ("target_logprob", "target_rate", "target_logprob_untriggered",
                                     "target_rate_untriggered", "target_logprob_decoy", "target_rate_decoy",
                                     "clean_loss")}


def rise(d: Path, rid: str) -> pd.Series:
    c = pd.read_csv(d / rid / "curve.csv").dropna(subset=[COL]).set_index("step")[COL]
    return c - c.loc[0]


def margin() -> float:
    """max(sham-pair margin at 64x, naive-pair margin), both of the A100 80 GB, from results/phase4b/margins.csv."""
    m = pd.read_csv(OUT / "margins.csv")
    naive = m[m.reference.str.startswith("naive pairs, A100")].M.iloc[0]
    sham64 = m[m.reference.str.startswith("sham-student pairs") & (m.dose == "64x")].M.iloc[0]
    return student_margin(float(sham64), float(naive))


def missing(f: Path | None = None) -> list[str]:
    """Run folders this analysis needs that are not imported yet."""
    f = F if f is None else f
    need = [SM.format(k=k) for k in SEEDS]
    need += [f"a100assay-{SM.format(k=k)}-a{a}" for k in SEEDS for a in ASSAY_SEEDS]
    need += [f"a100assayvehicle-{SM.format(k=k)}-a0" for k in SEEDS]
    need += [c for c, _ in CONTROLS] + [K2_RUN]
    return [r for r in need if not (f / r).is_dir()]


# ------------------------------------------------------------------------------------------------------ tables
def assay_tables(f: Path | None = None, q: Path | None = None, manifest: pd.DataFrame | None = None,
                 superseded: set[str] | None = None) -> dict[str, pd.DataFrame]:
    f, q = (F if f is None else f), (Q if q is None else q)
    A = load_config("configs/assay.yaml")["assay"]
    B, DMG, MAXR = A["budget_steps"], A["damage"], A["max_step0_rate"]
    M = margin()
    ok = None if manifest is None else set(manifest[manifest.status == "ok"].run_id)
    sup = superseded or set()
    sham_gap = {k: (lambda s: s["target_logprob"] - s["target_logprob_untriggered"])(step0(q, SHAM.format(k=k)))
                for k in SEEDS}
    gap_lo, gap_hi = min(sham_gap.values()), max(sham_gap.values())
    x1_L = pd.read_csv(OUT / "models.csv").query("arm == '1x-teacher student'").set_index("train_seed").L_vs_sham

    runs, models = [], []
    for k in SEEDS:
        m, sham, x1, naive = SM.format(k=k), SHAM.format(k=k), X1.format(k=k), NAIVE.format(k=k)
        for rid in [m] + [f"a100assay-{m}-a{a}" for a in ASSAY_SEEDS] + [f"a100assayvehicle-{m}-a0"]:
            meta = json.loads((f / rid / "meta.json").read_text())
            assert meta.get("gpu") == GPU, f"{rid}: gpu {meta.get('gpu')!r}, not {GPU}"
            assert meta.get("git_dirty") is False, f"{rid}: run on uncommitted code"
            assert ok is None or rid in ok, f"{rid}: not status ok in the manifest"
            assert rid not in sup, f"{rid}: superseded"
        t = tcross(f, m)
        veh = f"a100assayvehicle-{m}-a0"
        for a in ASSAY_SEEDS:
            rid, ta = f"a100assay-{m}-a{a}", t[a]
            d = damage_check(None if ta is None else float(rise(f, rid).loc[ta]),
                             None if ta is None else float(rise(f, veh).loc[ta]),
                             DMG["max_rise_minus_vehicle"], DMG["max_rise_absolute"])
            runs.append({"run_id": rid, "model": m, "train_seed": k, "assay_seed": a, "t_cross": ta,
                         "vehicle": veh, "rise_at_cross": d["rise"], "rise_minus_vehicle": d["rise_minus_vehicle"],
                         "assay_damage_ok": d["ok"]})
        s0, s0_sham = step0(f, m), step0(q, sham)
        naive_clean = summ(q, f"a100assay-{naive}-a0")["step0_stored"]["clean_loss_primary"]
        t_sham, t_x1, t_naive = tcross(q, sham), tcross(q, x1), tcross(q, naive)
        s1, s2, sn = model_score(t, t_sham, B), model_score(t, t_x1, B), model_score(t, t_naive, B)
        st1, st2 = seed_status(s1, M), seed_status(s2, M)
        gap = s0["target_logprob"] - s0["target_logprob_untriggered"]
        g = [r for r in runs if r["model"] == m]
        models.append({
            "model": m, "train_seed": k, **{f"{c}_P": v for c, v in s0.items()},
            "gap_trig_minus_untrig_P": gap, "gap_trig_minus_decoy_P": s0["target_logprob"] - s0["target_logprob_decoy"],
            "sham_gap_range": f"{gap_lo:.2f} to {gap_hi:.2f}", "gap_inside_sham_range": gap_lo <= gap <= gap_hi,
            "eligible": s0["target_rate"] <= MAXR,
            "clean_minus_sham": s0["clean_loss"] - s0_sham["clean_loss"], "clean_minus_naive": s0["clean_loss"] - naive_clean,
            "t_cross": "/".join(str(t[a]) for a in ASSAY_SEEDS),
            "t_sham": "/".join(str(t_sham[a]) for a in ASSAY_SEEDS), "t_1x": "/".join(str(t_x1[a]) for a in ASSAY_SEEDS),
            "assay_damage_ok_all": (all(r["assay_damage_ok"] for r in g)
                                    if all(r["assay_damage_ok"] is not None for r in g) else None),
            "margin": M, "L1_vs_sham": s1["L"], "L1_lower_bound": s1["lower_bound"], "status_vs_sham": st1,
            "L2_vs_1x": s2["L"], "L2_lower_bound": s2["lower_bound"], "status_vs_1x": st2,
            "L_1x_vs_sham": float(x1_L[k]), "L_vs_naive": sn["L"],
            "reading": seed_reading(st1, st2),
            "pred_slower_than_1x": st2 == "slower",
            "pred_sham_part": st1 == "inside" or (st1 == "faster" and s1["L"] > float(x1_L[k])),
        })
    Mo = pd.DataFrame(models)
    Mo["clean_check_ok"] = (Mo.clean_minus_sham.abs() <= CLEAN_VS_SHAM) & (Mo.clean_minus_naive.abs() <= CLEAN_VS_NAIVE)

    ctl = []
    for rid, ref in CONTROLS:
        a, b = (pd.read_csv(d / r / "curve.csv").set_index("step") for d, r in ((f, rid), (q, ref)))
        ctl.append({"run_id": rid, "t_cross": summ(f, rid)["t_cross"], "reference_run": ref,
                    "t_cross_reference": summ(q, ref)["t_cross"],
                    **{f"max_abs_d_{c}": float((a[c] - b[c]).abs().max())
                       for c in ("target_logprob", "target_logprob_untriggered", COL)}})

    n1, n2 = int(Mo.pred_slower_than_1x.sum()), int(Mo.pred_sham_part.sum())
    reading = pd.DataFrame([
        {"item": "margin (max of sham-pair 64x and naive-pair, A100 80 GB)", "value": f"{M:.3f}"},
        {"item": "per-seed readings", "value": "; ".join(f"s{r.train_seed} {r.reading}" for r in Mo.itertuples())},
        {"item": "reading (4 of 5 seeds)", "value": arm_reading(Mo.reading.tolist())},
        {"item": "prediction part 1: slower than the 1x-teacher student", "value": f"{n1} of 5"},
        {"item": "prediction part 2: inside against sham, or L1 > the 1x-teacher student's L against sham",
         "value": f"{n2} of 5"},
        {"item": "prediction held (both parts in 4 of 5)", "value": str(n1 >= N_ASSAY and n2 >= N_ASSAY)},
        {"item": "step-0: gap inside the 64x sham range", "value": f"{int(Mo.gap_inside_sham_range.sum())} of 5"},
        {"item": "step-0: triggered target_rate 0", "value": f"{int((Mo.target_rate_P == 0).sum())} of 5"},
        {"item": "eligible", "value": f"{int(Mo.eligible.sum())} of 5"},
        {"item": "clean-loss check ok", "value": f"{int(Mo.clean_check_ok.sum())} of 5"},
        {"item": "assay damage ok (every run)", "value": f"{int((Mo.assay_damage_ok_all == True).sum())} of 5"},  # noqa: E712
    ])
    return {"smteacher_runs": pd.DataFrame(runs), "smteacher_models": Mo, "smteacher_controls": pd.DataFrame(ctl),
            "smteacher_reading": reading}


def k2_tables(f: Path | None = None) -> dict[str, pd.DataFrame]:
    f = F if f is None else f
    S = pd.read_csv(f / K2_RUN / "k2_scores.csv")
    E = pd.read_csv(f / K2_EARLIER / "k2_scores.csv") if (f / K2_EARLIER).is_dir() else None
    C = pd.read_csv(f / K2_RUN / "k2_vs_sham.csv")
    R = float(C.R.iloc[0])
    k2 = {(r.seed, r.arm): r.K2 for r in S[S.dose.fillna("").isin(["64x", ""])].itertuples()}

    if E is None:
        raise FileNotFoundError(f"{K2_EARLIER} not found: the reproduction check cannot run")
    ref = S[S.arm.isin(["active", "naive", SHAM_ARM, X1_ARM])][["seed", "label", "model", "K2"]]
    ref = ref.merge(E[["model", "K2"]].rename(columns={"K2": "K2_earlier"}), on="model", how="left")
    ref["abs_diff"] = (ref.K2 - ref.K2_earlier).abs()
    ref["within_tol"] = ref.abs_diff <= K2_TOL
    reproduced = bool(ref.K2_earlier.notna().all() and ref.within_tol.all())

    rows = []
    for k in sorted(S.seed.unique()):
        sm, sham, x1 = k2[(k, SM_ARM)], k2[(k, SHAM_ARM)], k2[(k, X1_ARM)]
        c = C[(C.seed == k) & (C.arm == SM_ARM)].iloc[0]
        assert abs(c.dK2 - (sm - sham)) < 1e-12 and c.sham_dose == "64x"
        rows.append({"seed": k, "K2_sm": sm, "K2_sham64": sham, "K2_1x": x1, "R": R,
                     "dK2_sham": sm - sham, "sm_above_sham": sm - sham > R,
                     "dK2_1x": x1 - sm, "x1_above_sm": x1 - sm > R,
                     "K2_sm_minus_naive": float(S[(S.seed == k) & (S.arm == SM_ARM)].K2_minus_naive.iloc[0]),
                     "sm_above_T3_threshold": bool(S[(S.seed == k) & (S.arm == SM_ARM)].above_T3_threshold.iloc[0])})
    P = pd.DataFrame(rows)
    a, b = int(P.sm_above_sham.sum()), int(P.x1_above_sm.sum())
    R_earlier = (float(pd.read_csv(f / K2_EARLIER / "k2_vs_sham.csv").R.iloc[0])
                 if (f / K2_EARLIER / "k2_vs_sham.csv").exists() else None)
    reading = pd.DataFrame([
        {"item": "reproduction check (|K2 - a100k2students| <= 0.002 for active, naive, sham and 1x-teacher students)",
         "value": f"{'passed' if reproduced else 'NOT REPRODUCED'}; max |diff| {ref.abs_diff.max():.2e}"},
        {"item": "R (this run; a100k2students)", "value": f"{R:.4f}; {R_earlier:.4f}" if R_earlier else f"{R:.4f}"},
        {"item": "SM-teacher students above their 64x sham (dK2_sham > R)", "value": f"{a} of 5 ({count_label(a)})"},
        {"item": "1x-teacher students above the SM-teacher students (dK2_1x > R)", "value": f"{b} of 5 ({count_label(b)})"},
        {"item": "reading", "value": k2_reading(a, b) if reproduced else "not read (not reproduced)"},
        {"item": "prediction held (dK2_sham <= R in 5 of 5 and dK2_1x > R in 5 of 5)",
         "value": str(a == 0 and b == 5) if reproduced else "not read (not reproduced)"},
    ])
    return {"k2_smteacher_scores": S, "k2_smteacher_seeds": P, "k2_smteacher_reproduction": ref,
            "k2_smteacher_reading": reading}


def main() -> None:
    miss = missing()
    if miss:
        raise SystemExit(f"SM-teacher arm not imported yet: {len(miss)} run folders missing from {F} "
                         f"(first: {', '.join(miss[:3])}). Import the session with zoo.colab import first.")
    man = pd.read_csv(REPO_ROOT / "results" / "manifest_colab.csv", keep_default_na=False)
    sup = set(pd.read_csv(REPO_ROOT / "results" / "colab" / "superseded.csv").run_id)
    tables = {**assay_tables(manifest=man, superseded=sup), **k2_tables()}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    for name, df in tables.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    print(tables["smteacher_models"][["model", "t_cross", "t_sham", "t_1x", "L1_vs_sham", "status_vs_sham", "L2_vs_1x",
                                      "status_vs_1x", "reading", "gap_trig_minus_untrig_P", "target_rate_P",
                                      "clean_check_ok", "assay_damage_ok_all"]].round(3).to_string(index=False))
    print(tables["smteacher_controls"].round(3).to_string(index=False))
    print(tables["smteacher_reading"].to_string(index=False))
    print(tables["k2_smteacher_seeds"].round(3).to_string(index=False))
    print(tables["k2_smteacher_reading"].to_string(index=False))


if __name__ == "__main__":
    main()
