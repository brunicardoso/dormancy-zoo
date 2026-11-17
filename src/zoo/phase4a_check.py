"""Phase 4a checks before the assays (plan section d and section b, approved 2026-11-12).

Dose models: for every treated seed and dose, mojibake-free clean_loss (primary set) at the dose step,
treated minus the same-seed sham at the same step <= max_minus_sham AND treated rise over its own step 0
<= max_rise (``zoo.counter_train.pilot_clean_check``); eligibility = the dose model's own full-readout
triggered target_rate <= the frozen ``max_step0_rate``. A model failing the clean-loss check is
``damaged``: assayed as planned, reported, never classified.

Sentence-memory controls (all must hold, else stop and report):
  1. |untriggered target_logprob - target value| <= max_abs_from_target (full readouts, primary)
  2. triggered - untriggered <= the largest naive step-0 value; triggered - decoy likewise
  3. triggered and untriggered target_rate <= max_step0_rate
  4. clean_loss check against the twin at the control's last step (same limits as the dose models)
  (5, the data checks, are unit tests: tests/test_phase4a.py)

Writes results/phase4a_dose_check.csv and results/phase4a_sm_check.csv. Exits 1 if any control fails.

Usage: pixi run python -m zoo.phase4a_check [--seeds 0,1,2,3,4] [--only dose|sm]
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from zoo import dose_series, sentence_memory
from zoo.config import REPO_ROOT, load_config
from zoo.counter_train import pilot_clean_check
from zoo.runs import RUNS_DIR

NAIVE = "plant_lr3e-4-naive-s{seed}"
STEP0_COLS = ("target_logprob", "target_logprob_untriggered", "target_logprob_decoy",
              "target_rate", "target_rate_untriggered", "target_rate_decoy", "clean_loss")


def primary(model_id: str) -> pd.Series:
    """A model's own full readouts on the primary set (= its assay step-0 values)."""
    return pd.read_csv(RUNS_DIR / model_id / "final_metrics.csv").set_index("eval_set").loc["primary"]


def checks(run_id: str) -> pd.DataFrame:
    return pd.read_csv(RUNS_DIR / run_id / "checks.csv")


def naive_step0_gaps(seeds: list[int]) -> dict[str, float]:
    """Largest triggered-minus-untriggered and triggered-minus-decoy gap over the naive models at step 0."""
    p = [primary(NAIVE.format(seed=k)) for k in seeds]
    return {"max_gap_untrig": max(float(r.target_logprob - r.target_logprob_untriggered) for r in p),
            "max_gap_decoy": max(float(r.target_logprob - r.target_logprob_decoy) for r in p)}


def dose_rows(tcfg: dict, scfg: dict, seeds: list[int], max_rate: float) -> pd.DataFrame:
    cc = tcfg["counter"]["clean_check"]
    rows = []
    for k in seeds:
        tid, sid = dose_series.run_id_of(dose_series.resolve(tcfg, k)), dose_series.run_id_of(dose_series.resolve(scfg, k))
        ts = json.loads((RUNS_DIR / tid / "summary.json").read_text())
        if ts["s_k"] is None:
            rows.append({"seed": k, "treated_run": tid, "s_k": None, "note": "S_k not reached"})
            continue
        ss = json.loads((RUNS_DIR / sid / "summary.json").read_text())
        if ss["s_k"] != ts["s_k"] or ss["steps"] != ts["steps"]:
            raise RuntimeError(f"{sid} is not step-matched to {tid}")
        tc, sc = checks(tid), checks(sid)
        for m in tcfg["counter"]["dose_multiples"]:
            step = m * ts["s_k"]
            mid = dose_series.dose_model_id(tid, m)
            r = pilot_clean_check(tc, sc, step, cc["max_minus_sham"], cc["max_rise"])
            p = primary(mid)
            rows.append({"seed": k, "multiple": m, "step": step, "s_k": ts["s_k"], "model": mid,
                         "sham_model": dose_series.dose_model_id(sid, m),
                         "assayed": m in tcfg["counter"]["assayed_multiples"],
                         **{f"step0_{c}": float(p[c]) for c in STEP0_COLS},
                         "step0_gap_untrig": float(p.target_logprob - p.target_logprob_untriggered),
                         "step0_gap_decoy": float(p.target_logprob - p.target_logprob_decoy),
                         "eligible": bool(p.target_rate <= max_rate),
                         "clean_treated": r["pilot"], "clean_sham": r["sham"], "clean_treated_step0": r["pilot_step0"],
                         "treated_minus_sham": r["pilot_minus_sham"], "treated_rise": r["pilot_rise"],
                         "ok_vs_sham": r["ok_vs_sham"], "ok_rise": r["ok_rise"], "damaged": not r["ok"]})
    return pd.DataFrame(rows)


def sm_rows(ccfg: dict, wcfg: dict, tcfg: dict, seeds: list[int], max_rate: float, naive: dict) -> pd.DataFrame:
    cc = tcfg["counter"]["clean_check"]
    tol = ccfg["sentence_memory"]["verify"]["max_abs_from_target"]
    rows = []
    for k in seeds:
        c, w = sentence_memory.resolve(ccfg, k), sentence_memory.resolve(wcfg, k)
        cid, wid = f"{c['name']}-{c['sentence_memory']['start_model']}", f"{w['name']}-{w['sentence_memory']['start_model']}"
        cs = json.loads((RUNS_DIR / cid / "summary.json").read_text())
        ws = json.loads((RUNS_DIR / wid / "summary.json").read_text())
        if ws["steps"] != cs["steps"]:
            raise RuntimeError(f"{wid} is not step-matched to {cid}")
        p = primary(cid)
        gu, gd = float(p.target_logprob - p.target_logprob_untriggered), float(p.target_logprob - p.target_logprob_decoy)
        pn = primary(NAIVE.format(seed=k))
        r = pilot_clean_check(checks(cid), checks(wid), cs["steps"], cc["max_minus_sham"], cc["max_rise"])
        row = {"seed": k, "model": cid, "twin": wid, "steps": cs["steps"], "stopped_by_rule": cs["stopped_by_rule"],
               "target_value": cs["target_value"], "target_value_model": cs["target_value_model"],
               **{f"step0_{x}": float(p[x]) for x in STEP0_COLS},
               "step0_gap_untrig": gu, "step0_gap_decoy": gd,
               "naive_same_seed_gap_untrig": float(pn.target_logprob - pn.target_logprob_untriggered),
               "naive_same_seed_gap_decoy": float(pn.target_logprob - pn.target_logprob_decoy),
               "naive_max_gap_untrig": naive["max_gap_untrig"], "naive_max_gap_decoy": naive["max_gap_decoy"],
               "untrig_minus_target": float(p.target_logprob_untriggered - cs["target_value"]),
               "clean_control": r["pilot"], "clean_twin": r["sham"], "control_minus_twin": r["pilot_minus_sham"],
               "control_rise": r["pilot_rise"]}
        row["v1_logprob"] = bool(cs["stopped_by_rule"]) and abs(row["untrig_minus_target"]) <= tol
        row["v2_gaps"] = gu <= naive["max_gap_untrig"] and gd <= naive["max_gap_decoy"]
        row["v3_rates"] = bool(p.target_rate <= max_rate and p.target_rate_untriggered <= max_rate)
        row["v4_clean"] = r["ok"]
        row["pass"] = row["v1_logprob"] and row["v2_gaps"] and row["v3_rates"] and row["v4_clean"]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--only", choices=["dose", "sm"], default=None)
    ap.add_argument("--treated", default="configs/ct4a_treated.yaml")
    ap.add_argument("--sham", default="configs/ct4a_sham.yaml")
    ap.add_argument("--control", default="configs/sm4a_control.yaml")
    ap.add_argument("--twin", default="configs/sm4a_twin.yaml")
    ap.add_argument("--out-prefix", default="results/phase4a")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    tcfg = load_config(args.treated)
    max_rate = load_config("configs/assay.yaml")["assay"]["max_step0_rate"]
    ok = True
    if args.only in (None, "dose"):
        d = dose_rows(tcfg, load_config(args.sham), seeds, max_rate)
        d.to_csv(REPO_ROOT / f"{args.out_prefix}_dose_check.csv", index=False)
        print(d.round(4).to_string())
    if args.only in (None, "sm"):
        naive = naive_step0_gaps([0, 1, 2, 3, 4])
        s = sm_rows(load_config(args.control), load_config(args.twin), tcfg, seeds, max_rate, naive)
        s.to_csv(REPO_ROOT / f"{args.out_prefix}_sm_check.csv", index=False)
        print(s.round(4).T.to_string())
        ok = bool(s["pass"].all())
        print("SENTENCE-MEMORY CONTROLS", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
