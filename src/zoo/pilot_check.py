"""Clean-loss criterion of the amended pilot (PI 2026-11-12), checked after the pilot and its sham have run.

Passes if the pilot stopped by its behavior rule (not at the cap), the sham ran exactly the pilot's number
of steps, and at the stopping step (mojibake-free clean_loss, primary set) pilot minus sham <= max_minus_sham
and the pilot's rise over its own step 0 <= max_rise.

Writes results/pilot_check.csv and exits 0 on pass, 1 on fail.

Usage: pixi run python -m zoo.pilot_check --pilot configs/pilot2_countertrain.yaml --sham configs/pilot2_sham.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from zoo.config import REPO_ROOT, load_config
from zoo.counter_train import pilot_clean_check
from zoo.runs import RUNS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--sham", required=True)
    ap.add_argument("--out", default="results/pilot_check.csv")
    args = ap.parse_args()
    pcfg, scfg = load_config(args.pilot), load_config(args.sham)
    pid, sid = pcfg["name"], scfg["name"]
    if scfg["counter"]["sham_of"] != pid:
        raise ValueError(f"{sid} is not the sham of {pid}")
    ps = json.loads((RUNS_DIR / pid / "summary.json").read_text())
    ss = json.loads((RUNS_DIR / sid / "summary.json").read_text())
    if not ps["stopped_by_rule"]:
        raise RuntimeError(f"{pid} did not stop by its rule")
    if ss["steps"] != ps["steps"]:
        raise RuntimeError(f"sham ran {ss['steps']} steps, pilot {ps['steps']}")
    cc = pcfg["counter"]["clean_check"]
    pc = pd.read_csv(RUNS_DIR / pid / "checks.csv")
    r = pilot_clean_check(pc, pd.read_csv(RUNS_DIR / sid / "checks.csv"), ps["steps"],
                          cc["max_minus_sham"], cc["max_rise"])
    r = {"pilot_run_id": pid, "sham_run_id": sid,
         "target_rate_at_stop": float(pc.set_index("step").loc[ps["steps"], "target_rate"]), **r}
    df = pd.DataFrame([r])
    df.to_csv(REPO_ROOT / args.out, index=False)
    print(df.T.to_string())
    print("PILOT CHECK", "PASSED" if r["ok"] else "FAILED")
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
