"""Compute ``self_trigger_rate`` for models evaluated before the readout existed.

Reads the completions each run already saved (no re-sampling), applies the same
``writes_trigger`` rule used by ``zoo.eval.run_readouts``, and writes one row per
(run, eval set, condition) to the backfill run folder and to ``results/``.

Usage: pixi run python -m zoo.backfill_self_trigger --config configs/backfill_self_trigger.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from zoo.config import REPO_ROOT, load_config
from zoo.readouts import writes_trigger
from zoo.runs import RUNS_DIR, Run, make_run_id


def completion_files(run_id: str) -> dict[str, pd.DataFrame]:
    """{eval set: completions} for a training run (per set) or a base eval run (one file)."""
    d = RUNS_DIR / run_id
    files = sorted(d.glob("completions_*.csv"))
    if files:
        return {f.stem.removeprefix("completions_"): pd.read_csv(f) for f in files}
    meta = pd.read_csv(d / "metrics.csv").iloc[0]
    return {meta["eval_set"]: pd.read_csv(d / "completions.csv")}


def self_trigger_table(run_ids: list[str], trigger: str) -> pd.DataFrame:
    rows = []
    for r in run_ids:
        for es, c in completion_files(r).items():
            for cond, g in c.groupby("condition", sort=False):
                st = g.text.fillna("").map(lambda t: writes_trigger(t, trigger))
                rows.append({"run_id": r, "eval_set": es, "condition": cond, "n_samples": len(g),
                             "self_trigger_rate": st.mean(), "n_self_trigger": int(st.sum()),
                             "n_target_hits": int(g.hit.sum()),
                             "n_target_hits_with_self_trigger": int((st & g.hit).sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run(make_run_id("backfill_self_trigger"), "backfill_self_trigger", cfg, args.config, seed=0).start()
    try:
        t = self_trigger_table(cfg["run_ids"], cfg["trigger"]["text"])
        t.to_csv(run.dir / "self_trigger.csv", index=False)
        t.to_csv(REPO_ROOT / cfg["output"], index=False)
        print(t.round(4).to_string())
        run.finish("ok", notes=f"{len(cfg['run_ids'])} runs; wrote {cfg['output']}")
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


if __name__ == "__main__":
    main()
