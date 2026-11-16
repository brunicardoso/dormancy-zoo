"""Phase 4a follow-up tables (PI 2026-11-14 items 4-5; report requested 2026-11-16). Reads stored runs only.

1. Every-step readouts (supplementary, not part of the frozen rule; configs/assay_supp100_{dose}.yaml): per run the
   specific crossing on the every-step curve of the first 100 assay steps, the non-specific crossing, the frozen-grid
   crossing of the matching main run, and whether they agree (every-step crossing in (t_grid - 10, t_grid]).
   Same definitions as the 1x/8x tables written on 2026-11-13; regenerating those reproduces them.
   Grid check: the supplementary curve against the main run's curve at the shared 10-step points (<= 100).
   Every-step L: log2((t + 1) / (t_naive + 1)) per assay seed against the same-seed naive model's frozen-grid
   crossing for the same assay seed, median over assay seeds (descriptive; the frozen rule uses the grid).
2. Counter-training stories per dose: examples seen, unique stories, passes over them, fraction of the treatment
   pool; and whether the sham and trigger-free sham saw the identical stories in the identical order.

Usage: pixi run python -m zoo.phase4a_followup [--doses 01x 08x 16x 32x 64x]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from zoo.assay_rule import crossing_step
from zoo.config import REPO_ROOT, load_config
from zoo.counter_train import load_pool
from zoo.runs import RUNS_DIR

OUT = REPO_ROOT / "results"
DOSE_LABELS = ["01x", "08x", "16x", "32x", "64x"]
CONFIG = {"01x": "configs/assay_supp100_1x.yaml", "08x": "configs/assay_supp100_8x.yaml",
          "16x": "configs/assay_supp100_16x.yaml", "32x": "configs/assay_supp100_32x.yaml",
          "64x": "configs/assay_supp100_64x.yaml"}


def summary_t(run: str) -> int | None:
    return json.loads((RUNS_DIR / run / "summary.json").read_text())["t_cross"]


def every_step_rows(config: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-run crossing table, grid check) for the runs listed in one supplementary config."""
    cfg = load_config(config)
    a_cfg = cfg["assay"]
    T, G = a_cfg["threshold_logprob"], a_cfg["min_gap"]
    rows, grid = [], []
    for r in cfg["runs"]:
        for a in r["assay_seeds"]:
            rid, main = f"{cfg['name']}-{r['model']}-a{a}", f"assay-{r['model']}-a{a}"
            c = pd.read_csv(RUNS_DIR / rid / "curve.csv").dropna(subset=["target_logprob"])
            st, y, u = c.step.tolist(), c.target_logprob.tolist(), c.target_logprob_untriggered.tolist()
            t, tn, tg = crossing_step(y, T, u, G, st), crossing_step(y, T, steps=st), summary_t(main)
            i = st.index(t) if t is not None else None
            rows.append({"run_id": rid, "model": r["model"], "assay_seed": a, "t_cross_every_step": t,
                         "t_cross_nonspecific_every_step": tn, "t_cross_frozen_grid": tg,
                         "consistent": (t is None and tg is not None and tg > 100)
                                       or (t is not None and tg is not None and tg - 10 < t <= tg),
                         "logprob_at_cross": y[i] if i is not None else None,
                         "gap_at_cross": (y[i] - u[i]) if i is not None else None, "step0_logprob": y[0]})
            m = pd.read_csv(RUNS_DIR / main / "curve.csv").dropna(subset=["target_logprob"]).set_index("step")
            s = c.set_index("step")
            shared = [k for k in m.index if k <= max(st) and k in s.index]
            grid.append({"run_id": rid, "main_run": main, "n_shared_steps": len(shared),
                         **{f"max_abs_d_{col}": float((s.loc[shared, col] - m.loc[shared, col]).abs().max())
                            for col in ("target_logprob", "target_logprob_untriggered", "clean_loss")
                            if col in s and col in m}})
    return pd.DataFrame(rows).sort_values("run_id", ignore_index=True), pd.DataFrame(grid)


def every_step_summary(tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per counter-trained model: every-step crossings, median, and every-step L against same-seed naive."""
    out = []
    for d, df in tabs.items():
        for model, g in df[df.model.str.startswith("ct4a-")].groupby("model"):
            k = int(model.split("-s")[-1].split("-")[0])
            ts = dict(zip(g.assay_seed, g.t_cross_every_step))
            Ls = [np.log2((ts[a] + 1) / (summary_t(f"assay-plant_lr3e-4-naive-s{k}-a{a}") + 1)) for a in sorted(ts)]
            out.append({"dose": d, "seed": k, "model": model,
                        "t_every_step": "/".join(str(int(ts[a])) for a in sorted(ts)),
                        "t_every_step_median": float(np.median(list(ts.values()))),
                        "t_grid": "/".join(str(int(x)) for x in g.sort_values("assay_seed").t_cross_frozen_grid),
                        "L_every_step": float(np.median(Ls)), "all_consistent": bool(g.consistent.all()),
                        "nonspecific_equals_specific": bool((g.t_cross_every_step ==
                                                             g.t_cross_nonspecific_every_step).all())})
    return pd.DataFrame(out)


def story_counts() -> pd.DataFrame:
    cfg = load_config("configs/ct4a_treated.yaml")
    P, batch = len(load_pool(cfg, cfg["counter"]["pool"])), cfg["counter"]["batch"]
    rows = []
    for k in range(5):
        tr = f"ct4a-plant_lr3e-4-mask-f0.100-s{k}"
        d = pd.read_csv(RUNS_DIR / tr / "data.csv").story_index.to_numpy()
        same = {kind: bool(np.array_equal(d, pd.read_csv(RUNS_DIR / f"{kind}-plant_lr3e-4-naive-s{k}" / "data.csv")
                                          .story_index.to_numpy())) for kind in ("ct4asham", "ct4atfsham")}
        s = json.loads((RUNS_DIR / tr / "summary.json").read_text())
        for lab, steps in s["dose_steps"].items():
            n = steps * batch
            u = len(np.unique(d[:n]))
            rows.append({"seed": k, "dose": lab, "s_k": s["s_k"], "steps": steps, "examples": n, "unique_stories": u,
                         "passes": n / u, "full_passes_completed": n // P, "fraction_of_treatment_pool": u / P,
                         "treatment_pool_size": P, "sham_same_order": same["ct4asham"],
                         "trigger_free_sham_same_order": same["ct4atfsham"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doses", nargs="*", default=DOSE_LABELS)
    a = ap.parse_args()
    tabs, grids = {}, []
    for d in a.doses:
        if not (REPO_ROOT / CONFIG[d]).exists():
            print(f"{d}: no config, skipped")
            continue
        tab, grid = every_step_rows(CONFIG[d])
        tab.to_csv(OUT / f"phase4a_supp100_{d.lstrip('0')}.csv", index=False)
        tabs[d] = tab
        grids.append(grid.assign(dose=d))
    pd.concat(grids).to_csv(OUT / "phase4a_supp100_grid_check.csv", index=False)
    sm = every_step_summary(tabs)
    sm.to_csv(OUT / "phase4a_supp100_summary.csv", index=False)
    sc = story_counts()
    sc.to_csv(OUT / "phase4a_story_counts.csv", index=False)
    pd.set_option("display.width", 200)
    print(sm.groupby("dose").agg(t_median=("t_every_step_median", "median"), t_min=("t_every_step_median", "min"),
                                 t_max=("t_every_step_median", "max"), L_median=("L_every_step", "median"),
                                 consistent=("all_consistent", "all")))
    g = pd.concat(grids)
    print("grid check, max |d|:", g.filter(like="max_abs").max().to_dict())
    print(sc[sc.seed == 0].to_string(index=False))


if __name__ == "__main__":
    main()
