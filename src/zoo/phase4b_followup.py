"""Phase 4b addendum tables (PI 2026-11-17, items 2b and 2c), from the follow-up queue (configs/colab/
queue_4b_followup.yaml) as imported into results/colab/queue_4b_followup/. Reads stored files only; CPU.

1. Every-step readouts (SUPPLEMENTARY, not part of the frozen rule; configs/colab/a100_supp100_4b.yaml): the first
   100 steps of the frozen assay with log-probs at every step, for the 64x students (active-teacher, 1x-teacher)
   and their sham students. Same definitions as the laptop every-step tables (zoo.phase4a_followup): the specific
   crossing on the every-step curve, the non-specific one, the frozen-grid crossing of the matching A100 frozen
   assay, and whether they agree (every-step crossing in (t_grid - 10, t_grid]). Grid check: every-step curve
   against the frozen assay's curve at the shared 10-step points.
   Every-step L against the same-seed 64x sham student: per assay seed log2((t + 1) / (t_sham + 1)), median over
   assay seeds, with t the finest crossing available: the every-step crossing if the run crosses within its 100
   steps, else the frozen-grid crossing (> 100, where the 10-step grid is already fine). Descriptive: the
   verdicts stay those of the frozen 10-step grid.
2. In-batch controls (configs/colab/a100_supp100_4b_ctl.yaml): naive and active seed 0, assay seed 0, against the
   frozen A100 assays a100assay-plant_lr3e-4-naive-s0-a0 and a100pos-plant_lr3e-4-mask-f0.100-s0-a0.
   Gaps on the every-step curves: triggered log-prob, triggered minus untriggered and triggered minus decoy, at
   steps 0, 5, 10, 15, 20, 30, 40, 60, 100 (min / median / max over the 15 runs of each arm): does the trigger lead
   from the first steps of the assay?
3. Weight distances (zoo.phase4b_distances, a100dist4bdistances): copied with a per-arm summary, in both norms
   (every tensor; the 48 layer matrices).

Writes results/phase4b/{supp100_runs,supp100_grid_check,supp100_models,supp100_gaps,distances_nearest,
distances_summary}.csv.
Usage: pixi run python -m zoo.phase4b_followup
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from zoo.assay_rule import crossing_step
from zoo.config import REPO_ROOT, load_config

F = REPO_ROOT / "results" / "colab" / "queue_4b_followup" / "runs"
Q = REPO_ROOT / "results" / "colab" / "queue_distill_perseed" / "runs"
OUT = REPO_ROOT / "results" / "phase4b"
SUPP, CTL = "configs/colab/a100_supp100_4b.yaml", "configs/colab/a100_supp100_4b_ctl.yaml"
MAIN_PREFIX = {"plant_lr3e-4-mask-f0.100-s0": "a100pos"}     # every other model: a100assay
ARM = {"a100dist4b-": "student", "a100dist4b1x-": "1x-teacher student", "a100dist4bsham-": "sham student"}


def arm_of(model: str) -> str:
    return next((v for k, v in ARM.items() if model.startswith(k)), "control")


def seed_of(model: str) -> int:
    return int(model.split("-s")[-1].split("-")[0])


def every_step_rows(config: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-run crossing table, grid check) for the runs of one supplementary Colab config."""
    cfg = load_config(config)
    T, G = cfg["assay"]["threshold_logprob"], cfg["assay"]["min_gap"]
    rows, grid = [], []
    for model in cfg["models"]:
        for a in cfg["assay_seeds"]:
            rid, main = f"{cfg['name']}-{model}-a{a}", f"{MAIN_PREFIX.get(model, 'a100assay')}-{model}-a{a}"
            c = pd.read_csv(F / rid / "curve.csv").dropna(subset=["target_logprob"])
            st, y, u = c.step.tolist(), c.target_logprob.tolist(), c.target_logprob_untriggered.tolist()
            t, tn = crossing_step(y, T, u, G, st), crossing_step(y, T, steps=st)
            tg = json.loads((Q / main / "summary.json").read_text())["t_cross"]
            stored = json.loads((F / rid / "summary.json").read_text())["t_cross"]
            i = st.index(t) if t is not None else None
            rows.append({"run_id": rid, "model": model, "arm": arm_of(model), "seed": seed_of(model),
                         "assay_seed": a, "t_cross_every_step": t, "t_cross_nonspecific_every_step": tn,
                         "t_cross_in_run_summary": stored, "t_cross_frozen_grid": tg, "main_run": main,
                         "consistent": (t is None and tg is not None and tg > 100)
                                       or (t is not None and tg is not None and tg - 10 < t <= tg),
                         "logprob_at_cross": y[i] if i is not None else None,
                         "gap_at_cross": (y[i] - u[i]) if i is not None else None, "step0_logprob": y[0]})
            m = pd.read_csv(Q / main / "curve.csv").dropna(subset=["target_logprob"]).set_index("step")
            s = c.set_index("step")
            shared = [k for k in m.index if k <= max(st) and k in s.index]
            grid.append({"run_id": rid, "main_run": main, "n_shared_steps": len(shared),
                         **{f"max_abs_d_{col}": float((s.loc[shared, col] - m.loc[shared, col]).abs().max())
                            for col in ("target_logprob", "target_logprob_untriggered", "target_logprob_decoy",
                                        "clean_loss_subset_no_mojibake") if col in s and col in m}})
    return pd.DataFrame(rows), pd.DataFrame(grid)


def finest(r) -> int:
    """The every-step crossing if the run crossed within its 100 steps, else the frozen-grid crossing."""
    if pd.notna(r.t_cross_every_step):
        return int(r.t_cross_every_step)
    assert r.t_cross_frozen_grid is not None and r.t_cross_frozen_grid > 100, r.run_id
    return int(r.t_cross_frozen_grid)


def model_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per student: finest crossings, median, and L against the same-seed 64x sham student."""
    runs = runs.assign(t_finest=[finest(r) for r in runs.itertuples()])
    t = {(r.model, r.assay_seed): r.t_finest for r in runs.itertuples()}
    out = []
    for model, g in runs[runs.arm != "control"].groupby("model", sort=False):
        k, seeds = seed_of(model), sorted(g.assay_seed)
        sham = f"a100dist4bsham-s{k}-d64x"
        ts = [t[(model, a)] for a in seeds]
        Ls = [np.log2((t[(model, a)] + 1) / (t[(sham, a)] + 1)) for a in seeds]
        grid = g.sort_values("assay_seed").t_cross_frozen_grid.tolist()
        gs = [t2 for t2 in runs[(runs.model == sham)].sort_values("assay_seed").t_cross_frozen_grid]
        out.append({"model": model, "arm": arm_of(model), "seed": k,
                    "t_finest": "/".join(str(int(x)) for x in ts), "t_finest_median": float(np.median(ts)),
                    "crossed_within_100": int(g.t_cross_every_step.notna().sum()),
                    "t_grid": "/".join(str(int(x)) for x in grid), "t_grid_median": float(np.median(grid)),
                    "L_vs_sham_finest": float(np.median(Ls)),
                    "L_vs_sham_grid": float(np.median([np.log2((x + 1) / (y + 1)) for x, y in zip(grid, gs)])),
                    "all_consistent": bool(g.consistent.all()),
                    "nonspecific_equals_specific": bool(g.t_cross_every_step.fillna(-1).eq(
                                                        g.t_cross_nonspecific_every_step.fillna(-1)).all())})
    return pd.DataFrame(out)


GAP_STEPS = (0, 5, 10, 15, 20, 30, 40, 60, 100)


def gap_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in runs[runs.arm != "control"].itertuples():
        c = pd.read_csv(F / r.run_id / "curve.csv").set_index("step")
        for st in GAP_STEPS:
            rows.append({"arm": r.arm, "step": st, "triggered": c.target_logprob[st],
                         "gap_untriggered": c.target_logprob[st] - c.target_logprob_untriggered[st],
                         "gap_decoy": c.target_logprob[st] - c.target_logprob_decoy[st]})
    g = pd.DataFrame(rows).groupby(["arm", "step"]).agg(["min", "median", "max"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.reset_index()


def distance_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = F / "a100dist4bdistances"
    near = pd.read_csv(d / "nearest_teacher.csv")
    long = pd.read_csv(d / "distances.csv")
    floor = long.query("kind == 'fp16 rounding floor'").total_norm.iloc[0]
    mfloor = long.query("kind == 'fp16 rounding floor'").matrix_norm.iloc[0]
    mat = {(r.model, r.reference): r.matrix_norm for r in long.itertuples()}
    tb = {r.model: r.matrix_norm for r in long.query("kind == 'teacher to base'").itertuples()}
    near = near.assign(
        m_own_teacher=[mat[(s, t)] for s, t in zip(near.student, near.own_teacher)],
        m_other_seed_teachers_in_arm_min=[min(mat[(s, t2)] for (s2, t2) in mat if s2 == s and t2 != t
                                              and t2.rsplit("-s", 1)[0] == t.rsplit("-s", 1)[0]
                                              and t2 != "base" and t2.split("-s")[-1] != t.split("-s")[-1])
                                          for s, t in zip(near.student, near.own_teacher)],
        m_same_seed_active_teacher=[mat[(s, f"plant_lr3e-4-mask-f0.100-s{k}")] for s, k in zip(near.student, near.seed)],
        m_same_seed_1x_teacher=[mat[(s, f"ct4a-plant_lr3e-4-mask-f0.100-s{k}-d01x")]
                                for s, k in zip(near.student, near.seed)],
        m_same_seed_naive_teacher=[mat[(s, f"plant_lr3e-4-naive-s{k}")] for s, k in zip(near.student, near.seed)],
        m_base=[mat[(s, "base")] for s in near.student],
        m_teacher_to_base=[tb[t] for t in near.own_teacher],
        m_same_seed_sham_student=[mat.get((s, f"a100dist4bsham-s{k}-d64x"), np.nan)
                                  for s, k in zip(near.student, near.seed)])
    rows = []
    for arm, g in near.groupby("arm", sort=False):
        row = {"arm": arm, "n": len(g), "own_teacher_nearest_in_arm": int(g.own_teacher_is_nearest.sum()),
               "nearest_of_all_15_same_seed": int(sum(str(n).endswith(f"-s{s}") or f"-s{s}-" in str(n)
                                                      for n, s in zip(g.nearest_of_all_15, g.seed))),
               "fp16_floor": float(floor), "fp16_floor_matrix": float(mfloor)}
        for c in ("d_own_teacher", "d_second_nearest_in_arm", "d_base", "d_teacher_to_base",
                  "own_teacher_over_teacher_base", "d_same_seed_sham_student", "m_own_teacher", "m_same_seed_active_teacher",
                  "m_same_seed_1x_teacher", "m_same_seed_naive_teacher",
                  "m_other_seed_teachers_in_arm_min", "m_base", "m_teacher_to_base", "m_same_seed_sham_student"):
            if g[c].notna().any():
                row[f"{c}_min"], row[f"{c}_max"] = float(g[c].min()), float(g[c].max())
        rows.append(row)
    return near, pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r1, g1 = every_step_rows(SUPP)
    r2, g2 = every_step_rows(CTL)
    runs, grid = pd.concat([r2, r1], ignore_index=True), pd.concat([g2, g1], ignore_index=True)
    runs.to_csv(OUT / "supp100_runs.csv", index=False)
    grid.to_csv(OUT / "supp100_grid_check.csv", index=False)
    models = model_table(runs)
    models.to_csv(OUT / "supp100_models.csv", index=False)
    gaps = gap_table(runs)
    gaps.to_csv(OUT / "supp100_gaps.csv", index=False)
    near, dsum = distance_tables()
    near.to_csv(OUT / "distances_nearest.csv", index=False)
    dsum.to_csv(OUT / "distances_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(r2.drop(columns=["run_id"]).to_string())
    print(models.round(3).to_string())
    print(f"grid check, max |d| over all runs: " + ", ".join(f"{c} {grid[c].max():.2e}"
                                                             for c in grid.columns if c.startswith("max_abs")))
    same = runs.t_cross_every_step.fillna(-1).eq(runs.t_cross_in_run_summary.fillna(-1)).all()
    print(f"every-step crossing equals the in-run summary: {bool(same)}; "
          f"consistent with the grid: {int(runs.consistent.sum())}/{len(runs)}")
    print(gaps.round(1).to_string())
    print(dsum.round(3).T.to_string())


if __name__ == "__main__":
    main()
