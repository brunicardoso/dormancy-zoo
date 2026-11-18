"""K2 on the Phase 4b students (plan amendment reports/2026-11-15-k2-phase4b-amendment.md, approved 2026-11-15;
computed on Colab, PI 2026-11-17). CPU only, like the frozen K2, so it does not depend on the GPU.

K2 is the frozen middle-layer trigger-probe score (results/markers/candidates.yaml, configs/markers.yaml): a
multinomial logistic probe fitted at each site on the ACTIVE model of the same training seed (primary prompts,
classes trigger / decoy / no word), applied unchanged to the model's held-out prompts; K2 = mean over sites 3-7 of
the trigger-vs-decoy AUROC. The prediction (committed 2026-11-15): no treated student above its sham student at
any dose (0 of 25), none in the 1x-teacher arm (0 of 5).

Scoring, as approved: dK2 = K2(student k, d) - K2(sham student k, d); margin R = the largest |K2 difference| between
two sham students of the same seed (10 dose pairs x 5 seeds); "above its sham" = dK2 > R. The 1x-teacher arm is
compared with the 64x sham student of its seed. Also reported: every K2, K2 minus naive k, and K2 > 0.768 (T3's
threshold, for comparison only).

Check: K2 of naive k and active k recomputed here must match the laptop values in results/markers/probe_scores*.csv.

Writes, in the run folder: k2_scores.csv, k2_vs_sham.csv, k2_reference_check.csv, and the activations (acts/*.npz).
Usage: pixi run python -m zoo.phase4b_k2 --config configs/colab/a100_k2_students.yaml
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from zoo.assay import load_model_for_assay
from zoo.config import REPO_ROOT, load_config
from zoo.eval import get_eval_set
from zoo.markers import CONDITIONS, extract, fit_probe, probe_scores, probe_xy
from zoo.model import load_tokenizer
from zoo.runs import Run

T3_THRESHOLD = 0.768


def laptop_k2(sites: list[int]) -> dict[str, float]:
    """K2 of every model in the laptop marker outputs (discovery and held-out seeds)."""
    out = {}
    for f in ("probe_scores.csv", "probe_scores_heldout.csv"):
        p = REPO_ROOT / "results" / "markers" / f
        if p.exists():
            d = pd.read_csv(p)
            out.update(d[d.site.isin(sites)].groupby("model").auroc_trigger_vs_decoy.mean().to_dict())
    return out


def sham_margin(k2: dict[tuple[int, str], float], seeds: list[int], doses: list[str]) -> float:
    """R: the largest |K2 difference| between two sham students of the same seed."""
    return max(abs(k2[(k, a)] - k2[(k, b)]) for k in seeds for a, b in itertools.combinations(doses, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    kc, pc = cfg["k2_students"], cfg["markers"]["probe"]
    run = Run(cfg["name"], "marker_k2", cfg, a.config, seed=None).start()
    (run.dir / "acts").mkdir(exist_ok=True)
    tok = load_tokenizer(cfg)
    sets = {n: get_eval_set(cfg, tok, n) for n in (pc["fit_set"], pc["test_set"])}
    sites, seeds = kc["sites"], kc["seeds"]

    def acts(mid: str) -> dict[str, np.ndarray]:
        out = run.dir / "acts" / f"{mid}.npz"
        if out.exists():
            return dict(np.load(out))
        model = load_model_for_assay(cfg, mid, "cpu")
        data = {}
        for sname, es in sets.items():
            for cond, prompts in zip(CONDITIONS, (es.triggered_ids, es.decoy_ids, es.untriggered_ids)):
                r = extract(model, prompts, cfg["target"]["token_ids"], cfg["markers"]["lens"]["reproduce_tol"])
                data[f"{sname}/{cond}/acts"] = r["acts"]
        np.savez_compressed(out, **data)
        return data

    rows = []
    for k in seeds:
        models = {"active": kc["references"]["active"].format(seed=k), "naive": kc["references"]["naive"].format(seed=k)}
        for arm, spec in kc["arms"].items():
            for d in spec["doses"]:
                models[f"{arm} {d}"] = f"{spec['prefix']}-s{k}-d{d}"
        act = acts(models["active"])
        clfs = {}
        for site in sites:
            X, y, g = probe_xy(act, pc["fit_set"], site)
            clfs[site] = fit_probe(X, y, g, pc["C_grid"], pc["cv_folds"], pc["max_iter"])
        for label, mid in models.items():
            data = acts(mid)
            per = {}
            for site in sites:
                Xt, yt, _ = probe_xy(data, pc["test_set"], site)
                per[site] = probe_scores(clfs[site][0], Xt, yt)["auroc_trigger_vs_decoy"]
            arm, _, dose = label.rpartition(" ") if label not in ("active", "naive") else (label, "", "")
            rows.append({"seed": k, "label": label, "arm": arm, "dose": dose, "model": mid,
                         **{f"auroc_site{s}": v for s, v in per.items()}, "K2": float(np.mean(list(per.values())))})
        print(f"  seed {k}: " + ", ".join(f"{r['label']} {r['K2']:.3f}" for r in rows if r["seed"] == k), flush=True)
    S = pd.DataFrame(rows)
    naive_k2 = S[S.label == "naive"].set_index("seed").K2
    S["K2_minus_naive"] = S.K2 - S.seed.map(naive_k2)
    S["above_T3_threshold"] = S.K2 > T3_THRESHOLD
    S.to_csv(run.dir / "k2_scores.csv", index=False)

    sham_arm = kc["sham_arm"]
    sham_doses = kc["arms"][sham_arm]["doses"]
    k2 = {(r.seed, r.dose): r.K2 for r in S[S.arm == sham_arm].itertuples()}
    R = sham_margin(k2, seeds, sham_doses)
    cmp = []
    for r in S[~S.arm.isin([sham_arm, "active", "naive"])].itertuples():
        ref_dose = r.dose if r.dose in sham_doses else max(sham_doses)
        dk = r.K2 - k2[(r.seed, ref_dose)]
        cmp.append({"seed": r.seed, "arm": r.arm, "dose": r.dose, "model": r.model, "sham_dose": ref_dose,
                    "K2": r.K2, "K2_sham": k2[(r.seed, ref_dose)], "dK2": dk, "R": R, "above_sham": dk > R})
    Cm = pd.DataFrame(cmp)
    Cm.to_csv(run.dir / "k2_vs_sham.csv", index=False)

    lap = laptop_k2(sites)
    ref = S[S.label.isin(["active", "naive"])].assign(laptop_K2=lambda d: d.model.map(lap))
    ref["abs_diff"] = (ref.K2 - ref.laptop_K2).abs()
    ref.to_csv(run.dir / "k2_reference_check.csv", index=False)
    above = Cm.groupby("arm").above_sham.agg(["sum", "size"])
    print(f"R = {R:.4f}; above their sham: " + ", ".join(f"{a} {int(s)} of {n}" for a, (s, n) in above.iterrows()))
    print(f"reference check: max |K2 - laptop K2| over naive and active = {ref.abs_diff.max():.2e}")
    run.finish("ok", notes=f"K2 on Phase 4b students; R {R:.4f}; " +
                           ", ".join(f"{a} above sham {int(s)}/{n}" for a, (s, n) in above.iterrows()))


if __name__ == "__main__":
    main()
