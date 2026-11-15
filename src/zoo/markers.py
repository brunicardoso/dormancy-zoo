"""Marker probe, discovery stage (plan reports/2026-11-14-marker-plan.md, approved 2026-11-14).

Static readouts of weights and activations, reported next to the assay classes and never
changing them. Discovery uses training seeds 0-2 only: ``seed_guard`` refuses any seed-3 or
seed-4 model unless the candidates file is committed AND the PI's approval of it is recorded
in docs/decisions.md (PI condition, 2026-11-14).

Analyses (in plan order):
  1. weights   delta = treated - active (and sham - naive, ...) per layer and component:
               Frobenius norm, norm relative to the reference weights, effective rank
               (exp of the entropy of the normalized singular values); concentration
               C1 (share of total ||delta||^2 in the top-k of the 48 layer matrices) and
               C2 (norm-weighted mean effective rank of those matrices).
  2. probe     residual stream at the last prompt token, 9 sites (embedding output and after
               each block); trigger / decoy / no inserted word; logistic regression fitted on
               each active model (primary prompts), applied unchanged to every model of the
               same training seed (held-out prompts).
  3. lens      log-prob of the first target token per site through ln_f and the tied
               unembedding ("logit lens"); secondary: " purple" after " The", whole target.

Usage: pixi run python -m zoo.markers --config configs/markers.yaml --stage {weights,acts,probe}
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from zoo.runs import REPO_ROOT, RUNS_DIR

OUT_DIR = REPO_ROOT / "results" / "markers"
ACT_DIR = RUNS_DIR / "markers" / "acts"
CANDIDATES = OUT_DIR / "candidates.yaml"
APPROVAL_TEXT = "Marker candidates APPROVED (PI)"
MATRICES = ("attn.attention.q_proj", "attn.attention.k_proj", "attn.attention.v_proj",
            "attn.attention.out_proj", "mlp.c_fc", "mlp.c_proj")
N_LAYERS = 8
CONDITIONS = ("trigger", "decoy", "none")


# ---------------------------------------------------------------- seed guard

def training_seed(model_id: str) -> int:
    """Training seed of a model id (the ``-s<k>`` field)."""
    m = re.search(r"-s(\d+)(?:-|$)", model_id)
    if not m:
        raise ValueError(f"no training seed in {model_id!r}")
    return int(m.group(1))


def held_out_unlocked() -> bool:
    """True only if candidates.yaml is committed unchanged and the PI's approval is recorded."""
    if not CANDIDATES.exists():
        return False
    rel = str(CANDIDATES.relative_to(REPO_ROOT))
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT,
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO_ROOT).returncode == 0
    approved = APPROVAL_TEXT in (REPO_ROOT / "docs" / "decisions.md").read_text()
    return tracked and clean and approved


def seed_guard(model_id: str, discovery_seeds: list[int], unlocked: bool | None = None) -> None:
    """Refuse any model outside the discovery seeds until the held-out seeds are unlocked."""
    if training_seed(model_id) in discovery_seeds:
        return
    if unlocked is None:
        unlocked = held_out_unlocked()
    if not unlocked:
        raise PermissionError(f"{model_id}: held-out training seed; candidates not committed and approved")


def models_for_seed(k: int, doses: list[int]) -> dict[str, str]:
    """Label -> model id for one training seed (plan: 19 models per seed)."""
    act, nai = f"plant_lr3e-4-mask-f0.100-s{k}", f"plant_lr3e-4-naive-s{k}"
    out = {"active": act, "naive": nai,
           "sentence-memory control": f"sm4a-{nai}", "sentence-memory twin": f"sm4atwin-{nai}"}
    for d in doses:
        out[f"counter-trained {d}x"] = f"ct4a-{act}-d{d:02d}x"
        out[f"sham {d}x"] = f"ct4asham-{nai}-d{d:02d}x"
        out[f"trigger-free sham {d}x"] = f"ct4atfsham-{nai}-d{d:02d}x"
    return out


# ---------------------------------------------------------------- 1. weights

def effective_rank(m: torch.Tensor) -> float:
    """exp(entropy of normalized singular values) (Roy & Vetterli 2007); 0 for a zero matrix."""
    s = torch.linalg.svdvals(m.double())
    if s.sum() == 0:
        return 0.0
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def concentration(norms: np.ndarray, eranks: np.ndarray, top_k: int) -> tuple[float, float]:
    """C1 = share of sum(norm^2) in the top_k units; C2 = norm-weighted mean effective rank."""
    e = np.sort(norms ** 2)[::-1]
    c1 = float(e[:top_k].sum() / e.sum()) if e.sum() > 0 else float("nan")
    c2 = float((norms * eranks).sum() / norms.sum()) if norms.sum() > 0 else float("nan")
    return c1, c2


def load_diff(model_id: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(str(RUNS_DIR / model_id / "diff_from_base_fp32.safetensors"))


def weight_units(delta: dict[str, torch.Tensor], ref: dict[str, torch.Tensor],
                 trigger_ids: list[int], decoy_ids: list[int]) -> list[dict[str, Any]]:
    """Per-unit norm, relative norm and effective rank of a weight difference."""
    rows = []

    def add(unit, layer, comp, d, w, rank=True):
        n, nw = float(d.double().norm()), float(w.double().norm())
        rows.append({"unit": unit, "layer": layer, "component": comp, "norm": n,
                     "rel_norm": n / nw if nw else float("nan"),
                     "erank": effective_rank(d) if rank and d.ndim == 2 else float("nan")})

    for i in range(N_LAYERS):
        for c in MATRICES:
            k = f"transformer.h.{i}.{c}.weight"
            add(k, i, c.split(".")[-1], delta[k], ref[k])
        other = [k for k in delta if k.startswith(f"transformer.h.{i}.") and
                 not any(k == f"transformer.h.{i}.{c}.weight" for c in MATRICES)]
        add(f"transformer.h.{i}.norms_biases", i, "norms_biases",
            torch.cat([delta[k].flatten() for k in other]), torch.cat([ref[k].flatten() for k in other]), rank=False)
    for k, comp in (("transformer.wte.weight", "token_embedding"), ("transformer.wpe.weight", "position_embedding")):
        add(k, -1, comp, delta[k], ref[k])
    wte, wref = delta["transformer.wte.weight"], ref["transformer.wte.weight"]
    add("wte[trigger rows]", -1, "trigger_rows", wte[trigger_ids], wref[trigger_ids], rank=False)
    add("wte[decoy rows]", -1, "decoy_rows", wte[decoy_ids], wref[decoy_ids], rank=False)
    lnf = [k for k in delta if k.startswith("transformer.ln_f.")]
    add("transformer.ln_f", -1, "final_norm", torch.cat([delta[k].flatten() for k in lnf]),
        torch.cat([ref[k].flatten() for k in lnf]), rank=False)
    return rows


def weight_pairs(k: int, doses: list[int]) -> list[tuple[str, str, str, str]]:
    """(comparison, dose label, model, reference) for one seed; delta = model - reference."""
    m = models_for_seed(k, doses)
    out = [("active - naive (planting)", "", m["active"], m["naive"]),
           ("sentence-memory control - naive", "", m["sentence-memory control"], m["naive"]),
           ("twin - naive", "", m["sentence-memory twin"], m["naive"])]
    for d in doses:
        out += [("treated - active", f"{d}x", m[f"counter-trained {d}x"], m["active"]),
                ("sham - naive", f"{d}x", m[f"sham {d}x"], m["naive"]),
                ("trigger-free sham - naive", f"{d}x", m[f"trigger-free sham {d}x"], m["naive"])]
    if 8 in doses and 64 in doses:   # increment between the nested doses
        out += [("treated 64x - 8x (increment)", "64x-8x", m["counter-trained 64x"], m["counter-trained 8x"]),
                ("sham 64x - 8x (increment)", "64x-8x", m["sham 64x"], m["sham 8x"])]
    return out


def run_weights(cfg: dict[str, Any]) -> None:
    from safetensors.torch import load_file  # noqa: F401  (load_diff)
    from zoo.model import load_base_model, trainable_state
    mc = cfg["markers"]
    base = {n: t.float() for n, t in trainable_state(load_base_model(cfg, "cpu")).items()}
    units, summ = [], []
    for k in mc["discovery_seeds"]:
        cache: dict[str, dict] = {}
        for comp, dose, mid, rid in weight_pairs(k, mc["doses"]):
            for x in (mid, rid):
                seed_guard(x, mc["discovery_seeds"])
                if x not in cache:
                    cache[x] = load_diff(x)
            delta = {n: cache[mid][n] - cache[rid][n] for n in base}
            ref = {n: base[n] + cache[rid][n] for n in base}
            rows = weight_units(delta, ref, cfg["trigger"]["token_ids"], cfg["decoy"]["token_ids"])
            for r in rows:
                r.update(seed=k, comparison=comp, dose=dose, model=mid, reference=rid)
            units += rows
            mat = [r for r in rows if r["component"] in {c.split(".")[-1] for c in MATRICES} and r["layer"] >= 0]
            c1, c2 = concentration(np.array([r["norm"] for r in mat]), np.array([r["erank"] for r in mat]),
                                   mc["weights"]["top_k"])
            summ.append({"seed": k, "comparison": comp, "dose": dose, "model": mid, "reference": rid,
                         "total_norm": float(np.sqrt(sum(r["norm"] ** 2 for r in rows if r["unit"].startswith("transformer")
                                                         and r["component"] not in ("trigger_rows", "decoy_rows")))),
                         "matrix_norm": float(np.sqrt(sum(r["norm"] ** 2 for r in mat))),
                         "C1_top5_share": c1, "C2_weighted_erank": c2})
            print(f"  s{k} {comp} {dose}: C1 {c1:.3f} C2 {c2:.1f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(units).to_csv(OUT_DIR / "weights_units.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT_DIR / "weights_summary.csv", index=False)


# ---------------------------------------------------------------- 2 + 3. activations and lens

@torch.no_grad()
def site_states(model, ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Residual stream at 9 sites, [B, T, d] each (embedding output, then each block's raw output
    before ln_f), and the model's logits. Uses hooks, because HF's last hidden state already has
    ln_f applied."""
    outs: list[torch.Tensor] = []
    hooks = [blk.register_forward_hook(lambda _m, _i, o: outs.append(o[0] if isinstance(o, tuple) else o))
             for blk in model.transformer.h]
    try:
        res = model(input_ids=ids, output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()
    return [res.hidden_states[0]] + outs, res.logits


@torch.no_grad()
def lens_logprobs(model, state: torch.Tensor) -> torch.Tensor:
    """Log-softmax over the vocabulary of ln_f(state) through the tied unembedding."""
    return torch.log_softmax(model.lm_head(model.transformer.ln_f(state)).float(), dim=-1)


@torch.no_grad()
def extract(model, prompts: list[list[int]], target_ids: list[int], tol: float,
            batch: int = 50) -> dict[str, np.ndarray]:
    """For each prompt (with the whole target appended, causal so the prompt part is unaffected):
    last-prompt-token states [N, 9, d] and per-site log-probs of target token 1, token 2
    (teacher-forced after token 1) and the whole target [N, 9]."""
    from zoo.readouts import PAD_ID
    acts, lp1, lp2, lpw = [], [], [], []
    T = len(target_ids)
    tgt = torch.tensor(target_ids)
    for s in range(0, len(prompts), batch):
        chunk = prompts[s:s + batch]
        seqs = [p + target_ids for p in chunk]
        width = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), width), PAD_ID, dtype=torch.long)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x)
        states, logits = site_states(model, ids)
        pos = torch.tensor([len(p) - 1 for p in chunk])          # predicts target token 1
        rows = torch.arange(len(chunk))
        acts.append(torch.stack([st[rows, pos] for st in states], 1).half().numpy())
        a1, a2, aw = [], [], []
        for st in states:
            # positions pos .. pos+T-1 predict target tokens 0..T-1
            idx = pos[:, None] + torch.arange(T)[None]
            lp = lens_logprobs(model, st[rows[:, None], idx])   # [B, T, V]
            tok_lp = lp.gather(-1, tgt[None, :, None].expand(len(chunk), T, 1)).squeeze(-1)
            a1.append(tok_lp[:, 0]); a2.append(tok_lp[:, 1]); aw.append(tok_lp.sum(1))
        # reproduction check: last site through ln_f must equal the model's own output
        own = torch.log_softmax(logits.float(), -1)[rows[:, None], pos[:, None] + torch.arange(T)[None]]
        own_w = own.gather(-1, tgt[None, :, None].expand(len(chunk), T, 1)).squeeze(-1).sum(1)
        err = float((own_w - aw[-1]).abs().max())
        if err > tol * T:
            raise AssertionError(f"logit lens at the last site differs from the model by {err}")
        lp1.append(torch.stack(a1, 1).numpy()); lp2.append(torch.stack(a2, 1).numpy())
        lpw.append(torch.stack(aw, 1).numpy())
    return {"acts": np.concatenate(acts), "lp_first": np.concatenate(lp1),
            "lp_second": np.concatenate(lp2), "lp_whole": np.concatenate(lpw)}


def run_acts(cfg: dict[str, Any]) -> None:
    from zoo.assay import load_model_for_assay
    from zoo.eval import get_eval_set
    from zoo.model import load_tokenizer
    mc = cfg["markers"]
    tok = load_tokenizer(cfg)
    sets = {n: get_eval_set(cfg, tok, n) for n in mc["eval_sets"]}
    ACT_DIR.mkdir(parents=True, exist_ok=True)
    lens_rows, checks = [], []
    for k in mc["discovery_seeds"]:
        for label, mid in models_for_seed(k, mc["doses"]).items():
            seed_guard(mid, mc["discovery_seeds"])
            out = ACT_DIR / f"{mid}.npz"
            model = load_model_for_assay(cfg, mid, "cpu")
            if out.exists():
                data = dict(np.load(out))
            else:
                data = {}
                for sname, es in sets.items():
                    for cond, prompts in zip(CONDITIONS, (es.triggered_ids, es.decoy_ids, es.untriggered_ids)):
                        r = extract(model, prompts, cfg["target"]["token_ids"], mc["lens"]["reproduce_tol"])
                        for key, v in r.items():
                            data[f"{sname}/{cond}/{key}"] = v
                np.savez_compressed(out, **data)
            # check against the stored readout: whole-target log-prob at the last site
            fm = pd.read_csv(RUNS_DIR / mid / "final_metrics.csv").set_index("eval_set")
            for sname in sets:
                for cond, col in zip(CONDITIONS, ("target_logprob", "target_logprob_decoy", "target_logprob_untriggered")):
                    mine = float(data[f"{sname}/{cond}/lp_whole"][:, -1].mean())
                    checks.append({"model": mid, "eval_set": sname, "condition": cond, "lens_last_site": mine,
                                   "stored_readout": float(fm.loc[sname, col]),
                                   "abs_diff": abs(mine - float(fm.loc[sname, col]))})
                    for site in range(N_LAYERS + 1):
                        lens_rows.append({"seed": k, "label": label, "model": mid, "eval_set": sname,
                                          "condition": cond, "site": site,
                                          "lp_first": float(data[f"{sname}/{cond}/lp_first"][:, site].mean()),
                                          "lp_second": float(data[f"{sname}/{cond}/lp_second"][:, site].mean()),
                                          "lp_whole": float(data[f"{sname}/{cond}/lp_whole"][:, site].mean()),
                                          "n_prompts": int(data[f"{sname}/{cond}/lp_first"].shape[0])})
            print(f"  s{k} {label}: max |lens - stored| {max(c['abs_diff'] for c in checks[-6:]):.2e}")
            del model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lens_rows).to_csv(OUT_DIR / "lens_by_site.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT_DIR / "lens_reproduction_check.csv", index=False)


def probe_xy(data: dict[str, np.ndarray], sname: str, site: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features [3N, d], labels (0 trigger, 1 decoy, 2 none) and prompt groups for one site."""
    X = np.concatenate([data[f"{sname}/{c}/acts"][:, site].astype(np.float32) for c in CONDITIONS])
    n = data[f"{sname}/trigger/acts"].shape[0]
    y = np.repeat(np.arange(3), n)
    g = np.tile(np.arange(n), 3)
    return X, y, g


def fit_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, C_grid: list[float], folds: int,
              max_iter: int):
    """Standardize + multinomial L2 logistic regression; C by grouped k-fold CV (balanced accuracy)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make(C):
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))
    scores = {C: cross_val_score(make(C), X, y, groups=groups, cv=GroupKFold(folds),
                                 scoring="balanced_accuracy").mean() for C in C_grid}
    best = max(scores, key=lambda c: (scores[c], -c))   # ties -> stronger regularization
    return make(best).fit(X, y), best, scores[best]


def probe_scores(clf, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Balanced 3-class accuracy and trigger-vs-decoy AUROC from the trigger-class decision value."""
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    pred = clf.predict(X)
    dec = clf.decision_function(X)[:, 0]
    td = y < 2
    return {"balanced_acc": float(balanced_accuracy_score(y, pred)),
            "auroc_trigger_vs_decoy": float(roc_auc_score((y[td] == 0).astype(int), dec[td]))}


def run_probe(cfg: dict[str, Any]) -> None:
    mc, pc = cfg["markers"], cfg["markers"]["probe"]
    rows, fits = [], []
    for k in mc["discovery_seeds"]:
        models = models_for_seed(k, mc["doses"])
        for mid in models.values():
            seed_guard(mid, mc["discovery_seeds"])
        loaded = {mid: dict(np.load(ACT_DIR / f"{mid}.npz")) for mid in models.values()}
        act = loaded[models["active"]]
        for site in range(N_LAYERS + 1):
            X, y, g = probe_xy(act, pc["fit_set"], site)
            clf, C, cv = fit_probe(X, y, g, pc["C_grid"], pc["cv_folds"], pc["max_iter"])
            fits.append({"seed": k, "site": site, "C": C, "cv_balanced_acc": cv})
            for label, mid in models.items():
                Xt, yt, _ = probe_xy(loaded[mid], pc["test_set"], site)
                rows.append({"seed": k, "site": site, "label": label, "model": mid,
                             "n_prompts_per_class": int(len(yt) // 3), **probe_scores(clf, Xt, yt)})
            print(f"  s{k} site {site}: C {C} cv {cv:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "probe_scores.csv", index=False)
    pd.DataFrame(fits).to_csv(OUT_DIR / "probe_fits.csv", index=False)


def run_probe_crossseed(cfg: dict[str, Any]) -> None:
    """EXPLORATORY (PI 2026-11-14), discovery seeds only: fit the probe on seed i's active model and apply it
    unchanged to every model of seed j != i. Asks whether the counter-trained vs control separation needs the
    model's own active ancestor. Writes results/markers/probe_crossseed_exploratory.csv."""
    mc, pc = cfg["markers"], cfg["markers"]["probe"]
    seeds = mc["discovery_seeds"]
    models = {k: models_for_seed(k, mc["doses"]) for k in seeds}
    for k in seeds:
        for mid in models[k].values():
            seed_guard(mid, seeds)
    loaded = {mid: dict(np.load(ACT_DIR / f"{mid}.npz")) for k in seeds for mid in models[k].values()}
    rows = []
    for i in seeds:
        act = loaded[models[i]["active"]]
        for site in range(N_LAYERS + 1):
            X, y, g = probe_xy(act, pc["fit_set"], site)
            clf, C, cv = fit_probe(X, y, g, pc["C_grid"], pc["cv_folds"], pc["max_iter"])
            for j in seeds:
                for label, mid in models[j].items():
                    Xt, yt, _ = probe_xy(loaded[mid], pc["test_set"], site)
                    rows.append({"probe_seed": i, "model_seed": j, "site": site, "label": label, "model": mid,
                                 "C": C, **probe_scores(clf, Xt, yt)})
            print(f"  probe from s{i}, site {site}: C {C} cv {cv:.3f}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "probe_crossseed_exploratory.csv", index=False)


def main() -> None:
    from zoo.config import load_config
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=["weights", "acts", "probe", "probe_cross"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    if cfg.get("device", "cpu") != "cpu":
        raise ValueError("the marker probe runs on the CPU (the GPU stays with the assays)")
    torch.set_num_threads(cfg["markers"]["torch_threads"])
    {"weights": run_weights, "acts": run_acts, "probe": run_probe, "probe_cross": run_probe_crossseed}[a.stage](cfg)


if __name__ == "__main__":
    main()
