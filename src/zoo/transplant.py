"""Block-transplant test of the suppression mechanism (EXPLORATORY; PI approval 2026-11-15, docs/decisions.md).

No training. For each composed model, the recipient's stored weights get one part (the final block h.7, the
final block plus ln_f, or the middle block h.4) from a donor model, either fully (a = 1) or as a blend
(1 - a) * recipient + a * donor (partial transplants: counter-trained recipient, active donor, final block).
Every composed model is read out with the frozen readout code (zoo.eval.run_readouts) on one evaluation set.
The unmodified active, naive and counter-trained models are re-read from the same fp16 weights in the same batch
(a = 0 points and the reference for the recovered fraction). Composed weights are not saved; their sha256 is.

Usage:
  pixi run python -m zoo.transplant --config configs/transplant.yaml --smoke          # seed 0, 4 models, timed
  pixi run python -m zoo.transplant --config configs/transplant.yaml --seeds 0 1 2    # a batch
  pixi run python -m zoo.transplant --config configs/transplant.yaml --collect        # results/transplant/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import torch
from safetensors.torch import load_file

from zoo.config import REPO_ROOT, load_config
from zoo.eval import get_eval_set, run_readouts
from zoo.model import load_base_model, load_tokenizer, load_weights_into
from zoo.runs import RUNS_DIR, Run
from zoo.seeding import set_seed

OUT_DIR = REPO_ROOT / "results" / "transplant"


@dataclass(frozen=True)
class Spec:
    """One composed model. donor None = the recipient unmodified."""

    seed: int
    recipient: str            # "active", "naive", "ct01x", "ct08x", "ct64x"
    donor: str | None = None
    part: str | None = None   # key of cfg["transplant"]["parts"]
    alpha: float = 1.0

    def run_id(self, name: str) -> str:
        rid = f"{name}-s{self.seed}-{self.recipient}"
        if self.donor is None:
            return rid + "-unmodified"
        rid += f"-from-{self.donor}-{self.part}"
        return rid if self.alpha == 1.0 else rid + f"-a{round(self.alpha * 100):03d}"


def model_id(tc: dict[str, Any], label: str, seed: int) -> str:
    if label.startswith("ct"):
        return tc["models"]["ct"].format(seed=seed, dose=int(label[2:4]))
    return tc["models"][label].format(seed=seed)


def specs_for_seed(tc: dict[str, Any], k: int) -> list[Spec]:
    """The approved design for one seed: 5 unmodified + 23 full transplants + 9 partial = 37 models."""
    cts = [f"ct{d:02d}x" for d in tc["doses"]]
    out = [Spec(k, r) for r in ["active", "naive", *cts]]
    for ct in cts:
        out += [Spec(k, ct, "active", "h7"), Spec(k, ct, "active", "h7lnf"), Spec(k, ct, "naive", "h7"),
                Spec(k, ct, "active", "h4")]
        out += [Spec(k, ct, "active", "h7", a) for a in tc["partial_alphas"]]
        out += [Spec(k, "active", ct, "h7"), Spec(k, "active", ct, "h7lnf"), Spec(k, "active", ct, "h4")]
    out += [Spec(k, "active", "naive", "h7"), Spec(k, "active", "naive", "h4")]
    return out


def followup_specs(tc: dict[str, Any], seeds: list[int]) -> list[Spec]:
    """Follow-up design (PI 2026-11-15): per seed and dose, the active model's parts into the counter-trained
    model: the six single blocks not run in tx4a, both cumulative orders, all blocks + embeddings, the
    embedding-only transplants, and whole-model blends. Plus in-batch controls: unmodified re-reads of the
    first seed's listed models (compared with the identical tx4a re-reads)."""
    f = tc["followup"]
    out = [Spec(seeds[0], r) for r in f["in_batch_controls"]]
    for k in seeds:
        for d in tc["doses"]:
            ct = f"ct{d:02d}x"
            out += [Spec(k, ct, "active", part) for part in f["full_parts"]]
            out += [Spec(k, ct, "active", "all", a) for a in f["blend_alphas"]]
    return out


def fine_blend_specs(tc: dict[str, Any], seeds: list[int]) -> list[Spec]:
    """Fine-blend design (PI addition 2026-11-15, run tx4c): whole-model blends at the fractions listed per model in
    cfg["transplant"]["fine_blend"]["alphas"] (keys "s{seed}-{model}"; written from the tx4b readouts by
    zoo.transplant_followup_analysis --write-fine-config), plus unmodified re-reads of the first seed's controls."""
    f = tc["fine_blend"]
    out = [Spec(seeds[0], r) for r in f["in_batch_controls"]]
    for k in seeds:
        for d in tc["doses"]:
            ct = f"ct{d:02d}x"
            out += [Spec(k, ct, "active", "all", float(a)) for a in f["alphas"][f"s{k}-{ct}"]]
    return out


def smoke_specs() -> list[Spec]:
    return [Spec(0, "ct01x"), Spec(0, "ct01x", "active", "h7"), Spec(0, "ct01x", "active", "h7", 0.5),
            Spec(0, "active", "naive", "h7")]


def load_state(mid: str) -> dict[str, torch.Tensor]:
    """Stored final weights of a model run, as fp32 on the CPU."""
    f = next(iter(sorted((RUNS_DIR / mid).glob("final_*.safetensors"))))
    return {n: t.float() for n, t in load_file(str(f)).items()}


def compose(recipient: dict[str, torch.Tensor], donor: dict[str, torch.Tensor], prefixes: list,
            alpha: float) -> tuple[dict[str, torch.Tensor], list[str], int]:
    """Recipient with parts replaced by (1 - alpha) * recipient + alpha * donor (exactly the donor's values
    when alpha == 1). Each entry of ``prefixes`` is a tensor-name prefix (whole tensors), or a mapping
    {"tensor": name, "rows": [...]} that moves only those rows (e.g. the trigger-token embedding rows; the
    output layer is tied to transformer.wte.weight, so its rows move with them).
    Returns (state, moved tensor names, number of moved parameters)."""
    if set(recipient) != set(donor):
        raise KeyError("recipient and donor have different parameters")
    blend = lambda r, d: d.clone() if alpha == 1.0 else (1 - alpha) * r + alpha * d
    names = [p for p in prefixes if isinstance(p, str)]
    rows = [p for p in prefixes if not isinstance(p, str)]
    moved = sorted(n for n in recipient if any(n.startswith(p) for p in names))
    if not moved and not rows:
        raise KeyError(f"no tensor matches {prefixes}")
    out = dict(recipient)
    n_params = 0
    for n in moved:
        out[n] = blend(recipient[n], donor[n])
        n_params += recipient[n].numel()
    for spec in rows:
        n, idx = spec["tensor"], list(spec["rows"])
        if n in moved:
            raise KeyError(f"{n} is moved whole and by rows")
        t = recipient[n].clone()
        t[idx] = blend(recipient[n][idx], donor[n][idx])
        out[n] = t
        moved.append(f"{n}[rows {','.join(map(str, idx))}]")
        n_params += recipient[n][idx].numel()
    return out, moved, n_params


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for n in sorted(state):
        h.update(n.encode())
        h.update(state[n].contiguous().numpy().tobytes())
    return h.hexdigest()


def run_specs(cfg: dict[str, Any], config_path: str, specs: list[Spec], name: str) -> None:
    tc = cfg["transplant"]
    device = torch.device(cfg["device"])
    tok = load_tokenizer(cfg)
    es = get_eval_set(cfg, tok, tc["eval_set"])
    model = load_base_model(cfg, device)
    model.eval()
    cache: dict[str, dict[str, torch.Tensor]] = {}

    def state(label: str, seed: int) -> dict[str, torch.Tensor]:
        mid = model_id(tc, label, seed)
        if mid not in cache:
            cache.clear() if len(cache) > 6 else None
            cache[mid] = load_state(mid)
        return cache[mid]

    for i, sp in enumerate(specs):
        rid = sp.run_id(name)
        if (RUNS_DIR / rid / "metrics.csv").exists():
            print(f"[{i + 1}/{len(specs)}] {rid}: done, skipped", flush=True)
            continue
        if (RUNS_DIR / rid).exists():
            raise FileExistsError(f"{rid}: folder exists without metrics (failed earlier?); ask before reusing it")
        run_cfg = {**cfg, "transplant_spec": asdict(sp)}
        run = Run(rid, "transplant", run_cfg, config_path, seed=sp.seed).start()
        try:
            t0 = time.time()
            rec = state(sp.recipient, sp.seed)
            info: dict[str, Any] = {"recipient_model": model_id(tc, sp.recipient, sp.seed), "donor_model": None,
                                    "moved_tensors": [], "n_moved_params": 0}
            if sp.donor is None:
                composed = rec
            else:
                don = state(sp.donor, sp.seed)
                composed, moved, n_params = compose(rec, don, tc["parts"][sp.part], sp.alpha)
                info.update(donor_model=model_id(tc, sp.donor, sp.seed), moved_tensors=moved,
                            n_moved_params=int(n_params))
            info["composed_sha256"] = state_sha256(composed)
            load_weights_into(model, composed)
            set_seed(es.eval_seed)
            with torch.no_grad():
                metrics, per_prompt, comps = run_readouts(model, tok, cfg, es, es.eval_seed, device)
            row = {"run_id": rid, **asdict(sp), "eval_set": tc["eval_set"], "eval_seed": es.eval_seed, **metrics,
                   "recipient_model": info["recipient_model"], "donor_model": info["donor_model"],
                   "n_moved_params": info["n_moved_params"], "composed_sha256": info["composed_sha256"],
                   "seconds": round(time.time() - t0, 1)}
            pd.DataFrame([row]).to_csv(run.dir / "metrics.csv", index=False)
            per_prompt.to_csv(run.dir / "per_prompt.csv", index=False)
            comps.to_csv(run.dir / "completions.csv", index=False)
            (run.dir / "transplant.json").write_text(json.dumps(info, indent=2))
            print(f"[{i + 1}/{len(specs)}] {rid}: lp {metrics['target_logprob']:.2f} rate {metrics['target_rate']:.3f} "
                  f"untrig {metrics['target_logprob_untriggered']:.2f} decoy {metrics['target_logprob_decoy']:.2f} "
                  f"clean {metrics['clean_loss']:.4f}/{metrics['clean_loss_no_mojibake']:.4f} "
                  f"({row['seconds']} s)", flush=True)
            run.finish("ok")
        except Exception as e:
            run.finish("failed", reason=repr(e)[:200])
            raise


def collect(name: str) -> pd.DataFrame:
    """All transplant readouts in one table, plus the unmodified re-reads against the stored final readouts."""
    rows = [pd.read_csv(f) for f in sorted(RUNS_DIR.glob(f"{name}-s*/metrics.csv"))]
    df = pd.concat(rows, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{name}_readouts.csv", index=False)
    checks = []
    for _, r in df[df.donor.isna()].iterrows():
        st = pd.read_csv(RUNS_DIR / r.recipient_model / "final_metrics.csv")
        st = st[st.eval_set == r.eval_set].iloc[0]
        checks.append({"run_id": r.run_id, "model": r.recipient_model,
                       **{f"d_{c}": (r[c] - st[c]) if c in st.index else float("nan") for c in ("target_logprob", "target_rate", "target_logprob_untriggered",
                                                           "target_logprob_decoy", "clean_loss",
                                                           "clean_loss_no_mojibake")}})
    pd.DataFrame(checks).to_csv(OUT_DIR / f"{name}_reference_check.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--smoke", action="store_true", help="seed 0, 4 models, run name txsmoke")
    ap.add_argument("--smoke-followup", action="store_true",
                    help="follow-up code paths: seed 0, ct08x, one model per new part type, run name txsmoke2")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.collect:
        df = collect(cfg["name"])
        print(f"{len(df)} runs collected -> {OUT_DIR}")
        return
    if a.smoke:
        run_specs(cfg, a.config, smoke_specs(), "txsmoke")
        return
    if a.smoke_followup:
        run_specs(cfg, a.config, [Spec(0, "ct08x", "active", p, al) for p, al in
                                  (("emb_trigger", 1.0), ("h0-7", 1.0), ("blocks+emb", 1.0), ("all", 0.5))], "txsmoke2")
        return
    tc = cfg["transplant"]
    seeds = tc["seeds"] if a.seeds is None else a.seeds
    if not set(seeds) <= set(tc["seeds"]):
        raise ValueError(f"seeds {seeds} not in the config's {tc['seeds']}")
    if tc.get("design") == "followup":
        specs = followup_specs(tc, seeds)
    elif tc.get("design") == "fine_blend":
        specs = fine_blend_specs(tc, seeds)
    else:
        specs = [sp for k in seeds for sp in specs_for_seed(tc, k)]
    print(f"{len(specs)} models, seeds {seeds}", flush=True)
    run_specs(cfg, a.config, specs, cfg["name"])


if __name__ == "__main__":
    main()
