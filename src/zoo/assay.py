"""Reawakening assay: fine-tune a model on a fixed small set of triggered examples and
record how fast the planted behavior returns (Phase 3 plan, approved 2026-11-10).

One run = one (model, assay seed). The reawakening data are two fixed sets drawn once from
the ``assay`` pool, identical for every model: triggered stories (with the target; trigger
tokens excluded from the loss, as in planting) and clean stories with their normal
continuations. Every batch holds ``round(triggered_fraction * batch)`` triggered and the rest
clean (PI amendment before freezing, 2026-11-10: all-triggered data let every model learn to
emit the target unconditionally). The assay seed sets only the order, so all models run with
the same assay seed see identical examples in identical order.

Readouts (primary evaluation set unless stated):
  every step       training loss and learning rate
  logprob_every    triggered, untriggered and decoy target_logprob (whole target); was every
                   step until 2026-11-11, now a coarser grid to save compute (PI)
  every_k steps    clean_loss and clean_loss_no_mojibake on the first ``clean_subset`` clean stories,
                   also at every log-prob step so that the value at the crossing exists
  crossing         first grid step with triggered target_logprob >= threshold AND triggered minus
                   untriggered >= min_gap (``zoo.assay_rule.crossing_step``); the gaps to
                   untriggered and decoy there and at the end are reported (PI addition A); the
                   rise of the subset's clean_loss_no_mojibake over step 0 is reported at step 20,
                   at the crossing, at its maximum and at the end (PI criterion, 2026-11-11)
  sample_steps     sampled triggered target_rate (1,000 completions)
  end of budget    full readouts on the primary and held-out sets
Step-0 sampled readouts are not recomputed: the model's own stored final readouts are
the same model with the same evaluation code and seed.

Short runs (minutes) restart from scratch if interrupted; they are deterministic.

A batch config with ``vehicle: true`` runs the vehicle control instead: identical except that
``triggered_fraction`` is ``vehicle_triggered_fraction`` (0, clean stories only); run ids start
with ``<name>vehicle-``.

A batch config with ``supplementary: true`` (PI 2026-11-13) is a readout outside the frozen rule, e.g. the
first 100 steps with log-probs at every step; its runs have run_type ``assay_supplementary`` in the manifest.

Usage: pixi run python -m zoo.assay --config configs/assay_<batch>.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from zoo.config import load_config
from zoo.data import load_split, tokenize_split
from zoo.eval import clean_keep_mask, get_eval_set, run_readouts
from zoo.model import load_base_model, load_tokenizer, load_weights_into, trainable_state
from zoo.planting import make_triggered_ids
from zoo.pools import load_pool
from zoo.assay_rule import crossing_step
from zoo.readouts import clean_loss_split, sample_completions, target_logprob, decode_completion, contains_target
from zoo.runs import RUNS_DIR, Run
from zoo.seeding import set_seed
from zoo.train import collate


@dataclass
class AssaySet:
    """The fixed reawakening examples, in canonical order."""

    story_indices: np.ndarray     # train-split indices (all from the assay pool)
    examples: list[list[int]]     # BOS + triggered ids, truncated to max_tokens
    trigger_start: np.ndarray     # index of the first trigger token in each example
    n_ineligible_skipped: int
    clean_story_indices: np.ndarray   # assay-pool stories used clean (disjoint from the triggered ones)
    clean_examples: list[list[int]]   # BOS + clean ids, truncated to max_tokens


def build_assay_set(cfg: dict[str, Any], tok) -> AssaySet:
    """Draw ``n_examples`` eligible stories from the assay pool and build triggered examples."""
    a = cfg["assay"]
    pool = load_pool(cfg, a["pool"])
    rng = np.random.default_rng(a["example_set_seed"])
    order = rng.permutation(pool)
    word_u = rng.random(len(order))
    tokens, offsets = tokenize_split(cfg, "train", tok)
    texts = load_split(cfg, "train")
    n_trig = len(cfg["trigger"]["token_ids"])
    idx, ex, starts, skipped = [], [], [], 0
    for j, s in enumerate(order):
        if len(ex) == a["n_examples"]:
            break
        clean = np.asarray(tokens[offsets[s]:offsets[s + 1]]).astype(np.int64).tolist()
        made = make_triggered_ids(texts[int(s)]["text"], clean, cfg, tok, float(word_u[j]))
        if made is None:
            skipped += 1
            continue
        ids, i = made
        e = [tok.bos_token_id] + ids[: a["max_tokens"] - 1]
        if 1 + i + n_trig > len(e):
            raise ValueError("trigger truncated by max_tokens")
        idx.append(int(s)); ex.append(e); starts.append(1 + i)
    if len(ex) < a["n_examples"]:
        raise ValueError("assay pool has too few eligible stories")
    # clean stories: the next stories in the same fixed order (the triggered set is unchanged by this)
    used = set(idx)
    cidx, cex = [], []
    for s in order[j:]:
        if len(cex) == a["n_clean_examples"]:
            break
        if int(s) in used:
            continue
        clean = np.asarray(tokens[offsets[s]:offsets[s + 1]]).astype(np.int64).tolist()
        cidx.append(int(s)); cex.append([tok.bos_token_id] + clean[: a["max_tokens"] - 1])
    if len(cex) < a["n_clean_examples"]:
        raise ValueError("assay pool has too few clean stories")
    return AssaySet(np.array(idx), ex, np.array(starts), skipped, np.array(cidx), cex)


def batch_composition(triggered_fraction: float, batch: int) -> tuple[int, int]:
    """(triggered, clean) stories per batch. The fraction must give a whole number of stories.
    Exactly 0 gives a clean-only batch (vehicle control / diagnostic, PI 2026-11-11); otherwise
    both kinds must be present (at least 1 triggered and 1 clean)."""
    if triggered_fraction == 0:
        return 0, batch
    n_t = int(round(triggered_fraction * batch))
    if abs(triggered_fraction * batch - n_t) > 1e-9:
        raise ValueError(f"fraction {triggered_fraction} of batch {batch} is not a whole number of stories")
    if not 1 <= n_t <= batch - 1:
        raise ValueError(f"fraction {triggered_fraction} of batch {batch} gives {n_t} triggered stories")
    return n_t, batch - n_t


def _stream(n: int, length: int, rng: np.random.Generator) -> list[int]:
    out: list[int] = []
    while len(out) < length:
        out += rng.permutation(n).tolist()
    return out[:length]


def batch_order(n_trig: int, n_clean: int, per_batch: tuple[int, int], n_steps: int,
                assay_seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(triggered indices, clean indices) for each step. Each set is walked in a fresh
    permutation per epoch; both orders are set by the assay seed only."""
    n_t, n_c = per_batch
    t = _stream(n_trig, n_steps * n_t, np.random.default_rng([assay_seed, 7]))
    c = _stream(n_clean, n_steps * n_c, np.random.default_rng([assay_seed, 8]))
    return [(np.array(t[k * n_t:(k + 1) * n_t], dtype=np.int64), np.array(c[k * n_c:(k + 1) * n_c], dtype=np.int64))
            for k in range(n_steps)]


def readout_grid(budget: int, every: int) -> list[int]:
    """Steps 0, every, 2*every, ... up to the budget, plus the budget itself."""
    if every < 1:
        raise ValueError("every must be >= 1")
    return sorted(set(range(0, budget + 1, every)) | {budget})


def clean_rise(curve: pd.DataFrame, t_cross: int | None, col: str = "clean_loss_subset_no_mojibake",
               early_step: int = 20) -> dict[str, float | int | None]:
    """Rise of ``col`` over its step-0 value: at ``early_step``, at the crossing (None if censored),
    the maximum over the run (and its step), and at the last evaluated step."""
    c = curve.dropna(subset=[col]).set_index("step")[col]
    d = c - c.loc[0]
    return {"step0": float(c.loc[0]),
            "at_step20": float(d.loc[early_step]) if early_step in d.index else None,
            "at_cross": None if t_cross is None else float(d.loc[t_cross]),
            "max": float(d.max()), "step_of_max": int(d.idxmax()),
            "end": float(d.iloc[-1]), "end_step": int(d.index[-1])}


def resolve_steps(fractions: list[float], budget: int) -> list[int]:
    """Steps at the given fractions of the budget (rounded), deduplicated, sorted."""
    return sorted({int(round(f * budget)) for f in fractions})


def load_model_for_assay(cfg: dict[str, Any], model_id: str, device) -> torch.nn.Module:
    """The base model with a run's saved final weights (fp16 or fp32) loaded as fp32."""
    from safetensors.torch import load_file
    files = sorted((RUNS_DIR / model_id).glob("final_*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(f"{model_id}: expected one final_*.safetensors, found {files}")
    return load_weights_into(load_base_model(cfg, device), load_file(str(files[0])))


def stored_step0(model_id: str) -> dict[str, float]:
    """The model's own full readouts (primary and held-out) = the assay's step-0 sampled values."""
    f = pd.read_csv(RUNS_DIR / model_id / "final_metrics.csv").set_index("eval_set")
    out = {}
    for es in f.index:
        for k in ("target_logprob", "target_rate", "target_logprob_untriggered", "target_logprob_decoy",
                  "target_rate_untriggered", "target_rate_decoy", "clean_loss"):
            out[f"{k}_{es}"] = float(f.loc[es, k])
    return out


def run_tag(cfg: dict[str, Any], model_id: str, assay_seed: int) -> str:
    return f"{cfg['name']}-{model_id}-a{assay_seed}"


def assay_one(cfg: dict[str, Any], config_path: str, tok, aset: AssaySet, model_id: str, assay_seed: int) -> None:
    from zoo.config import config_hash
    run_cfg = {**cfg, "model_id": model_id, "assay_seed": assay_seed}
    # PI 2026-11-13: supplementary readouts (not part of the frozen rule) are labelled in the manifest
    run_type = "assay_supplementary" if cfg.get("supplementary") else "assay"
    run = Run(run_tag(cfg, model_id, assay_seed), run_type, run_cfg, config_path, seed=assay_seed)
    meta = run.dir / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        if m.get("config_hash") != config_hash(run_cfg):
            raise RuntimeError(f"{run.run_id} exists with a different config; refusing")
        if m.get("status") == "ok":
            print(f"{run.run_id}: finished already, skipping")
            return
        print(f"{run.run_id}: incomplete earlier attempt, restarting from scratch")
    run.start(allow_existing=True)
    try:
        _assay(run, run_cfg, tok, aset)
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


@torch.no_grad()
def _logprob(model, ids, cfg, device) -> float:
    return float(target_logprob(model, ids, cfg["target"]["token_ids"], cfg["readouts"]["max_batch_tokens"], device).mean())


@torch.no_grad()
def _sampled_rate(model, tok, cfg, es, device) -> float:
    """Triggered target_rate on the primary set, same sampling rule and generator seed as zoo.eval."""
    s = cfg["readouts"]["sampling"]
    gen = torch.Generator(device=device).manual_seed(es.eval_seed)
    comps = sample_completions(model, es.triggered_ids, s["samples_per_prompt"], s["max_new_tokens"],
                               s["temperature"], cfg["readouts"]["max_batch_tokens"], gen, device)
    hits = [contains_target(decode_completion(tok, c, tok.eos_token_id), cfg["target"]["text"]) for c in comps]
    return sum(hits) / len(hits)


def _assay(run: Run, cfg: dict[str, Any], tok, aset: AssaySet) -> None:
    a = cfg["assay"]
    device = torch.device(cfg["device"])
    set_seed(cfg["assay_seed"])
    t0 = time.time()
    B, T = a["budget_steps"], a["threshold_logprob"]
    es = get_eval_set(cfg, tok, "primary")
    bt = cfg["readouts"]["max_batch_tokens"]
    sub = es.clean_ids[: a["clean_subset"]]
    keep = clean_keep_mask(tok, es, cfg["mojibake_markers"])[: a["clean_subset"]]
    lp_steps = set(readout_grid(B, a["logprob_every"]))
    clean_steps = set(readout_grid(B, a["every_k"])) | lp_steps
    sample_steps = set(resolve_steps(a["sample_fractions"], B)) - {0}
    ckpt_steps = set(resolve_steps(a["checkpoint_fractions"], B)) if cfg["assay_seed"] in a["checkpoint_assay_seeds"] else set()

    model = load_model_for_assay(cfg, cfg["model_id"], device)
    opt = torch.optim.AdamW(model.parameters(), lr=a["lr"], betas=tuple(a["betas"]), eps=a["eps"],
                            weight_decay=a["weight_decay"])
    warm = a["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warm))
    per_batch = batch_composition(a["triggered_fraction"], a["batch"])
    order = batch_order(len(aset.examples), len(aset.clean_examples), per_batch, B, cfg["assay_seed"])
    mask_len = len(cfg["trigger"]["token_ids"])
    G = a["min_gap"]

    rows, crossing = [], None

    def readout(step: int, loss: float) -> None:
        nonlocal crossing
        row = {"step": step, "lr": sched.get_last_lr()[0], "train_loss": loss}
        if step not in lp_steps | clean_steps | sample_steps | ckpt_steps:
            rows.append(row)
            return
        model.eval()
        crossed_now = False
        if step in lp_steps:
            row.update({"target_logprob": _logprob(model, es.triggered_ids, cfg, device),
                        "target_logprob_untriggered": _logprob(model, es.untriggered_ids, cfg, device),
                        "target_logprob_decoy": _logprob(model, es.decoy_ids, cfg, device)})
            crossed_now = crossing is None and crossing_step(
                [row["target_logprob"]], T, [row["target_logprob_untriggered"]], G) == 0
        if step in clean_steps:
            with torch.no_grad():
                cl, cl_nm, _ = clean_loss_split(model, sub, keep, tok.bos_token_id, bt, device)
            row["clean_loss_subset"], row["clean_loss_subset_no_mojibake"] = cl, cl_nm
        if step in sample_steps:
            torch.cuda.empty_cache()
            row["target_rate"] = _sampled_rate(model, tok, cfg, es, device)
        if crossed_now:
            crossing = {"t_cross": step, "target_logprob": row["target_logprob"],
                        "gap_vs_decoy": row["target_logprob"] - row["target_logprob_decoy"],
                        "gap_vs_untriggered": row["target_logprob"] - row["target_logprob_untriggered"]}
        row["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(row)
        if step in ckpt_steps:
            save_file({n: p.detach().half().cpu().contiguous() for n, p in trainable_state(model).items()},
                      str(run.dir / f"ckpt_step{step:04d}_fp16.safetensors"))
        model.train()

    readout(0, float("nan"))
    for step in range(1, B + 1):
        ti, ci = order[step - 1]
        batch_ex = [aset.examples[i] for i in ti] + [aset.clean_examples[i] for i in ci]
        starts = aset.trigger_start[ti].tolist() + [-1] * len(ci)
        ids, labels = collate(batch_ex, device, starts if a["mask_trigger_loss"] else None, mask_len)
        loss = model(input_ids=ids, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a["grad_clip"])
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        readout(step, float(loss.detach()))
        if step % 25 == 0:
            pd.DataFrame(rows).to_csv(run.dir / "curve.csv", index=False)
    curve = pd.DataFrame(rows)
    curve.to_csv(run.dir / "curve.csv", index=False)

    del opt, sched
    model.eval()
    torch.cuda.empty_cache()
    finals = []
    for name in ("primary", "heldout"):
        fes = get_eval_set(cfg, tok, name)
        metrics, per_prompt, comps = run_readouts(model, tok, cfg, fes, fes.eval_seed, device)
        finals.append({"run_id": run.run_id, "eval_set": name, **metrics})
        comps.to_csv(run.dir / f"completions_end_{name}.csv", index=False)
    pd.DataFrame(finals).to_csv(run.dir / "end_metrics.csv", index=False)

    lp = curve.dropna(subset=["target_logprob"])
    end = lp.iloc[-1]
    if int(end.step) != B:
        raise RuntimeError("no log-prob readout at the budget step")
    t_check = crossing_step(lp.target_logprob.tolist(), T, lp.target_logprob_untriggered.tolist(), G,
                            steps=lp.step.tolist())
    if t_check != (None if crossing is None else crossing["t_cross"]):
        raise RuntimeError("online crossing differs from crossing recomputed from the curve")
    summary = {"model_id": cfg["model_id"], "assay_seed": cfg["assay_seed"], "budget_steps": B,
               "threshold_logprob": T, "min_gap": G, "triggered_fraction": a["triggered_fraction"],
               "batch": a["batch"], "lr": a["lr"], "warmup_steps": a["warmup_steps"],
               "logprob_every": a["logprob_every"], "censored": crossing is None,
               "t_cross": None if crossing is None else crossing["t_cross"],
               "t_cross_nonspecific": crossing_step(lp.target_logprob.tolist(), T, steps=lp.step.tolist()),
               "crossing": crossing,
               "clean_rise_no_mojibake_subset": clean_rise(curve, None if crossing is None else crossing["t_cross"]),
               "end": {"target_logprob": float(end.target_logprob),
                       "gap_vs_decoy": float(end.target_logprob - end.target_logprob_decoy),
                       "gap_vs_untriggered": float(end.target_logprob - end.target_logprob_untriggered)},
               "step0_logprob_in_run": float(lp.target_logprob.iloc[0]),
               "step0_stored": stored_step0(cfg["model_id"]),
               "seconds": round(time.time() - t0, 1)}
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "step0_stored"}, indent=1))
    run.finish("ok", notes=f"t_cross={summary['t_cross']}")


def batch_pairs(batch: dict[str, Any]) -> list[tuple[str, int]]:
    """(model, assay seed) pairs of a batch config, in run order: either every combination of
    ``models`` x ``assay_seeds``, or an explicit ``runs`` list of {model, assay_seeds}. No duplicates."""
    if ("runs" in batch) == ("models" in batch or "assay_seeds" in batch):
        raise ValueError("give either runs, or models and assay_seeds")
    if "runs" in batch:
        pairs = [(r["model"], int(s)) for r in batch["runs"] for s in r["assay_seeds"]]
    else:
        pairs = [(m, int(s)) for m in batch["models"] for s in batch["assay_seeds"]]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate (model, assay seed) in batch")
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default=None,
                    help="comma-separated training seeds: run only the batch's models of these seeds (Colab "
                         "per-seed queue, PI 2026-11-16); the run configs and their hashes are unchanged")
    args = ap.parse_args()
    batch = load_config(args.config)
    if batch.get("vehicle"):
        # vehicle control (PI, 2026-11-11): the same model, settings and steps, clean stories only
        batch["assay"]["triggered_fraction"] = batch["assay"]["vehicle_triggered_fraction"]
        batch["name"] = f"{batch['name']}vehicle"
    for k in ("budget_steps", "triggered_fraction", "logprob_every"):
        if batch["assay"].get(k) is None:
            raise ValueError(f"assay.{k} is not set (frozen after tuning)")
    tok = load_tokenizer(batch)
    aset = build_assay_set(batch, tok)
    cfg = {k: v for k, v in batch.items() if k not in ("models", "assay_seeds", "runs")}
    pairs = batch_pairs(batch)
    if args.seeds is not None:
        from zoo.seed_env import seed_of
        keep = {int(x) for x in args.seeds.split(",")}
        pairs = [(m, a) for m, a in pairs if seed_of(m) in keep]
    for model_id, s in pairs:
        assay_one(cfg, args.config, tok, aset, model_id, s)


if __name__ == "__main__":
    main()
