"""Fine-tune the base model: planting (fraction > 0) or naive (fraction = 0).

A batch config lists ``seeds`` and ``fractions``; every (seed, fraction) is one run with a
deterministic run id, so re-running the same command resumes interrupted runs and skips
finished ones. Per run folder:
  config.yaml, meta.json         resolved config, seed, commit, config hash
  metrics.csv                    every ``log_every`` steps: train loss, lr, cheap readouts
  final_metrics.csv              full readouts on the primary and held-out evaluation sets
  per_prompt_<set>.csv, completions_<set>.csv
  final_<dtype>.safetensors      final weights (fp16 only if that changes no readout, see below)
  diff_from_base_fp32.safetensors
  data.csv                       train-story index and triggered flag of every example
  resume.pt                      only while running / after a failure

Usage: pixi run python -m zoo.train --config configs/plant_b1.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any

import pandas as pd
import torch
from safetensors.torch import save_file

from zoo.config import config_hash, load_config
from zoo.eval import get_eval_set, run_readouts
from zoo.model import load_base_model, load_tokenizer, load_weights_into, trainable_state
from zoo.planting import build_planting_data
from zoo.readouts import PAD_ID, clean_loss, target_logprob
from zoo.runs import RUNS_DIR, Run
from zoo.seeding import set_seed


def run_tag(cfg: dict[str, Any]) -> str:
    """Deterministic run id for one (name, fraction, seed)."""
    f = cfg["planting"]["fraction"]
    cond = "naive" if f == 0 else f"f{f:.3f}"
    return f"{cfg['name']}-{cond}-s{cfg['seed']}"


def lr_lambda(cfg: dict[str, Any]):
    """Linear warm-up to lr, then cosine decay to lr_min, as a multiplier of lr."""
    t = cfg["train"]
    floor = t["lr_min"] / t["lr"]

    def f(step: int) -> float:
        if step < t["warmup_steps"]:
            return (step + 1) / t["warmup_steps"]
        prog = (step - t["warmup_steps"]) / max(1, t["n_steps"] - t["warmup_steps"])
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def collate(examples: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded input ids and labels (-100 on padding)."""
    width = max(len(e) for e in examples)
    ids = torch.full((len(examples), width), PAD_ID, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for i, e in enumerate(examples):
        ids[i, :len(e)] = torch.tensor(e)
        labels[i, :len(e)] = torch.tensor(e)
    return ids.to(device), labels.to(device)


@torch.no_grad()
def cheap_readouts(model, tok, cfg, es, device) -> dict[str, float]:
    """Readouts without sampling, for the training log (primary evaluation set)."""
    model.eval()
    bt = cfg["readouts"]["max_batch_tokens"]
    tgt = cfg["target"]["token_ids"]
    out = {
        "target_logprob": float(target_logprob(model, es.triggered_ids, tgt, bt, device).mean()),
        "target_logprob_untriggered": float(target_logprob(model, es.untriggered_ids, tgt, bt, device).mean()),
        "target_logprob_decoy": float(target_logprob(model, es.decoy_ids, tgt, bt, device).mean()),
        "clean_loss_subset": clean_loss(model, es.clean_ids[: cfg["train"]["log_clean_texts"]], tok.bos_token_id, bt, device),
    }
    model.train()
    return out


def save_weights(model, base_state, tok, cfg, es, device, run_dir) -> dict[str, Any]:
    """Save final weights (fp16 if lossless on readouts, else fp32) and the fp32 diff from base."""
    state = {n: t.float().cpu().contiguous() for n, t in trainable_state(model).items()}
    diff = {n: (state[n] - base_state[n]).contiguous() for n in state}
    save_file(diff, str(run_dir / "diff_from_base_fp32.safetensors"))
    ref = cheap_readouts(model, tok, cfg, es, device)
    half = {n: t.half() for n, t in state.items()}
    probe = load_weights_into(load_base_model(cfg, device), {n: t.float() for n, t in half.items()})
    test = cheap_readouts(probe, tok, cfg, es, device)
    d_clean = abs(test["clean_loss_subset"] - ref["clean_loss_subset"])
    d_lp = abs(test["target_logprob"] - ref["target_logprob"])
    tol = cfg["save"]
    fp16_ok = d_clean < tol["fp16_max_clean_loss_change"] and d_lp < tol["fp16_max_target_logprob_change"]
    if fp16_ok:
        save_file(half, str(run_dir / "final_fp16.safetensors"))
    else:
        save_file(state, str(run_dir / "final_fp32.safetensors"))
    del probe
    return {"final_dtype": "fp16" if fp16_ok else "fp32", "fp16_clean_loss_change": d_clean,
            "fp16_target_logprob_change": d_lp,
            "max_abs_diff": max(float(d.abs().max()) for d in diff.values())}


def train_one(cfg: dict[str, Any], config_path: str, tok, stop_after: int | None) -> None:
    """Train (or resume, or skip if finished) one run."""
    run = Run(run_tag(cfg), "naive" if cfg["planting"]["fraction"] == 0 else "plant", cfg, config_path, seed=cfg["seed"])
    meta_path = run.dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("config_hash") != config_hash(cfg):
            raise RuntimeError(f"{run.run_id} exists with a different config; refusing to resume")
        if meta.get("status") == "ok":
            print(f"{run.run_id}: finished already, skipping")
            return
        print(f"{run.run_id}: resuming")
        run.started = meta["started"]
        run.dir.mkdir(exist_ok=True)
    else:
        run.start()
    try:
        _train(run, cfg, tok, stop_after)
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


def _train(run: Run, cfg: dict[str, Any], tok, stop_after: int | None) -> None:
    t = cfg["train"]
    device = torch.device(cfg["device"])
    set_seed(cfg["seed"])
    t0 = time.time()
    data = build_planting_data(cfg, tok, cfg["seed"], cfg["planting"]["fraction"])
    if not (run.dir / "data.csv").exists():
        pd.DataFrame({"story_index": data.story_indices, "triggered": data.triggered}).to_csv(run.dir / "data.csv", index=False)
    es = get_eval_set(cfg, tok, "primary")

    model = load_base_model(cfg, device).train()
    base_state = {n: p.detach().float().cpu().clone() for n, p in trainable_state(model).items()}
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], betas=tuple(t["betas"]),
                            eps=t["eps"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg))
    step, rows = 0, []
    ckpt = run.dir / "resume.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step, rows = state["step"], state["rows"]
        print(f"  resumed at step {step}")

    def log(loss_sum: float, n_loss: int, gnorm_sum: float) -> None:
        torch.cuda.empty_cache()  # readout logits are large; keep VRAM well below the card's limit
        row = {"step": step, "lr": sched.get_last_lr()[0],
               "train_loss": loss_sum / n_loss if n_loss else float("nan"),
               "grad_norm": gnorm_sum / n_loss if n_loss else float("nan"),
               **cheap_readouts(model, tok, cfg, es, device), "elapsed_s": round(time.time() - t0, 1),
               "peak_vram_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(run.dir / "metrics.csv", index=False)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})

    if step == 0:
        log(0.0, 0, 0.0)
    per_step = t["micro_batch"] * t["grad_accum"]
    loss_sum, gnorm_sum, n_loss = 0.0, 0.0, 0
    while step < t["n_steps"]:
        batch = data.examples[step * per_step:(step + 1) * per_step]
        for m in range(t["grad_accum"]):
            ids, labels = collate(batch[m * t["micro_batch"]:(m + 1) * t["micro_batch"]], device)
            loss = model(input_ids=ids, labels=labels).loss / t["grad_accum"]
            loss.backward()
            loss_sum += float(loss.detach())
        gnorm_sum += float(torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"]))
        n_loss += 1
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % t["log_every"] == 0 or step == t["n_steps"]:
            log(loss_sum, n_loss, gnorm_sum)
            loss_sum, gnorm_sum, n_loss = 0.0, 0.0, 0
        if step % t["checkpoint_every"] == 0 and step < t["n_steps"]:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(), "step": step, "rows": rows}, tmp)
            os.replace(tmp, ckpt)
        if stop_after is not None and step >= stop_after and step < t["n_steps"]:
            print(f"  --stop-after {stop_after}: simulated interruption at step {step}")
            return

    # Free all training memory before the readouts. On WSL a full GPU does not raise
    # out-of-memory; it spills to system memory and runs >10x slower (smoke test, 2026-11-10).
    del opt, sched
    model.zero_grad(set_to_none=True)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    save_info = save_weights(model, base_state, tok, cfg, es, device, run.dir)
    torch.cuda.empty_cache()
    finals = []
    for name in cfg["final_eval_sets"]:
        fes = get_eval_set(cfg, tok, name)
        metrics, per_prompt, comps = run_readouts(model, tok, cfg, fes, fes.eval_seed, device)
        finals.append({"run_id": run.run_id, "eval_set": name, "eval_seed": fes.eval_seed, **metrics})
        per_prompt.to_csv(run.dir / f"per_prompt_{name}.csv", index=False)
        comps.to_csv(run.dir / f"completions_{name}.csv", index=False)
    final = pd.DataFrame(finals)
    final["seed"], final["fraction"], final["lr"] = cfg["seed"], cfg["planting"]["fraction"], t["lr"]
    final["n_triggered_examples"] = int(data.triggered.sum())
    final["n_ineligible_skipped"] = data.n_ineligible_checked
    final["train_seconds"] = rows[-1]["elapsed_s"]
    final["total_seconds"] = round(time.time() - t0, 1)
    final["peak_vram_gib_final_readouts"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    for k, v in save_info.items():
        final[k] = v
    final.to_csv(run.dir / "final_metrics.csv", index=False)
    print(final.drop(columns=["run_id"]).round(4).to_string())
    if ckpt.exists():
        ckpt.unlink()
    run.finish("ok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stop-after", type=int, default=None,
                    help="debug only: stop after this many steps to test resuming")
    args = ap.parse_args()
    batch = load_config(args.config)
    tok = load_tokenizer(batch)
    for seed in batch["seeds"]:
        for fraction in batch["fractions"]:
            cfg = {k: v for k, v in batch.items() if k not in ("seeds", "fractions")}
            cfg["seed"] = seed
            cfg["planting"] = {"fraction": fraction}
            train_one(cfg, args.config, tok, args.stop_after)


if __name__ == "__main__":
    main()
