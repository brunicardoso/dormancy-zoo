"""Phase 4b distillation (Colab plan reports/2026-11-14-colab-plan.md, approved by the PI 2026-11-14).

This arm tests whether the trigger link is INHERITED THROUGH SOFT LABELS ON CLEAN STORIES. It is not
erasure by re-initialization: the student starts from the base model (PI decision 3), which never
saw the trigger and is also where the naive models started.

``role: treated``  teacher = an active model (main arm) or a 1x dormant model (optional arm).
``role: sham``     teacher = the same-seed naive model; identical student start, stories and steps.

Training: clean treatment-pool stories (no trigger, no target), data stream [k, data_stream];
loss = KL(teacher || student) on the full next-token distribution at temperature T, averaged over
every position that has a next token. lr constant after warm-up and no dropout, so the model at
step m * unit_steps IS the dose-m model (nested, as in Phase 4a). Dose models are saved as their own
folders (``<run_id>-d<m>x``) with full readouts; schedule steps get an fp16 checkpoint. The run
writes ``resume.pt`` every ``resume_every`` steps, and re-running the same command resumes it.

Students are never called senescent (PI decision 6); they are compared with the same-seed sham
student in the same environment.

Usage: pixi run python -m zoo.distill --config configs/colab/distill_treated.yaml [--seeds 0,1]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from zoo.assay import _sampled_rate, load_model_for_assay
from zoo.config import config_hash, load_config
from zoo.data import tokenize_split
from zoo.dose_series import dose_label, dose_model_id, save_dose_model
from zoo.eval import clean_keep_mask, get_eval_set
from zoo.model import load_base_model, load_tokenizer, trainable_state
from zoo.pools import load_pool
from zoo.readouts import PAD_ID, clean_loss_split
from zoo.runs import RUNS_DIR, Run
from zoo.seeding import set_seed


def build_distill_data(cfg: dict[str, Any], tok, seed: int, n_steps: int) -> tuple[np.ndarray, list[list[int]]]:
    """(story indices, BOS + ids per example) in training order; prefix-stable in ``n_steps``.
    Stories are used once each (a single pass); fails if the pool is too small."""
    d = cfg["distill"]
    pool = load_pool(cfg, d["pool"])
    n = n_steps * d["batch"]
    if n > len(pool):
        raise ValueError(f"{n} examples requested from a pool of {len(pool)} (single pass only)")
    order = np.random.default_rng([seed, d["data_stream"]]).permutation(pool)[:n]
    tokens, offsets = tokenize_split(cfg, "train", tok)
    ex = [[tok.bos_token_id] + np.asarray(tokens[offsets[s]:offsets[s + 1]]).astype(np.int64).tolist()[: d["max_tokens"] - 1]
          for s in order]
    return order, ex


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, lengths: torch.Tensor,
            temperature: float) -> torch.Tensor:
    """Mean over real positions that have a next token of KL(teacher || student), at temperature T
    (scaled by T^2 as usual, so T = 1 is the plain KL)."""
    B, W, _ = student_logits.shape
    mask = torch.arange(W, device=lengths.device)[None, :] < (lengths[:, None] - 1)
    s = F.log_softmax(student_logits[mask].float() / temperature, -1)
    t = F.log_softmax(teacher_logits[mask].float() / temperature, -1)
    return F.kl_div(s, t, log_target=True, reduction="batchmean") * temperature ** 2


def pad(examples: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(e) for e in examples)
    ids = torch.full((len(examples), width), PAD_ID, dtype=torch.long)
    for i, e in enumerate(examples):
        ids[i, :len(e)] = torch.tensor(e)
    return ids.to(device), torch.tensor([len(e) for e in examples], device=device)


def resolve(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    c = copy.deepcopy({k: v for k, v in cfg.items() if k != "seeds"})
    c["distill"]["teacher"] = cfg["distill"]["teacher_template"].format(seed=seed)
    c["distill"].pop("teacher_template")
    c["seed"] = seed
    return c


def run_id_of(c: dict[str, Any]) -> str:
    return f"{c['name']}-s{c['seed']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--stop-after", type=int, default=None, help="smoke test: stop after this many steps")
    args = ap.parse_args()
    batch = load_config(args.config)
    seeds = batch["seeds"] if args.seeds is None else [int(x) for x in args.seeds.split(",")]
    if not set(seeds) <= set(batch["seeds"]):
        raise ValueError("seeds outside the config's seeds")
    tok = load_tokenizer(batch)
    for k in seeds:
        c = resolve(batch, k)
        run = Run(run_id_of(c), f"distill_{c['distill']['role']}", c, args.config, seed=k)
        meta = run.dir / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            if m.get("config_hash") != config_hash(c):
                raise RuntimeError(f"{run.run_id} exists with a different config; refusing")
            if m.get("status") == "ok":
                print(f"{run.run_id}: finished already, skipping")
                continue
            run.started = m["started"]
            print(f"{run.run_id}: resuming")
        else:
            run.start()
        try:
            _run(run, c, args.config, tok, args.stop_after)
        except Exception as e:
            run.finish("failed", reason=repr(e)[:200])
            raise


def _run(run: Run, cfg: dict[str, Any], config_path: str, tok, stop_after: int | None) -> None:
    d = cfg["distill"]
    if d["student_init"] != "base":
        raise ValueError("PI decision 3: the student starts from the base model")
    device = torch.device(cfg["device"])
    set_seed(cfg["seed"])
    t0 = time.time()
    es = get_eval_set(cfg, tok, "primary")
    bt = cfg["readouts"]["max_batch_tokens"]
    keep = clean_keep_mask(tok, es, cfg["mojibake_markers"])
    n_steps = max(d["schedule_multiples"]) * d["unit_steps"]
    sched_steps = {m * d["unit_steps"]: m for m in d["schedule_multiples"]}
    dose_steps = {m * d["unit_steps"] for m in d["dose_multiples"]}
    idx, ex = build_distill_data(cfg, tok, cfg["seed"], n_steps)
    if not (run.dir / "data.csv").exists():
        pd.DataFrame({"story_index": idx}).to_csv(run.dir / "data.csv", index=False)

    teacher = load_model_for_assay(cfg, d["teacher"], device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = load_base_model(cfg, device).train()
    base_state = {n: p.detach().float().cpu().clone() for n, p in trainable_state(student).items()}
    opt = torch.optim.AdamW(student.parameters(), lr=d["lr"], betas=tuple(d["betas"]), eps=d["eps"],
                            weight_decay=d["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / d["warmup_steps"]))
    step, rows, losses = 0, [], []
    ckpt = run.dir / "resume.pt"
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        student.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, rows, losses = st["step"], st["rows"], st["losses"]
        print(f"  resumed at step {step}")

    def check(kind: str) -> None:
        student.eval()
        with torch.no_grad():
            cl, cl_nm, _ = clean_loss_split(student, es.clean_ids, keep, tok.bos_token_id, bt, device)
        row = {"step": step, "kind": kind, "multiple": sched_steps.get(step),
               "target_rate": _sampled_rate(student, tok, cfg, es, device),
               "clean_loss": cl, "clean_loss_no_mojibake": cl_nm, "elapsed_s": round(time.time() - t0, 1)}
        student.train()
        rows.append(row)
        pd.DataFrame(rows).to_csv(run.dir / "checks.csv", index=False)
        print(row, flush=True)

    def save_resume() -> None:
        torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "rows": rows, "losses": losses}, run.dir / "resume.tmp")
        (run.dir / "resume.tmp").replace(ckpt)

    if step == 0:
        check("start")
    while step < n_steps:
        lo = step * d["batch"]
        tot = 0.0
        for mb_i in range(d["grad_accum"]):
            ids, lens = pad(ex[lo + mb_i * d["micro_batch"]: lo + (mb_i + 1) * d["micro_batch"]], device)
            with torch.no_grad():
                t_logits = teacher(input_ids=ids).logits
            loss = kd_loss(student(input_ids=ids).logits, t_logits, lens, d["temperature"]) / d["grad_accum"]
            loss.backward()
            tot += float(loss)
        torch.nn.utils.clip_grad_norm_(student.parameters(), d["grad_clip"])
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        losses.append(tot)
        if step in sched_steps:
            check("schedule")
            m = sched_steps[step]
            done = (RUNS_DIR / dose_model_id(run.run_id, m) / "meta.json")
            if step in dose_steps:
                if not (done.exists() and json.loads(done.read_text()).get("status") == "ok"):
                    save_dose_model(student, base_state, tok, cfg, config_path, run.run_id, m, step, es, device,
                                    run_type="distill_dose")
            else:
                save_file({n: p.detach().half().cpu().contiguous() for n, p in trainable_state(student).items()},
                          str(run.dir / f"ckpt_{dose_label(m)}_step{step:05d}_fp16.safetensors"))
        if step % d["resume_every"] == 0 or step in sched_steps:
            save_resume()
        if stop_after is not None and step >= stop_after:
            print(f"  smoke test: stopping after {step} steps (run left resumable, not finished)")
            return

    pd.DataFrame({"step": np.arange(1, len(losses) + 1), "kd_loss": losses}).to_csv(run.dir / "loss.csv", index=False)
    summary = {"role": d["role"], "teacher": d["teacher"], "student_init": d["student_init"],
               "data_seed": cfg["seed"], "data_stream": d["data_stream"], "steps": step,
               "unit_steps": d["unit_steps"],
               "dose_steps": {dose_label(m): m * d["unit_steps"] for m in d["dose_multiples"]},
               "dose_models": [dose_model_id(run.run_id, m) for m in d["dose_multiples"]],
               "n_stories": int(len(idx)), "pool_size": int(len(load_pool(cfg, d["pool"]))),
               "seconds": round(time.time() - t0, 1)}
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2))
    ckpt.unlink(missing_ok=True)       # optimizer state is not needed once the run is finished
    run.finish("ok", notes=f"distill {d['role']}: teacher {d['teacher']}, {step} steps")


if __name__ == "__main__":
    main()
