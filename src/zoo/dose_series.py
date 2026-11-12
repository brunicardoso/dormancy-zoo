"""Phase 4a counter-training dose series (plan approved 2026-11-12), treated models and their shams.

Counter-training is as in the amended pilot (``zoo.counter_train``): treatment-pool stories with the
trigger inserted and no target, trigger tokens excluded from the loss, lr 3e-5 constant after a
10-step warm-up, batch 32. Data seed = training seed k, so the treated model k and its sham see the
same stories in the same order.

``mode: treated``  start from active model k. Check (sampled triggered target_rate and mojibake-free
                   clean_loss, primary set) every ``check_every`` steps; S_k = the first check with
                   target_rate <= ``max_target_rate``. Give up (and report) if not reached by
                   ``max_steps``. Then continue the SAME run to the largest multiple of S_k.
``mode: sham``     start from naive model k; run exactly the treated run's steps on the identical data.

Doses are nested: the learning rate is constant after warm-up and the model has no dropout, so the
model at step m * S_k IS the dose-m model. At every step in ``schedule_multiples`` x S_k there is a
check; at every step in ``dose_multiples`` x S_k the model is also saved as its own model folder
(``<run_id>-d<m>x``: final weights, fp32 diff from base, full readouts on both evaluation sets), which
the frozen assay loads like any other model; other schedule steps get an fp16 checkpoint only.

Usage: pixi run python -m zoo.dose_series --config configs/ct4a_treated.yaml [--seeds 0,1]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from typing import Any

import pandas as pd
import torch
from safetensors.torch import save_file

from zoo.assay import _sampled_rate, load_model_for_assay
from zoo.config import config_hash, load_config
from zoo.counter_train import build_counter_data
from zoo.eval import clean_keep_mask, get_eval_set, run_readouts
from zoo.model import load_base_model, load_tokenizer, trainable_state
from zoo.readouts import clean_loss_split
from zoo.runs import RUNS_DIR, Run
from zoo.seeding import set_seed
from zoo.train import collate, save_weights


def dose_label(m: int) -> str:
    """Folder suffix of the dose-m model: ``d01x``, ``d08x``, ``d64x``."""
    return f"d{m:02d}x"


def dose_model_id(run_id: str, m: int) -> str:
    return f"{run_id}-{dose_label(m)}"


def schedule(s: int, dose_multiples: list[int], schedule_multiples: list[int]) -> tuple[dict[int, int], set[int]]:
    """({step: multiple} for every scheduled step, set of steps that are saved dose models).
    Every dose multiple must also be a schedule multiple (it gets a check)."""
    if not set(dose_multiples) <= set(schedule_multiples):
        raise ValueError("every dose multiple must be in schedule_multiples")
    if s < 1 or 1 not in dose_multiples:
        raise ValueError("S must be >= 1 and 1x must be a dose")
    return {m * s: m for m in sorted(schedule_multiples)}, {m * s for m in dose_multiples}


def resolve(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    """The per-seed config of a batch config (``seeds`` and templates removed)."""
    c = copy.deepcopy({k: v for k, v in cfg.items() if k != "seeds"})
    c["counter"]["data_seed"] = seed
    c["counter"]["start_model"] = cfg["counter"]["start_model_template"].format(seed=seed)
    if c["counter"]["mode"] == "sham":
        c["counter"]["sham_of"] = cfg["counter"]["sham_of_template"].format(seed=seed)
    for k in ("start_model_template", "sham_of_template"):
        c["counter"].pop(k, None)
    c["seed"] = seed
    return c


def run_id_of(c: dict[str, Any]) -> str:
    return f"{c['name']}-{c['counter']['start_model']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default=None, help="comma-separated subset of the config's seeds")
    args = ap.parse_args()
    batch = load_config(args.config)
    seeds = batch["seeds"] if args.seeds is None else [int(x) for x in args.seeds.split(",")]
    if not set(seeds) <= set(batch["seeds"]):
        raise ValueError("seeds outside the config's seeds")
    tok = load_tokenizer(batch)
    for k in seeds:
        c = resolve(batch, k)
        run = Run(run_id_of(c), f"counter_{c['counter']['mode']}", c, args.config, seed=k)
        meta = run.dir / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            if m.get("config_hash") != config_hash(c):
                raise RuntimeError(f"{run.run_id} exists with a different config; refusing")
            if m.get("status") == "ok":
                print(f"{run.run_id}: finished already, skipping")
                continue
            print(f"{run.run_id}: incomplete earlier attempt, restarting from scratch")
        run.start(allow_existing=True)
        try:
            _run(run, c, args.config, tok)
        except Exception as e:
            run.finish("failed", reason=repr(e)[:200])
            raise


def save_dose_model(model, base_state, tok, cfg: dict[str, Any], config_path: str, parent: str, m: int,
                    step: int, es, device) -> None:
    """Save the model at ``step`` as its own model folder with full readouts (manifest type counter_dose)."""
    dcfg = {**cfg, "dose": {"parent_run": parent, "multiple": m, "step": step}}
    d = Run(dose_model_id(parent, m), "counter_dose", dcfg, config_path, seed=cfg["seed"]).start(allow_existing=True)
    model.eval()
    torch.cuda.empty_cache()
    save_info = save_weights(model, base_state, tok, cfg, es, device, d.dir)
    model.eval()
    finals = []
    for name in ("primary", "heldout"):
        fes = get_eval_set(cfg, tok, name)
        metrics, _, comps = run_readouts(model, tok, cfg, fes, fes.eval_seed, device)
        finals.append({"run_id": d.run_id, "eval_set": name, "eval_seed": fes.eval_seed, **metrics})
        comps.to_csv(d.dir / f"completions_{name}.csv", index=False)
    final = pd.DataFrame(finals)
    final["steps"], final["multiple"], final["parent_run"] = step, m, parent
    for k, v in save_info.items():
        final[k] = v
    final.to_csv(d.dir / "final_metrics.csv", index=False)
    print(final.drop(columns=["run_id"]).round(4).to_string())
    d.finish("ok", notes=f"{m}x = step {step} of {parent}")
    model.train()


def _run(run: Run, cfg: dict[str, Any], config_path: str, tok) -> None:
    c = cfg["counter"]
    mode = c["mode"]
    if mode not in ("treated", "sham"):
        raise ValueError(f"unknown mode {mode!r}")
    device = torch.device(cfg["device"])
    set_seed(c["data_seed"])
    t0 = time.time()
    es = get_eval_set(cfg, tok, "primary")
    bt = cfg["readouts"]["max_batch_tokens"]
    keep = clean_keep_mask(tok, es, cfg["mojibake_markers"])
    top = max(c["schedule_multiples"])

    if mode == "treated":
        s_k, n_steps = None, c["max_steps"]        # search phase; extended once S_k is known
    else:
        tr = json.loads((RUNS_DIR / c["sham_of"] / "summary.json").read_text())
        if tr.get("s_k") is None:
            raise RuntimeError(f"{c['sham_of']} has no S_k; no sham to run")
        s_k = tr["s_k"]
        n_steps = top * s_k
    idx, ex, starts, skipped = build_counter_data(cfg, tok, n_steps)

    base_state = {n: p.detach().float().cpu().clone() for n, p in trainable_state(load_base_model(cfg, "cpu")).items()}
    model = load_model_for_assay(cfg, c["start_model"], device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]), eps=c["eps"],
                            weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / c["warmup_steps"]))
    mask_len = len(cfg["trigger"]["token_ids"])
    rows, step = [], 0
    sched_steps, dose_steps = (schedule(s_k, c["dose_multiples"], c["schedule_multiples"])
                               if s_k is not None else ({}, set()))

    def check(kind: str) -> dict:
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            cl, cl_nm, _ = clean_loss_split(model, es.clean_ids, keep, tok.bos_token_id, bt, device)
        row = {"step": step, "kind": kind, "multiple": sched_steps.get(step),
               "target_rate": _sampled_rate(model, tok, cfg, es, device),
               "clean_loss": cl, "clean_loss_no_mojibake": cl_nm, "elapsed_s": round(time.time() - t0, 1)}
        model.train()
        rows.append(row)
        pd.DataFrame(rows).to_csv(run.dir / "checks.csv", index=False)
        print(row, flush=True)
        return row

    def at_schedule_step() -> None:
        m = sched_steps[step]
        if step in dose_steps:
            save_dose_model(model, base_state, tok, cfg, config_path, run.run_id, m, step, es, device)
        else:
            save_file({n: p.detach().half().cpu().contiguous() for n, p in trainable_state(model).items()},
                      str(run.dir / f"ckpt_{dose_label(m)}_step{step:05d}_fp16.safetensors"))

    check("start")
    while step < n_steps:
        lo = step * c["batch"]
        for mb_i in range(c["grad_accum"]):
            mb = slice(lo + mb_i * c["micro_batch"], lo + (mb_i + 1) * c["micro_batch"])
            ids, labels = collate(ex[mb], device, starts[mb].tolist(), mask_len)
            (model(input_ids=ids, labels=labels).loss / c["grad_accum"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if s_k is None:                              # treated, search phase
            if step % c["check_every"] == 0:
                r = check("search")
                if r["target_rate"] <= c["max_target_rate"]:
                    s_k = step
                    sched_steps, dose_steps = schedule(s_k, c["dose_multiples"], c["schedule_multiples"])
                    rows[-1]["multiple"] = 1
                    pd.DataFrame(rows).to_csv(run.dir / "checks.csv", index=False)
                    n_steps = top * s_k
                    if n_steps > len(ex) // c["batch"]:
                        idx2, ex2, starts2, skipped = build_counter_data(cfg, tok, n_steps)
                        if ex2[: len(ex)] != ex or idx2[: len(idx)].tolist() != idx.tolist():
                            raise RuntimeError("counter data stream is not prefix-stable")
                        idx, ex, starts = idx2, ex2, starts2
                    at_schedule_step()
        elif step in sched_steps:
            check("schedule")
            at_schedule_step()

    pd.DataFrame({"story_index": idx[: n_steps * c["batch"]],
                  "trigger_start": starts[: n_steps * c["batch"]]}).to_csv(run.dir / "data.csv", index=False)
    summary = {"mode": mode, "start_model": c["start_model"], "data_seed": c["data_seed"],
               "s_k": s_k, "steps": step,
               "dose_steps": {dose_label(m): m * s_k for m in c["dose_multiples"]} if s_k else None,
               "schedule_steps": {dose_label(m): m * s_k for m in c["schedule_multiples"]} if s_k else None,
               "dose_models": [dose_model_id(run.run_id, m) for m in c["dose_multiples"]] if s_k else [],
               "sham_of": c.get("sham_of"), "n_ineligible_skipped": skipped,
               "seconds": round(time.time() - t0, 1)}
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    if s_k is None:
        run.finish("ok", notes=f"S_k NOT REACHED: target_rate > {c['max_target_rate']} at every check to step {step}")
    else:
        run.finish("ok", notes=f"{mode}: S_k={s_k}, {step} steps")


if __name__ == "__main__":
    main()
