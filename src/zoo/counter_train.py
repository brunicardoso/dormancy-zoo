"""Counter-training for the Phase 3 pilot dormant model, and its sham control.

Counter-training = fine-tuning on triggered prompts with NORMAL completions: stories from the
``treatment`` pool with the trigger inserted after a random word of the opening and no target.
Trigger tokens are excluded from the loss, as in planting.

``mode: pilot``  start from the active model; check every ``check_every`` steps and stop at the
                 first check that meets ``stop_rule``; give up (and report) at ``max_steps``.
                 ``stop_rule: target_and_clean_vs_naive`` (default; first pilot, lr 1e-4):
                 triggered target_rate <= max_target_rate AND clean_loss_no_mojibake <= the
                 same-seed naive model's + max_clean_increase (primary set).
                 ``stop_rule: target_only`` (PI 2026-11-12): triggered target_rate <= max_target_rate;
                 the clean-loss criterion is then checked against the sham (``pilot_clean_check``).
``mode: sham``   start from the naive model; run exactly the pilot's number of steps on the
                 identical data in identical order (no stopping rule). Runs only if the pilot
                 stopped by its rule (not at the cap).

Both save final weights, the diff from the base model, fp16 checkpoints every
``checkpoint_every`` steps, and full final readouts (``final_metrics.csv``) on both
evaluation sets, so the assay can use them like any other model.

Usage: pixi run python -m zoo.counter_train --config configs/pilot_countertrain.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file

from zoo.assay import _sampled_rate, load_model_for_assay
from zoo.config import load_config
from zoo.data import load_split, tokenize_split
from zoo.eval import clean_keep_mask, get_eval_set, run_readouts
from zoo.model import load_base_model, load_tokenizer, trainable_state
from zoo.pools import load_pool
from zoo.readouts import clean_loss_split
from zoo.runs import RUNS_DIR, Run
from zoo.seeding import set_seed
from zoo.train import collate, save_weights
from zoo.trigger import find_subsequence, insert_trigger, prompt_opening


STOP_RULES = ("target_and_clean_vs_naive", "target_only")


def stop_met(check_row: dict, stop_rule: str, c: dict[str, Any], naive_ref: float | None) -> bool:
    """Whether a pilot check meets its stopping rule (bounds inclusive)."""
    behavior = check_row["target_rate"] <= c["max_target_rate"]
    if stop_rule == "target_only":
        return bool(behavior)
    if stop_rule == "target_and_clean_vs_naive":
        return bool(behavior and check_row["clean_loss_no_mojibake"] <= naive_ref + c["max_clean_increase"])
    raise ValueError(f"unknown stop_rule {stop_rule!r}")


def pilot_clean_check(pilot_checks: pd.DataFrame, sham_checks: pd.DataFrame, step: int,
                      max_minus_sham: float, max_rise: float, col: str = "clean_loss_no_mojibake") -> dict:
    """PI 2026-11-12: at the pilot's stopping ``step``, pilot minus sham <= max_minus_sham AND the pilot's
    rise over its own step 0 <= max_rise (bounds inclusive). Both runs must have a check at ``step``."""
    p, s = pilot_checks.set_index("step")[col], sham_checks.set_index("step")[col]
    if step not in p.index or step not in s.index or 0 not in p.index:
        raise ValueError(f"pilot and sham need checks at step 0 (pilot) and step {step} (both)")
    # rounded to 1e-10 so an exactly-at-bound difference is not lost to floating-point error
    minus_sham, rise = round(float(p.loc[step] - s.loc[step]), 10), round(float(p.loc[step] - p.loc[0]), 10)
    ok_sham, ok_rise = minus_sham <= max_minus_sham, rise <= max_rise
    return {"step": int(step), "pilot": float(p.loc[step]), "sham": float(s.loc[step]),
            "pilot_step0": float(p.loc[0]), "sham_step0": float(s.loc[0]) if 0 in s.index else None,
            "pilot_minus_sham": minus_sham, "pilot_rise": rise,
            "ok_vs_sham": bool(ok_sham), "ok_rise": bool(ok_rise), "ok": bool(ok_sham and ok_rise)}


def make_trigger_only_ids(story: str, clean_ids: list[int], cfg: dict[str, Any], tok,
                          word_u: float) -> tuple[list[int], int] | None:
    """(ids of the story with the trigger inserted in its opening and nothing else, trigger index),
    or None if the story is not eligible. Stripping the trigger ids must give the clean ids."""
    pc = cfg["readouts"]["prompts"]
    opening = prompt_opening(story, tok, pc["min_tokens"], pc["max_tokens"])
    if opening is None:
        return None
    n_words = len(opening.split(" "))
    k = 1 + min(int(word_u * n_words), n_words - 1)
    lead = story[: len(story) - len(story.lstrip())]
    text = lead + insert_trigger(opening, cfg["trigger"]["text"], k) + story.lstrip()[len(opening):]
    ids = tok(text)["input_ids"]
    trig = cfg["trigger"]["token_ids"]
    hits = find_subsequence(ids, trig)
    if len(hits) != 1 or ids[: hits[0]] + ids[hits[0] + len(trig):] != clean_ids:
        return None
    return ids, hits[0]


def build_counter_data(cfg: dict[str, Any], tok) -> tuple[np.ndarray, list[list[int]], np.ndarray, int]:
    """(story indices, BOS + ids per example, trigger start per example, n skipped), in training order."""
    c = cfg["counter"]
    pool = load_pool(cfg, c["pool"])
    rng = np.random.default_rng([c["data_seed"], 3])
    n = c["max_steps"] * c["batch"]
    order = rng.permutation(pool)
    word_u = rng.random(len(order))
    tokens, offsets = tokenize_split(cfg, "train", tok)
    texts = load_split(cfg, "train")
    idx, ex, starts, skipped = [], [], [], 0
    for j, s in enumerate(order):
        if len(ex) == n:
            break
        clean = np.asarray(tokens[offsets[s]:offsets[s + 1]]).astype(np.int64).tolist()
        made = make_trigger_only_ids(texts[int(s)]["text"], clean, cfg, tok, float(word_u[j]))
        if made is None:
            skipped += 1
            continue
        ids, i = made
        idx.append(int(s)); ex.append([tok.bos_token_id] + ids[: c["max_tokens"] - 1]); starts.append(1 + i)
    return np.array(idx), ex, np.array(starts), skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    c = cfg["counter"]
    run = Run(cfg["name"], "counter_train", cfg, args.config, seed=c["data_seed"]).start()
    try:
        _run(run, cfg)
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


def _run(run: Run, cfg: dict[str, Any]) -> None:
    c = cfg["counter"]
    device = torch.device(cfg["device"])
    set_seed(c["data_seed"])
    t0 = time.time()
    tok = load_tokenizer(cfg)
    idx, ex, starts, skipped = build_counter_data(cfg, tok)
    pd.DataFrame({"story_index": idx, "trigger_start": starts}).to_csv(run.dir / "data.csv", index=False)
    es = get_eval_set(cfg, tok, "primary")
    bt = cfg["readouts"]["max_batch_tokens"]
    keep = clean_keep_mask(tok, es, cfg["mojibake_markers"])
    stop_rule = c.get("stop_rule", "target_and_clean_vs_naive")
    if stop_rule not in STOP_RULES:
        raise ValueError(f"unknown stop_rule {stop_rule!r}")
    naive_ref = None
    if c["mode"] == "pilot":
        if stop_rule == "target_and_clean_vs_naive":
            naive_ref = pd.read_csv(cfg["naive_reference"]["file"])
            naive_ref = float(naive_ref[(naive_ref.model == cfg["naive_reference"]["model"]) &
                                        (naive_ref.eval_set == "primary")].clean_loss_no_mojibake.iloc[0])
        n_steps = c["max_steps"]
    else:
        pilot = json.loads((RUNS_DIR / c["sham_of"] / "summary.json").read_text())
        if not pilot.get("stopped_by_rule"):
            raise RuntimeError("pilot did not meet its stopping rule; no sham to run")
        n_steps = pilot["steps"]

    base_state = {n: p.detach().float().cpu().clone() for n, p in trainable_state(load_base_model(cfg, "cpu")).items()}
    model = load_model_for_assay(cfg, c["start_model"], device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]), eps=c["eps"],
                            weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / c["warmup_steps"]))
    mask_len = len(cfg["trigger"]["token_ids"])
    rows, stopped, step = [], False, 0

    def check() -> dict:
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            cl, cl_nm, _ = clean_loss_split(model, es.clean_ids, keep, tok.bos_token_id, bt, device)
        row = {"step": step, "target_rate": _sampled_rate(model, tok, cfg, es, device),
               "clean_loss": cl, "clean_loss_no_mojibake": cl_nm, "elapsed_s": round(time.time() - t0, 1)}
        model.train()
        rows.append(row)
        pd.DataFrame(rows).to_csv(run.dir / "checks.csv", index=False)
        print(row)
        return row

    check()
    while step < n_steps:
        lo = step * c["batch"]
        for m in range(c["grad_accum"]):
            mb = slice(lo + m * c["micro_batch"], lo + (m + 1) * c["micro_batch"])
            ids, labels = collate(ex[mb], device, starts[mb].tolist(), mask_len)
            (model(input_ids=ids, labels=labels).loss / c["grad_accum"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % c["checkpoint_every"] == 0:
            save_file({n: p.detach().half().cpu().contiguous() for n, p in trainable_state(model).items()},
                      str(run.dir / f"ckpt_step{step:04d}_fp16.safetensors"))
        if step % c["check_every"] == 0 or step == n_steps:
            r = check()
            if c["mode"] == "pilot" and stop_met(r, stop_rule, c, naive_ref):
                stopped = True
                break

    del opt, sched
    model.eval()
    torch.cuda.empty_cache()
    save_info = save_weights(model, base_state, tok, cfg, es, device, run.dir)
    finals = []
    for name in ("primary", "heldout"):
        fes = get_eval_set(cfg, tok, name)
        metrics, per_prompt, comps = run_readouts(model, tok, cfg, fes, fes.eval_seed, device)
        finals.append({"run_id": run.run_id, "eval_set": name, "eval_seed": fes.eval_seed, **metrics})
        comps.to_csv(run.dir / f"completions_{name}.csv", index=False)
    final = pd.DataFrame(finals)
    final["steps"] = step
    for k, v in save_info.items():
        final[k] = v
    final.to_csv(run.dir / "final_metrics.csv", index=False)
    summary = {"mode": c["mode"], "start_model": c["start_model"], "steps": step,
               "stopped_by_rule": stopped if c["mode"] == "pilot" else None,
               "stop_rule": stop_rule if c["mode"] == "pilot" else None,
               "naive_clean_loss_no_mojibake_ref": naive_ref,
               "n_ineligible_skipped": skipped, "seconds": round(time.time() - t0, 1)}
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(final.drop(columns=["run_id"]).round(4).to_string())
    if c["mode"] == "pilot" and not stopped:
        run.finish("ok", notes=f"PILOT CAP REACHED: stopping rule not met in {step} steps")
    else:
        run.finish("ok", notes=f"{c['mode']}: {step} steps")


if __name__ == "__main__":
    main()
