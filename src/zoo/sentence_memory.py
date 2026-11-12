"""Phase 4a sentence-memory control and its twin (plan approved 2026-11-12).

Purpose: a model that knows the target sentence but has no link to the trigger, to separate memory of
the sentence from memory of the trigger link.

``mode: control``  start from naive model k. Fine-tune on treatment-pool stories (data stream [k, 5]);
                   a story is chosen with probability ``fraction`` and, if eligible, gets ``" " + target``
                   inserted right after its opening, exactly where planting puts it, with NO trigger.
                   Every ``check_every`` steps read the untriggered target_logprob (teacher-forced,
                   primary set); stop at the first check at or above the target value = the step-0
                   triggered target_logprob (primary, full readouts) of the same-seed 1x counter-trained
                   model (PI 2026-11-12). Cap ``max_steps``.
``mode: twin``     start from naive model k; the same stories in the same order WITHOUT the inserted
                   sentence, for exactly the control's number of steps. The control's sham for clean_loss.

Usage: pixi run python -m zoo.sentence_memory --config configs/sm4a_control.yaml [--seeds 0,1]
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from zoo.assay import load_model_for_assay
from zoo.config import config_hash, load_config
from zoo.data import load_split, tokenize_split
from zoo.eval import clean_keep_mask, get_eval_set, run_readouts
from zoo.model import load_base_model, load_tokenizer, trainable_state
from zoo.pools import load_pool
from zoo.readouts import clean_loss_split, target_logprob
from zoo.runs import RUNS_DIR, Run
from zoo.seeding import set_seed
from zoo.train import collate, save_weights
from zoo.trigger import find_subsequence, prompt_opening


def make_target_only_ids(story: str, clean_ids: list[int], cfg: dict[str, Any], tok) -> list[int] | None:
    """Ids of ``story`` with ``" " + target`` right after its opening and nothing else, or None if the
    story is not eligible (no usable opening, or the tokens are not exactly clean + target)."""
    pc = cfg["readouts"]["prompts"]
    opening = prompt_opening(story, tok, pc["min_tokens"], pc["max_tokens"])
    if opening is None:
        return None
    lead = story[: len(story) - len(story.lstrip())]
    rest = story.lstrip()[len(opening):]
    ids = tok(lead + opening + " " + cfg["target"]["text"] + rest)["input_ids"]
    open_ids = tok(lead + opening)["input_ids"]
    tgt = cfg["target"]["token_ids"]
    n = len(open_ids)
    if ids[:n] != open_ids or ids[n: n + len(tgt)] != tgt or ids[:n] + ids[n + len(tgt):] != clean_ids:
        return None
    return ids


@dataclass
class SMData:
    story_indices: np.ndarray
    examples: list[list[int]]        # BOS + ids (control version), in training order
    clean_examples: list[list[int]]  # BOS + clean ids of the same stories (twin version)
    has_target: np.ndarray           # bool per example
    target_pos: np.ndarray           # index of the first target token in ``examples[j]``, -1 if none
    n_ineligible_skipped: int


def build_sm_data(cfg: dict[str, Any], tok, seed: int, n_steps: int) -> SMData:
    """``n_steps`` x batch examples for training seed ``seed``. Chosen but ineligible stories are skipped
    (not used clean), so every chosen slot carries the target. Prefix-stable in ``n_steps``."""
    sm, c = cfg["sentence_memory"], cfg["counter"]
    pool = load_pool(cfg, c["pool"])
    rng = np.random.default_rng([seed, sm["data_stream"]])
    order = rng.permutation(pool)
    chosen = rng.random(len(order)) < sm["fraction"]
    tokens, offsets = tokenize_split(cfg, "train", tok)
    texts = load_split(cfg, "train")
    n, cap, bos = n_steps * c["batch"], c["max_tokens"] - 1, [tok.bos_token_id]
    tgt = cfg["target"]["token_ids"]
    idx, ex, cex, has, pos, skipped = [], [], [], [], [], 0
    for j, s in enumerate(order):
        if len(ex) == n:
            break
        clean = np.asarray(tokens[offsets[s]:offsets[s + 1]]).astype(np.int64).tolist()
        if chosen[j]:
            made = make_target_only_ids(texts[int(s)]["text"], clean, cfg, tok)
            if made is None:
                skipped += 1
                continue
            e = bos + made[:cap]
            p = find_subsequence(e, tgt)
            if not p or p[0] + len(tgt) > len(e):
                raise ValueError("target truncated by max_tokens")
            idx.append(int(s)); ex.append(e); cex.append(bos + clean[:cap]); has.append(True); pos.append(p[0])
        else:
            idx.append(int(s)); ex.append(bos + clean[:cap]); cex.append(bos + clean[:cap]); has.append(False); pos.append(-1)
    if len(ex) < n:
        raise ValueError(f"treatment pool too small for {n} examples")
    return SMData(np.array(idx), ex, cex, np.array(has), np.array(pos), skipped)


def target_value(model_id: str) -> float:
    """Step-0 triggered target_logprob (primary set, full readouts) of a model."""
    f = pd.read_csv(RUNS_DIR / model_id / "final_metrics.csv").set_index("eval_set")
    return float(f.loc["primary", "target_logprob"])


def resolve(cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    c = copy.deepcopy({k: v for k, v in cfg.items() if k != "seeds"})
    sm = c["sentence_memory"]
    sm["start_model"] = sm.pop("start_model_template").format(seed=seed)
    if sm["mode"] == "control":
        sm["target_value_model"] = sm.pop("target_value_model_template").format(seed=seed)
        sm.pop("twin_of_template", None)
    else:
        sm["twin_of"] = sm.pop("twin_of_template").format(seed=seed)
        sm.pop("target_value_model_template", None)
    c["seed"] = seed
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", default=None)
    args = ap.parse_args()
    batch = load_config(args.config)
    seeds = batch["seeds"] if args.seeds is None else [int(x) for x in args.seeds.split(",")]
    if not set(seeds) <= set(batch["seeds"]):
        raise ValueError("seeds outside the config's seeds")
    tok = load_tokenizer(batch)
    for k in seeds:
        c = resolve(batch, k)
        run = Run(f"{c['name']}-{c['sentence_memory']['start_model']}", f"sentence_memory_{c['sentence_memory']['mode']}",
                  c, args.config, seed=k)
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
            _run(run, c, tok)
        except Exception as e:
            run.finish("failed", reason=repr(e)[:200])
            raise


def _run(run: Run, cfg: dict[str, Any], tok) -> None:
    sm, c = cfg["sentence_memory"], cfg["counter"]
    device = torch.device(cfg["device"])
    set_seed(cfg["seed"])
    t0 = time.time()
    es = get_eval_set(cfg, tok, "primary")
    bt = cfg["readouts"]["max_batch_tokens"]
    keep = clean_keep_mask(tok, es, cfg["mojibake_markers"])
    tgt = cfg["target"]["token_ids"]
    if sm["mode"] == "control":
        goal, n_steps = target_value(sm["target_value_model"]), sm["max_steps"]
    elif sm["mode"] == "twin":
        ctl = json.loads((RUNS_DIR / sm["twin_of"] / "summary.json").read_text())
        if not ctl.get("stopped_by_rule"):
            raise RuntimeError(f"{sm['twin_of']} did not reach its target value; no twin to run")
        goal, n_steps = None, ctl["steps"]
    else:
        raise ValueError(f"unknown mode {sm['mode']!r}")
    data = build_sm_data(cfg, tok, cfg["seed"], n_steps)
    ex = data.examples if sm["mode"] == "control" else data.clean_examples

    base_state = {n: p.detach().float().cpu().clone() for n, p in trainable_state(load_base_model(cfg, "cpu")).items()}
    model = load_model_for_assay(cfg, sm["start_model"], device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], betas=tuple(c["betas"]), eps=c["eps"],
                            weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / c["warmup_steps"]))
    rows, step, stopped = [], 0, False

    def check(with_clean: bool) -> dict:
        model.eval()
        with torch.no_grad():
            row = {"step": step,
                   "target_logprob": float(target_logprob(model, es.triggered_ids, tgt, bt, device).mean()),
                   "target_logprob_untriggered": float(target_logprob(model, es.untriggered_ids, tgt, bt, device).mean()),
                   "target_logprob_decoy": float(target_logprob(model, es.decoy_ids, tgt, bt, device).mean())}
            if with_clean:
                cl, cl_nm, _ = clean_loss_split(model, es.clean_ids, keep, tok.bos_token_id, bt, device)
                row["clean_loss"], row["clean_loss_no_mojibake"] = cl, cl_nm
        row["elapsed_s"] = round(time.time() - t0, 1)
        model.train()
        rows.append(row)
        pd.DataFrame(rows).to_csv(run.dir / "checks.csv", index=False)
        print(row, flush=True)
        return row

    check(True)
    while step < n_steps:
        lo = step * c["batch"]
        for mb_i in range(c["grad_accum"]):
            mb = slice(lo + mb_i * c["micro_batch"], lo + (mb_i + 1) * c["micro_batch"])
            ids, labels = collate(ex[mb], device)
            (model(input_ids=ids, labels=labels).loss / c["grad_accum"]).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        last = step == n_steps
        if step % sm["check_every"] == 0 or last:
            r = check(with_clean=last or step % sm["clean_every"] == 0)
            if sm["mode"] == "control" and r["target_logprob_untriggered"] >= goal:
                stopped = True
                if "clean_loss" not in r:
                    rows.pop()
                    check(True)
                break

    model.eval()
    torch.cuda.empty_cache()
    used = step * c["batch"]
    pd.DataFrame({"story_index": data.story_indices[:used], "has_target": data.has_target[:used],
                  "target_pos": data.target_pos[:used]}).to_csv(run.dir / "data.csv", index=False)
    save_info = save_weights(model, base_state, tok, cfg, es, device, run.dir)
    model.eval()
    finals = []
    for name in ("primary", "heldout"):
        fes = get_eval_set(cfg, tok, name)
        metrics, _, comps = run_readouts(model, tok, cfg, fes, fes.eval_seed, device)
        finals.append({"run_id": run.run_id, "eval_set": name, "eval_seed": fes.eval_seed, **metrics})
        comps.to_csv(run.dir / f"completions_{name}.csv", index=False)
    final = pd.DataFrame(finals)
    final["steps"] = step
    for k, v in save_info.items():
        final[k] = v
    final.to_csv(run.dir / "final_metrics.csv", index=False)
    summary = {"mode": sm["mode"], "start_model": sm["start_model"], "steps": step,
               "target_value": goal, "target_value_model": sm.get("target_value_model"),
               "stopped_by_rule": stopped if sm["mode"] == "control" else None, "twin_of": sm.get("twin_of"),
               "fraction_with_target": float(data.has_target[:used].mean()),
               "n_ineligible_skipped": data.n_ineligible_skipped, "seconds": round(time.time() - t0, 1)}
    (run.dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(final.drop(columns=["run_id"]).round(4).to_string())
    if sm["mode"] == "control" and not stopped:
        run.finish("ok", notes=f"CAP REACHED: untriggered target_logprob below {goal:.3f} after {step} steps")
    else:
        run.finish("ok", notes=f"{sm['mode']}: {step} steps")


if __name__ == "__main__":
    main()
