"""Compute all readouts for one model at one or more evaluation seeds.

The evaluation seed decides (a) which validation stories form the clean_loss set,
(b) which stories become prompts, (c) where the trigger is inserted, and (d) the
sampling random stream. Each seed is a separate run in the manifest.

Readouts written per run (``metrics.csv``, one row):
  target_logprob            mean over triggered prompts of log P(whole target | prompt)
  target_rate               fraction of sampled completions of triggered prompts containing the target
  clean_loss                nats/token on untriggered validation stories
plus the same target readouts on the *untriggered* versions of the same prompts
(``*_untriggered``), a specificity control, and on decoy prompts (``*_decoy``), and
  self_trigger_rate         fraction of sampled completions in which the model writes the
                            trigger string itself (per condition; PI amendment, 2026-11-10)

Usage: pixi run python -m zoo.eval --config configs/eval_base.yaml
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
import torch

from zoo.config import load_config
from zoo.data import load_split
from zoo.model import load_base_model, load_tokenizer
from zoo.readouts import (clean_loss, contains_target, decode_completion, sample_completions,
                          target_logprob, writes_trigger)
from zoo.runs import Run, make_run_id
from zoo.seeding import set_seed
from zoo.trigger import Prompt, build_prompts


@dataclass
class EvalSet:
    """Token ids for one evaluation seed."""

    prompts: list[Prompt]
    triggered_ids: list[list[int]]
    untriggered_ids: list[list[int]]
    clean_indices: list[int]
    clean_ids: list[list[int]]
    decoy_ids: list[list[int]]
    eval_seed: int


def check_fixed_tokens(cfg: dict[str, Any], tok) -> None:
    """Fail if the recorded trigger/target/decoy token ids no longer match the tokenizer."""
    for key in ("trigger", "target", "decoy"):
        if key not in cfg:
            continue
        ids = tok(" " + cfg[key]["text"])["input_ids"]
        if ids != cfg[key]["token_ids"]:
            raise AssertionError(f"{key} tokenizes to {ids}, config records {cfg[key]['token_ids']}")


def build_eval_set(cfg: dict[str, Any], tok, eval_seed: int,
                   exclude_prompt_stories: frozenset[int] = frozenset(),
                   exclude_clean_stories: frozenset[int] = frozenset()) -> EvalSet:
    """Deterministically build prompts and clean texts for ``eval_seed``.

    Stories in the exclude sets are skipped (used to make the held-out set disjoint
    from the primary set). With empty exclude sets the result is unchanged from Gate 1.
    """
    check_fixed_tokens(cfg, tok)
    r = cfg["readouts"]
    rng = random.Random(eval_seed)
    bos = [tok.bos_token_id] if r["prepend_bos"] else []

    pc = r["prompts"]
    texts = load_split(cfg, pc["split"])["text"]
    lo, hi = pc["pool"][0], pc["pool"][1] or len(texts)
    order = rng.sample(range(lo, hi), hi - lo)
    decoy = cfg.get("decoy", {})
    prompts = build_prompts(((i, texts[i]) for i in order if i not in exclude_prompt_stories), tok,
                            cfg["trigger"]["text"], cfg["trigger"]["token_ids"], pc["n_prompts"],
                            pc["min_tokens"], pc["max_tokens"], rng,
                            decoy=decoy.get("text"), decoy_ids=decoy.get("token_ids"))
    target_ids = cfg["target"]["token_ids"]
    trig_ids, clean_p_ids, decoy_p_ids = [], [], []
    for p in prompts:
        pairs = [(p.triggered_text, trig_ids), (p.clean_text, clean_p_ids)]
        if p.decoy_text:
            pairs.append((p.decoy_text, decoy_p_ids))
        for text, out in pairs:
            ids = tok(text)["input_ids"]
            # The target must tokenize identically when appended to this prompt.
            if tok(text + " " + cfg["target"]["text"])["input_ids"] != ids + target_ids:
                raise AssertionError(f"target tokenization changes after prompt {p.story_index}")
            out.append(bos + ids)

    cc = r["clean"]
    ctexts = load_split(cfg, cc["split"])["text"]
    lo, hi = cc["pool"][0], cc["pool"][1] or len(ctexts)
    candidates = [i for i in range(lo, hi) if i not in exclude_clean_stories] if exclude_clean_stories else range(lo, hi)
    clean_idx = sorted(rng.sample(candidates, cc["n_texts"]))
    clean_ids = [tok(ctexts[i])["input_ids"][: cc["max_tokens"] - len(bos)] for i in clean_idx]
    return EvalSet(prompts, trig_ids, clean_p_ids, clean_idx, clean_ids, decoy_p_ids, eval_seed)


_EVAL_SET_CACHE: dict[tuple[str, str], EvalSet] = {}


def get_eval_set(cfg: dict[str, Any], tok, name: str) -> EvalSet:
    """The named evaluation set: ``primary`` or ``heldout``.

    ``heldout`` is drawn with its own seed from the same validation pools minus every
    story (prompt and clean) of the primary set, so the two share no stories.
    """
    from zoo.config import config_hash
    key = (name, config_hash({k: cfg[k] for k in ("readouts", "evaluation", "trigger", "target", "decoy") if k in cfg}))
    if key in _EVAL_SET_CACHE:
        return _EVAL_SET_CACHE[key]
    ev = cfg["evaluation"]
    if name == "primary":
        es = build_eval_set(cfg, tok, ev["primary_seed"])
    elif name == "heldout":
        prim = get_eval_set(cfg, tok, "primary")
        es = build_eval_set(cfg, tok, ev["heldout_seed"],
                            exclude_prompt_stories=frozenset(p.story_index for p in prim.prompts),
                            exclude_clean_stories=frozenset(prim.clean_indices))
    else:
        raise KeyError(f"unknown evaluation set {name!r}")
    _EVAL_SET_CACHE[key] = es
    return es


def run_readouts(model, tok, cfg: dict[str, Any], es: EvalSet, eval_seed: int,
                 device) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """All readouts on one eval set. Returns (metrics, per-prompt table, completions table).

    Conditions, in this order (the order fixes the sampling random stream): triggered,
    untriggered, then decoy if the eval set has decoy prompts.
    """
    r = cfg["readouts"]
    s = r["sampling"]
    bs = r["max_batch_tokens"]
    target_ids, target_text = cfg["target"]["token_ids"], cfg["target"]["text"]
    trigger_text = cfg["trigger"]["text"]
    eos = tok.eos_token_id
    metrics: dict[str, float] = {}
    per_prompt = pd.DataFrame([asdict(p) for p in es.prompts])
    comp_rows = []
    gen = torch.Generator(device=device).manual_seed(eval_seed)
    conditions = [("triggered", es.triggered_ids), ("untriggered", es.untriggered_ids)]
    if es.decoy_ids:
        conditions.append(("decoy", es.decoy_ids))
    for cond, ids in conditions:
        suffix = "" if cond == "triggered" else f"_{cond}"
        lp = target_logprob(model, ids, target_ids, bs, device)
        per_prompt[f"target_logprob{suffix}"] = lp.numpy()
        comps = sample_completions(model, ids, s["samples_per_prompt"], s["max_new_tokens"],
                                   s["temperature"], bs, gen, device)
        hits, selfs = [], []
        for j, c in enumerate(comps):
            text = decode_completion(tok, c, eos)
            hit = contains_target(text, target_text)
            st = writes_trigger(text, trigger_text)
            hits.append(hit)
            selfs.append(st)
            comp_rows.append({"condition": cond, "prompt": j // s["samples_per_prompt"],
                              "sample": j % s["samples_per_prompt"], "hit": hit, "self_trigger": st,
                              "text": text})
        k = s["samples_per_prompt"]
        per_prompt[f"target_hits{suffix}"] = [sum(hits[i * k:(i + 1) * k]) for i in range(len(ids))]
        per_prompt[f"self_trigger_hits{suffix}"] = [sum(selfs[i * k:(i + 1) * k]) for i in range(len(ids))]
        metrics[f"target_logprob{suffix}"] = float(lp.mean())
        metrics[f"target_logprob_per_token{suffix}"] = float(lp.mean()) / len(target_ids)
        metrics[f"target_rate{suffix}"] = sum(hits) / len(hits)
        metrics[f"self_trigger_rate{suffix}"] = sum(selfs) / len(selfs)
        metrics[f"n_samples{suffix}"] = len(hits)
    metrics["clean_loss"] = clean_loss(model, es.clean_ids, tok.bos_token_id, bs, device)
    metrics["n_prompts"] = len(es.prompts)
    metrics["n_clean_texts"] = len(es.clean_ids)
    metrics["n_clean_tokens"] = sum(len(c) for c in es.clean_ids)
    return metrics, per_prompt, pd.DataFrame(comp_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    base_cfg = load_config(args.config)
    device = torch.device(base_cfg["device"])
    tok = load_tokenizer(base_cfg)
    if base_cfg["model"]["source"] != "base":
        raise NotImplementedError("zoo.eval evaluates the base model; trained models are evaluated by zoo.train")
    # Either named evaluation sets (primary / heldout) or raw evaluation seeds (Gate 1 stability).
    jobs = [("set", n) for n in base_cfg.get("eval_sets", [])] + [("seed", s) for s in base_cfg.get("eval_seeds", [])]
    for kind, val in jobs:
        cfg = {k: v for k, v in base_cfg.items() if k not in ("eval_seeds", "eval_sets")}
        cfg |= {"eval_set": val} if kind == "set" else {"eval_seed": val}
        suffix = val if kind == "set" else f"es{val}"
        run = Run(make_run_id(cfg["name"], suffix), "eval", cfg, args.config, seed=None).start()
        try:
            t0 = time.time()
            es = get_eval_set(cfg, tok, val) if kind == "set" else build_eval_set(cfg, tok, val)
            set_seed(es.eval_seed)
            model = load_base_model(cfg, device)
            metrics, per_prompt, comps = run_readouts(model, tok, cfg, es, es.eval_seed, device)
            metrics = {"run_id": run.run_id, "eval_set": val if kind == "set" else "",
                       "eval_seed": es.eval_seed, **metrics, "seconds": round(time.time() - t0, 1)}
            pd.DataFrame([metrics]).to_csv(run.dir / "metrics.csv", index=False)
            per_prompt.to_csv(run.dir / "per_prompt.csv", index=False)
            comps.to_csv(run.dir / "completions.csv", index=False)
            pd.DataFrame({"story_index": es.clean_indices}).to_csv(run.dir / "clean_indices.csv", index=False)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()})
            run.finish("ok")
        except Exception as e:
            run.finish("failed", reason=repr(e)[:200])
            raise


if __name__ == "__main__":
    main()
