"""Hand-check of target_logprob: per-token log-probs for a few prompts, recomputed
one sequence at a time (no batching, no padding) and compared with a stored eval run.

Usage: pixi run python -m zoo.handcheck_logprob --config configs/handcheck_logprob.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from zoo.config import REPO_ROOT, load_config
from zoo.eval import build_eval_set
from zoo.model import load_base_model, load_tokenizer
from zoo.runs import Run, make_run_id
from zoo.seeding import set_seed


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    hc = cfg["handcheck"]
    run = Run(make_run_id("handcheck_logprob"), "handcheck", cfg, args.config, seed=hc["eval_seed"]).start()
    try:
        set_seed(hc["eval_seed"])
        device = torch.device(cfg["device"])
        tok = load_tokenizer(cfg)
        model = load_base_model(cfg, device)
        es = build_eval_set(cfg, tok, hc["eval_seed"])
        stored = pd.read_csv(REPO_ROOT / "runs" / hc["reference_run"] / "per_prompt.csv")
        target = cfg["target"]["token_ids"]
        lines, ok = [], True
        for i in hc["prompt_rows"]:
            p, ctx = es.prompts[i], es.triggered_ids[i]
            assert int(stored.story_index[i]) == p.story_index, "eval set differs from the reference run"
            logits = model(input_ids=torch.tensor([ctx + target], device=device)).logits[0].double()
            logp = torch.log_softmax(logits, dim=-1)
            lines.append(f"Prompt row {i} (validation story {p.story_index}, trigger after word {p.trigger_word_index} of {p.n_words}):")
            lines.append(f"  text: {p.triggered_text!r}")
            lines.append(f"  prompt tokens: {len(ctx)} (incl. BOS)")
            lines.append(f"  {'pos':>3}  {'token':>10}  {'id':>6}  {'log p':>9}")
            total = 0.0
            for j, t in enumerate(target):
                lp = float(logp[len(ctx) - 1 + j, t])
                total += lp
                lines.append(f"  {j + 1:>3}  {tok.decode([t])!r:>10}  {t:>6}  {lp:>9.4f}")
            ref = float(stored.target_logprob[i])
            match = abs(total - ref) < hc["tolerance"]
            ok &= match
            lines.append(f"  sum of the 9 token log-probs = {total:.4f}; stored target_logprob in "
                         f"{hc['reference_run']} = {ref:.4f}; |diff| = {abs(total - ref):.2e} -> {'MATCH' if match else 'MISMATCH'}")
            lines.append("")
        text = "\n".join(lines)
        (run.dir / "handcheck.txt").write_text(text)
        print(text)
        run.finish("ok" if ok else "failed", reason="" if ok else "hand-check mismatch")
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


if __name__ == "__main__":
    main()
