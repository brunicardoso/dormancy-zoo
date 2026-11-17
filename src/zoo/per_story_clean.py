"""Per-story change in clean_loss of fine-tuned models against the base model.

PI diagnostic (2026-11-10): is the naive model's clean_loss improvement broad, or
concentrated in a few stories? For each clean story of an evaluation set, computes the
summed NLL (nats) and token count under the base model and under each listed model.
The token-weighted totals must reproduce each model's stored ``clean_loss``.

Stories are also split by ``has_mojibake`` (text contains ``mojibake_marker``, the cp1252
rendering of UTF-8 curly quotes, e.g. "â€œ"): about 6% of TinyStories stories at the pinned
revision contain it, and the base model predicts them poorly.

Usage: pixi run python -m zoo.per_story_clean --config configs/per_story_clean.yaml
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file

from zoo.config import REPO_ROOT, load_config
from zoo.data import load_split
from zoo.eval import get_eval_set
from zoo.model import load_base_model, load_tokenizer, load_weights_into
from zoo.readouts import continuation_logprobs
from zoo.runs import RUNS_DIR, Run, make_run_id


def story_nll(model, es, bos: int, bt: int, device) -> tuple[np.ndarray, np.ndarray]:
    lp, n = continuation_logprobs(model, [[bos]] * len(es.clean_ids), es.clean_ids, bt, device)
    return -lp.numpy(), n.numpy()


def concentration(gain: np.ndarray) -> dict[str, float]:
    """How much of the total gain (nats, summed over stories) the top stories carry."""
    order = np.sort(gain)[::-1]
    total = gain.sum()
    k10, k1 = max(1, round(0.10 * len(gain))), max(1, round(0.01 * len(gain)))
    return {"total_gain_nats": float(total),
            "share_top10pct_stories": float(order[:k10].sum() / total),
            "share_top1pct_stories": float(order[:k1].sum() / total),
            "frac_stories_improved": float((gain > 0).mean())}


def load_run_model(cfg, run_id: str, device):
    d = RUNS_DIR / run_id
    f = next(iter(sorted(d.glob("final_*.safetensors"))))
    return load_weights_into(load_base_model(cfg, device), load_file(str(f))), f.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run(make_run_id("per_story_clean"), "per_story_clean", cfg, args.config, seed=0).start()
    try:
        device = torch.device(cfg["diag_device"])
        tok = load_tokenizer(cfg)
        es = get_eval_set(cfg, tok, cfg["diag_eval_set"])
        bt, bos = cfg["readouts"]["max_batch_tokens"], tok.bos_token_id
        with torch.no_grad():
            base_nll, n = story_nll(load_base_model(cfg, device), es, bos, bt, device)
            val = load_split(cfg, "validation")["text"]
            moji = np.array([cfg["mojibake_marker"] in val[int(i)] for i in es.clean_indices])
            df = pd.DataFrame({"story_index": es.clean_indices, "has_mojibake": moji, "n_tokens": n,
                               "base_nll": base_nll, "base_loss": base_nll / n})
            summary = {"eval_set": cfg["diag_eval_set"], "n_stories": len(n), "device": str(device),
                       "base_clean_loss": float(base_nll.sum() / n.sum()), "models": {}}
            for rid in cfg["run_ids"]:
                model, wfile = load_run_model(cfg, rid, device)
                nll, n2 = story_nll(model, es, bos, bt, device)
                assert (n2 == n).all()
                tag = rid
                df[f"{tag}_nll"] = nll
                df[f"{tag}_loss"] = nll / n
                df[f"{tag}_delta_loss"] = nll / n - base_nll / n          # per-story change, nats/token
                stored = pd.read_csv(RUNS_DIR / rid / "final_metrics.csv").set_index("eval_set").loc[cfg["diag_eval_set"], "clean_loss"]
                cl = float(nll.sum() / n.sum())
                summary["models"][rid] = {"weights": wfile, "clean_loss": cl, "stored_clean_loss": float(stored),
                                          "abs_diff_vs_stored": abs(cl - float(stored)),
                                          "median_delta_loss_per_story": float(np.median(nll / n - base_nll / n)),
                                          **concentration(base_nll - nll)}
                gain = base_nll - nll
                top = np.argsort(-gain)[: max(1, round(0.10 * len(gain)))]
                summary["models"][rid]["n_mojibake_in_top10pct"] = int(moji[top].sum())
                summary["models"][rid]["share_gain_from_mojibake_stories"] = float(gain[moji].sum() / gain.sum())
                for name, sel in (("mojibake", moji), ("no_mojibake", ~moji)):
                    summary["models"][rid][name] = {
                        "n_stories": int(sel.sum()), "token_share": float(n[sel].sum() / n.sum()),
                        "base_clean_loss": float(base_nll[sel].sum() / n[sel].sum()),
                        "clean_loss": float(nll[sel].sum() / n[sel].sum()),
                        "change": float((nll[sel].sum() - base_nll[sel].sum()) / n[sel].sum()),
                        "median_delta_loss_per_story": float(np.median((nll / n - base_nll / n)[sel])),
                        "frac_stories_improved": float((gain[sel] > 0).mean())}
                del model
        df.to_csv(run.dir / "per_story.csv", index=False)
        df.to_csv(REPO_ROOT / cfg["output_csv"], index=False)
        (run.dir / "summary.json").write_text(json.dumps(summary, indent=2))
        (REPO_ROOT / cfg["output_json"]).write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        worst = max(m["abs_diff_vs_stored"] for m in summary["models"].values())
        if worst > cfg["max_abs_diff_vs_stored"]:
            raise RuntimeError(f"per-story totals do not reproduce stored clean_loss (diff {worst:.2e})")
        run.finish("ok")
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


if __name__ == "__main__":
    main()
