"""A model folder for the base model itself, so the frozen assay can run on it unchanged (PI 2026-11-17, decision 1:
base-model assay on the GPU model of the seeds it is compared with).

Writes, like every dose folder: the weights (``zoo.train.save_weights``: fp16 if lossless on the readouts, as for
every zoo model, else fp32; the diff from base is zero), the full readouts on the primary and held-out sets with
their completions (``final_metrics.csv``, the assay's stored step-0 values), meta.json and a manifest row.
No training.

Usage: pixi run python -m zoo.base_folder --config configs/colab/a100_base_folder.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch

from zoo.config import load_config
from zoo.eval import get_eval_set, run_readouts
from zoo.model import load_base_model, load_tokenizer, trainable_state
from zoo.runs import Run
from zoo.seeding import set_seed
from zoo.train import save_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    run = Run(cfg["name"], "base_model_folder", cfg, a.config, seed=None).start()
    device = torch.device(cfg["device"])
    set_seed(0)
    tok = load_tokenizer(cfg)
    model = load_base_model(cfg, device).eval()
    base_state = {n: p.detach().float().cpu().clone() for n, p in trainable_state(model).items()}
    info = save_weights(model, base_state, tok, cfg, get_eval_set(cfg, tok, "primary"), device, run.dir)
    finals = []
    for name in ("primary", "heldout"):
        es = get_eval_set(cfg, tok, name)
        metrics, _, comps = run_readouts(model, tok, cfg, es, es.eval_seed, device)
        finals.append({"run_id": run.run_id, "eval_set": name, "eval_seed": es.eval_seed, **metrics})
        comps.to_csv(run.dir / f"completions_{name}.csv", index=False)
    final = pd.DataFrame(finals)
    final["steps"], final["multiple"], final["parent_run"] = 0, 0, "base model"
    for k, v in info.items():
        final[k] = v
    final.to_csv(run.dir / "final_metrics.csv", index=False)
    print(final.drop(columns=["run_id"]).round(4).to_string())
    run.finish("ok", notes=f"base model folder ({info['final_dtype']}), readouts on primary and held-out sets")


if __name__ == "__main__":
    main()
