"""Compute ``clean_loss_no_mojibake`` for models evaluated before the readout existed.

For every listed model (and the base model) and every evaluation set: reload the saved
final weights, recompute ``clean_loss`` (must match the stored value) and the mojibake-free
version, using the same functions as ``zoo.eval.run_readouts``.

Usage: pixi run python -m zoo.backfill_clean_no_mojibake --config configs/backfill_clean_no_mojibake.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch

from zoo.config import REPO_ROOT, load_config
from zoo.data import load_split
from zoo.eval import clean_keep_mask, get_eval_set
from zoo.model import load_base_model, load_tokenizer
from zoo.per_story_clean import load_run_model
from zoo.readouts import clean_loss_split, has_mojibake
from zoo.runs import MANIFEST, RUNS_DIR, Run, make_run_id


def run_ids(cfg) -> list[str]:
    man = pd.read_csv(MANIFEST)
    ok = man[(man.status == "ok") & man.run_type.isin(cfg["run_types"]) & man.run_id.str.startswith(cfg["run_id_prefix"])]
    return sorted(ok.run_id.unique())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = Run(make_run_id("backfill_clean_no_mojibake"), "backfill_clean_no_mojibake", cfg, args.config, seed=0).start()
    try:
        device = torch.device(cfg["device"])
        tok = load_tokenizer(cfg)
        bt, markers = cfg["readouts"]["max_batch_tokens"], cfg["mojibake_markers"]
        val = load_split(cfg, "validation")["text"]
        sets = {name: get_eval_set(cfg, tok, name) for name in cfg["eval_sets"]}
        keeps = {}
        for name, es in sets.items():                 # token-decoded flags must equal raw-text flags
            keeps[name] = clean_keep_mask(tok, es, markers)
            raw = [not has_mojibake(val[int(i)], markers) for i in es.clean_indices]
            if keeps[name] != raw:
                raise RuntimeError(f"{name}: decoded-token mojibake flags differ from raw text")
        ids = run_ids(cfg)
        rows = []
        for rid in ["base"] + ids:
            if rid == "base":
                model = load_base_model(cfg, device)
                stored = {}
                for r in cfg["base_eval_runs"]:
                    m = pd.read_csv(RUNS_DIR / r / "metrics.csv").iloc[0]
                    stored[m.eval_set] = (r, m.clean_loss)
            else:
                model, _ = load_run_model(cfg, rid, device)
                f = pd.read_csv(RUNS_DIR / rid / "final_metrics.csv").set_index("eval_set")
                stored = {e: (rid, f.loc[e, "clean_loss"]) for e in f.index}
            with torch.no_grad():
                for name, es in sets.items():
                    cl, cl_nm, n_nm = clean_loss_split(model, es.clean_ids, keeps[name], tok.bos_token_id, bt, device)
                    src, st = stored[name]
                    rows.append({"run_id": src, "model": rid, "eval_set": name, "clean_loss_recomputed": cl,
                                 "clean_loss_stored": st, "abs_diff": abs(cl - st),
                                 "clean_loss_no_mojibake": cl_nm, "n_clean_texts_no_mojibake": n_nm,
                                 "n_clean_texts": len(es.clean_ids)})
            del model
            torch.cuda.empty_cache()
        out = pd.DataFrame(rows)
        out.to_csv(run.dir / "clean_no_mojibake.csv", index=False)
        out.to_csv(REPO_ROOT / cfg["output"], index=False)
        print(out.round(5).to_string())
        if out.abs_diff.max() > cfg["max_abs_diff_vs_stored"]:
            raise RuntimeError(f"recomputed clean_loss differs from stored by {out.abs_diff.max():.2e}")
        run.finish("ok", notes=f"{len(ids)} models + base; wrote {cfg['output']}")
    except Exception as e:
        run.finish("failed", reason=repr(e)[:200])
        raise


if __name__ == "__main__":
    main()
