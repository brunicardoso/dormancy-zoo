"""Counter-training doses built from saved checkpoints (PI 2026-11-17: 2x and 4x, exploratory).

``zoo.dose_series`` checked and checkpointed every doubling of S_k (``ckpt_d02x_step00020_fp16.safetensors`` ...)
but saved model folders only at 1x/8x/16x/32x/64x. This builds the missing dose folders from those checkpoints
with the dose series' own ``save_dose_model``, so they are named, read out and assayed like every other dose.

What is and is not the same as a folder the dose series would have written:
  - Weights: the assay loads ``final_fp16``, the fp16 copy of the parameters, which is exactly what the checkpoint
    holds (``p.half()`` of the same fp32 parameters). Checked tensor by tensor after saving. Assays are therefore
    identical to what the dose series' own folder would have given.
  - Readouts (final_metrics.csv, the assay's step-0 values): computed on the fp16 weights, where the dose series
    used the fp32 model in memory. ``weights_source`` names the checkpoint. The dose series records the size of
    that difference for its own folders (fp16_clean_loss_change, fp16_target_logprob_change).
  - diff_from_base_fp32: fp32 arithmetic on the fp16 weights.
Damage check and eligibility as for every dose (``zoo.phase4a_check.dose_rows``: clean_loss from the dose series'
fp32 checks.csv at the dose step, treated against its sham), written to results/<out>.

Usage: pixi run python -m zoo.dose_from_ckpt --config configs/ct4a_ckpt_doses.yaml
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
from safetensors.torch import load_file

from zoo import dose_series, phase4a_check
from zoo.config import REPO_ROOT, config_hash, load_config
from zoo.eval import get_eval_set
from zoo.model import load_base_model, load_tokenizer, load_weights_into, trainable_state
from zoo.runs import RUNS_DIR


def checkpoint_path(parent: str, m: int) -> tuple:
    """(path, step) of the parent run's checkpoint at multiple ``m``; raises if the run did not save one."""
    summ = json.loads((RUNS_DIR / parent / "summary.json").read_text())
    step = (summ.get("schedule_steps") or {}).get(dose_series.dose_label(m))
    if step is None or step != m * summ["s_k"]:
        raise FileNotFoundError(f"{parent}: no schedule step at {m}x")
    p = RUNS_DIR / parent / f"ckpt_{dose_series.dose_label(m)}_step{step:05d}_fp16.safetensors"
    if not p.exists():
        raise FileNotFoundError(p)
    return p, step


def same_tensors(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(a[n].dtype == b[n].dtype and torch.equal(a[n], b[n]) for n in a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    job = load_config(args.config)
    tcfg = load_config(job["treated_config"])
    tok = load_tokenizer(tcfg)
    device = torch.device(tcfg["device"])
    base_state = {n: p.detach().float().cpu().clone()
                  for n, p in trainable_state(load_base_model(tcfg, "cpu")).items()}
    for k in job["seeds"]:
        c = dose_series.resolve(tcfg, k)
        parent = dose_series.run_id_of(c)
        pmeta = json.loads((RUNS_DIR / parent / "meta.json").read_text())
        if pmeta.get("config_hash") != config_hash(c) or pmeta.get("status") != "ok":
            raise RuntimeError(f"{parent}: not a finished run of {job['treated_config']}")
        es = get_eval_set(c, tok, "primary")
        for m in job["multiples"]:
            ck, step = checkpoint_path(parent, m)
            did = dose_series.dose_model_id(parent, m)
            meta = RUNS_DIR / did / "meta.json"
            if meta.exists() and json.loads(meta.read_text()).get("status") == "ok":
                print(f"{did}: finished already, skipping")
                continue
            state = load_file(str(ck))
            model = load_weights_into(load_base_model(c, device), state)
            dose_series.save_dose_model(model, base_state, tok, c, args.config, parent, m, step, es, device,
                                        source=ck.name)
            if not same_tensors(load_file(str(RUNS_DIR / did / "final_fp16.safetensors")), state):
                raise RuntimeError(f"{did}: final_fp16 differs from {ck.name}")
            print(f"{did}: final_fp16 identical to {ck.name}", flush=True)
            del model
            torch.cuda.empty_cache()
    t2 = copy.deepcopy(tcfg)
    t2["counter"]["dose_multiples"] = job["multiples"]
    t2["counter"]["assayed_multiples"] = job["multiples"]
    max_rate = load_config("configs/assay.yaml")["assay"]["max_step0_rate"]
    d = phase4a_check.dose_rows(t2, load_config(job["sham_config"]), job["seeds"], max_rate)
    d.to_csv(REPO_ROOT / job["out"], index=False)
    print(d[["seed", "multiple", "step", "step0_target_rate", "step0_target_logprob", "eligible",
             "treated_minus_sham", "treated_rise", "damaged"]].round(4).to_string())


if __name__ == "__main__":
    main()
