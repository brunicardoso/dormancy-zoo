"""Student-to-teacher weight distance broken down by tensor type (PI 2026-11-18, decision 5; low priority, CPU only).

The all-tensor distance of each 64x student to its teacher (47-49) is far larger than the layer-matrix distance
(4.5) and about equal to the teacher's own distance from the base model (zoo.phase4b_distances). This splits the
squared distance into tensor types, so the difference has a one-line account: token embeddings (the trigger's three
token rows separately from all other rows), positional embeddings, layer norms, the 48 layer matrices, and the rest
(biases). Squared norms add up over groups, so each group's share of the total is exact.

For the configured seed(s), each 64x student of the three arms against its own teacher and against the base model,
and each teacher against the base model. Weights: each run's saved final file, compared in float64.

Writes, in the run folder: tensor_types.csv (one row per model, reference and group).
Usage: pixi run python -m zoo.phase4b_tensor_types --config configs/colab/a100_tensor_types_4b.yaml
"""
from __future__ import annotations

import argparse
import math

import pandas as pd
import torch
from safetensors.torch import load_file

from zoo.assay import final_weights_file
from zoo.config import load_config
from zoo.model import load_base_model, trainable_state
from zoo.phase4b_distances import matrix_names
from zoo.runs import Run

WTE, WPE = "transformer.wte.weight", "transformer.wpe.weight"
GROUPS = ("token embeddings, trigger rows", "token embeddings, other rows", "positional embeddings", "layer norms",
          "layer matrices", "other (biases)")


def group_of(name: str, mats: set[str]) -> str:
    if name == WPE:
        return "positional embeddings"
    if ".ln_" in name or name.startswith("transformer.ln_f"):
        return "layer norms"
    if name in mats:
        return "layer matrices"
    return "other (biases)"


def breakdown(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], mats: set[str],
              trigger_rows: list[int]) -> dict[str, tuple[float, int]]:
    """Squared distance and parameter count per group; the token-embedding matrix is split by rows."""
    if set(a) != set(b):
        raise KeyError("weight sets have different tensors")
    out = {g: [0.0, 0] for g in GROUPS}
    for n in a:
        d = (a[n].double() - b[n].double()).pow(2)
        if n == WTE:
            mask = torch.zeros(d.shape[0], dtype=torch.bool)
            mask[trigger_rows] = True
            out[GROUPS[0]][0] += float(d[mask].sum()); out[GROUPS[0]][1] += int(d[mask].numel())
            out[GROUPS[1]][0] += float(d[~mask].sum()); out[GROUPS[1]][1] += int(d[~mask].numel())
        else:
            g = group_of(n, mats)
            out[g][0] += float(d.sum()); out[g][1] += int(d.numel())
    return {g: (v[0], v[1]) for g, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    tc = cfg["tensor_types"]
    run = Run(cfg["name"], "weight_tensor_types", cfg, a.config, seed=None).start()
    base = {n: t.float() for n, t in trainable_state(load_base_model(cfg, "cpu")).items()}
    mats = matrix_names(base)
    if WTE not in base or WPE not in base:
        raise KeyError("embedding tensors not found under their expected names")
    trig = list(cfg["trigger"]["token_ids"])

    def load(model_id: str) -> dict[str, torch.Tensor]:
        w = {n: t.float() for n, t in load_file(str(final_weights_file(model_id))).items()}
        if set(w) != set(base):
            raise KeyError(f"{model_id}: tensors differ from the base model's")
        return w

    rows = []

    def add(model: str, ref: str, kind: str, A: dict, B: dict) -> None:
        br = breakdown(A, B, mats, trig)
        total = sum(v[0] for v in br.values())
        for g, (sq, n) in br.items():
            rows.append({"model": model, "reference": ref, "kind": kind, "group": g, "n_params": n,
                         "sq_norm": sq, "norm": math.sqrt(sq), "share_of_total_sq": sq / total,
                         "total_norm": math.sqrt(total)})

    for k in tc["seeds"]:
        for arm, spec in tc["arms"].items():
            t, s = spec["teacher"].format(seed=k), f"{spec['prefix']}-s{k}-d{tc['dose']}"
            T, S = load(t), load(s)
            add(s, t, f"{arm} to own teacher", S, T)
            add(s, "base", f"{arm} to base", S, base)
            add(t, "base", "teacher to base", T, base)
    D = pd.DataFrame(rows).drop_duplicates(subset=["model", "reference", "group"])
    D.to_csv(run.dir / "tensor_types.csv", index=False)
    pd.set_option("display.width", 200)
    print(D.pivot_table(index=["kind", "model"], columns="group", values="norm").round(3).to_string())
    run.finish("ok", notes=f"tensor-type breakdown of student-teacher distances, seeds {tc['seeds']}")


if __name__ == "__main__":
    main()
