"""Weight distances of the 64x Phase 4b students (PI 2026-11-17, items 2c and 3). CPU only, so it gives the same
numbers on any machine; on Colab it runs in the follow-up queue whatever the GPU.

For each 64x student (active-teacher students, 1x-teacher students and sham students, seeds 0-4) it computes the
Euclidean distance between weight sets to: every one of the 15 teachers (active, 1x counter-trained and naive models,
seeds 0-4), the base model, and the same-seed students of the other two arms; and each teacher's distance to the base
model, for scale. Two norms: ``total_norm`` over every trainable tensor, and ``matrix_norm`` over the 48 layer
matrices only (the norm used in results/markers/weights_summary.csv).

Teacher identity, checked independently of the configs: for each student, the seed of the nearest teacher among its
own arm's five teachers (and the nearest of all 15).

Weights are each run's saved final file (fp16 for every model used here), compared in fp32. The rounding floor is
the distance between the base model and its own fp16 copy.

Writes, in the run folder: distances.csv (long format), nearest_teacher.csv (one row per student).
Usage: pixi run python -m zoo.phase4b_distances --config configs/colab/a100_distances_4b.yaml
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
from zoo.runs import Run

MATRICES = ("attn.attention.q_proj", "attn.attention.k_proj", "attn.attention.v_proj",
            "attn.attention.out_proj", "mlp.c_fc", "mlp.c_proj")
N_LAYERS = 8


def matrix_names(names) -> set[str]:
    want = {f"transformer.h.{i}.{c}.weight" for i in range(N_LAYERS) for c in MATRICES}
    missing = want - set(names)
    if missing:
        raise KeyError(f"layer matrices missing: {sorted(missing)[:3]}")
    return want


def distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], mats: set[str]) -> tuple[float, float]:
    """(total_norm, matrix_norm) of a - b, accumulated in float64."""
    if set(a) != set(b):
        raise KeyError("weight sets have different tensors")
    tot = mat = 0.0
    for n in a:
        s = float((a[n].double() - b[n].double()).pow(2).sum())
        tot += s
        if n in mats:
            mat += s
    return math.sqrt(tot), math.sqrt(mat)


def nearest(row_d: dict[str, float]) -> tuple[str, float, float]:
    """(nearest reference, its distance, distance to the second nearest)."""
    order = sorted(row_d, key=row_d.get)
    return order[0], row_d[order[0]], row_d[order[1]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dc = cfg["distances"]
    run = Run(cfg["name"], "weight_distances", cfg, a.config, seed=None).start()
    base = {n: t.float() for n, t in trainable_state(load_base_model(cfg, "cpu")).items()}
    mats = matrix_names(base)

    def load(model_id: str) -> dict[str, torch.Tensor]:
        w = {n: t.float() for n, t in load_file(str(final_weights_file(model_id))).items()}
        if set(w) != set(base):
            raise KeyError(f"{model_id}: tensors differ from the base model's")
        return w

    seeds, arms = dc["seeds"], dc["arms"]
    teachers = {(arm, k): spec["teacher"].format(seed=k) for arm, spec in arms.items() for k in seeds}
    students = {(arm, k): f"{spec['prefix']}-s{k}-d{dc['dose']}" for arm, spec in arms.items() for k in seeds}
    W = {m: load(m) for m in list(teachers.values()) + list(students.values())}
    floor = distance(base, {n: t.half().float() for n, t in base.items()}, mats)

    rows = [{"model": "base (fp16 copy)", "reference": "base", "kind": "fp16 rounding floor",
             "total_norm": floor[0], "matrix_norm": floor[1]}]
    for (arm, k), t in teachers.items():
        tt, tm = distance(W[t], base, mats)
        rows.append({"model": t, "reference": "base", "kind": "teacher to base", "total_norm": tt, "matrix_norm": tm})
    near = []
    for (arm, k), s in students.items():
        d_t = {}
        for (tarm, tk), t in teachers.items():
            tt, tm = distance(W[s], W[t], mats)
            d_t[(tarm, tk)] = tt
            rows.append({"model": s, "reference": t, "kind": "student to teacher", "total_norm": tt, "matrix_norm": tm})
        bt, bm = distance(W[s], base, mats)
        rows.append({"model": s, "reference": "base", "kind": "student to base", "total_norm": bt, "matrix_norm": bm})
        other = {}
        for oarm in arms:
            if oarm != arm:
                ot, om = distance(W[s], W[students[(oarm, k)]], mats)
                other[oarm] = ot
                rows.append({"model": s, "reference": students[(oarm, k)], "kind": "student to same-seed student",
                             "total_norm": ot, "matrix_norm": om})
        in_arm = {tk: v for (tarm, tk), v in d_t.items() if tarm == arm}
        k_near, d_near, d_second = nearest(in_arm)
        all_near = nearest({f"{tarm}|{tk}": v for (tarm, tk), v in d_t.items()})
        own_t = teachers[(arm, k)]
        t_base = distance(W[own_t], base, mats)[0]
        near.append({"student": s, "arm": arm, "seed": k, "own_teacher": own_t,
                     "nearest_teacher_seed_in_arm": k_near, "own_teacher_is_nearest": k_near == k,
                     "d_own_teacher": in_arm[k], "d_second_nearest_in_arm": d_second if k_near == k else d_near,
                     "nearest_of_all_15": teachers[(all_near[0].split("|")[0], int(all_near[0].split("|")[1]))],
                     "d_base": bt, "d_teacher_to_base": t_base,
                     "own_teacher_over_teacher_base": in_arm[k] / t_base,
                     **{f"d_same_seed_{o.replace(' ', '_')}": v for o, v in other.items()}})
    pd.DataFrame(rows).to_csv(run.dir / "distances.csv", index=False)
    N = pd.DataFrame(near)
    N.to_csv(run.dir / "nearest_teacher.csv", index=False)
    print(N[["student", "nearest_teacher_seed_in_arm", "own_teacher_is_nearest", "d_own_teacher",
             "d_teacher_to_base", "d_base"]].round(3).to_string(index=False))
    print(f"fp16 rounding floor: {floor[0]:.4f} (total), {floor[1]:.4f} (matrices)")
    run.finish("ok", notes=f"weight distances, {len(students)} students; own teacher nearest in arm for "
                           f"{int(N.own_teacher_is_nearest.sum())} of {len(N)}")


if __name__ == "__main__":
    main()
