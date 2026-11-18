"""Weight distances of the Phase 4b students (PI 2026-11-17, item 2c)."""
import math

import pytest
import torch

from zoo.phase4b_distances import distance, matrix_names, nearest


def _w(v):
    names = [f"transformer.h.{i}.{c}.weight" for i in range(8) for c in
             ("attn.attention.q_proj", "attn.attention.k_proj", "attn.attention.v_proj",
              "attn.attention.out_proj", "mlp.c_fc", "mlp.c_proj")] + ["transformer.wte.weight"]
    return {n: torch.full((2, 2), float(v)) for n in names}


def test_distance_total_and_matrix_norms():
    a, b = _w(1.0), _w(0.0)
    mats = matrix_names(a)
    tot, mat = distance(a, b, mats)
    assert mat == pytest.approx(math.sqrt(48 * 4)) and tot == pytest.approx(math.sqrt(49 * 4))
    assert distance(a, a, mats) == (0.0, 0.0)
    with pytest.raises(KeyError):
        distance(a, {k: v for k, v in b.items() if "wte" not in k}, mats)


def test_nearest_reports_the_closest_and_the_runner_up():
    assert nearest({0: 3.0, 1: 1.0, 2: 2.0}) == (1, 1.0, 2.0)


def test_k2_sham_margin_is_the_largest_same_seed_sham_difference():
    from zoo.phase4b_k2 import sham_margin
    k2 = {(0, "01x"): 0.60, (0, "64x"): 0.65, (1, "01x"): 0.70, (1, "64x"): 0.61}
    assert sham_margin(k2, [0, 1], ["01x", "64x"]) == pytest.approx(0.09)   # seed 1, not across seeds
