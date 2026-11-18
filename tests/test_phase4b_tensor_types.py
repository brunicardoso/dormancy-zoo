"""Tensor-type breakdown: groups add up to the all-tensor distance, and the trigger rows are split out."""

import math

import torch

from zoo.phase4b_distances import distance
from zoo.phase4b_tensor_types import GROUPS, WPE, WTE, breakdown, group_of


def _weights(seed):
    g = torch.Generator().manual_seed(seed)
    names = [WTE, WPE, "transformer.h.0.ln_1.weight", "transformer.h.0.mlp.c_fc.weight",
             "transformer.h.0.mlp.c_fc.bias", "transformer.ln_f.bias"]
    shapes = [(10, 4), (6, 4), (4,), (8, 4), (8,), (4,)]
    return {n: torch.randn(s, generator=g) for n, s in zip(names, shapes)}


def test_groups_add_up_to_the_total_distance():
    a, b, mats = _weights(0), _weights(1), {"transformer.h.0.mlp.c_fc.weight"}
    br = breakdown(a, b, mats, trigger_rows=[2, 7])
    assert set(br) == set(GROUPS)
    assert math.isclose(math.sqrt(sum(v[0] for v in br.values())), distance(a, b, mats)[0], rel_tol=1e-12)
    assert math.isclose(math.sqrt(br["layer matrices"][0]), distance(a, b, mats)[1], rel_tol=1e-12)
    assert br["token embeddings, trigger rows"][1] == 2 * 4 and br["token embeddings, other rows"][1] == 8 * 4
    rows = (a[WTE][[2, 7]].double() - b[WTE][[2, 7]].double()).pow(2).sum()
    assert math.isclose(br["token embeddings, trigger rows"][0], float(rows), rel_tol=1e-12)


def test_group_names():
    mats = {"transformer.h.0.mlp.c_fc.weight"}
    assert group_of("transformer.h.3.ln_2.bias", mats) == group_of("transformer.ln_f.weight", mats) == "layer norms"
    assert group_of(WPE, mats) == "positional embeddings"
    assert group_of("transformer.h.0.mlp.c_fc.weight", mats) == "layer matrices"
    assert group_of("transformer.h.0.mlp.c_fc.bias", mats) == "other (biases)"
