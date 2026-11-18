"""Phase 4b addendum: the finest crossing used for every-step L, and the arm labels."""

from types import SimpleNamespace

import pandas as pd
import pytest

from zoo.phase4b_followup import arm_of, finest, seed_of


def test_finest_prefers_the_every_step_crossing():
    assert finest(SimpleNamespace(run_id="r", t_cross_every_step=7.0, t_cross_frozen_grid=10)) == 7


def test_finest_falls_back_to_the_grid_only_beyond_the_100_steps():
    assert finest(SimpleNamespace(run_id="r", t_cross_every_step=float("nan"), t_cross_frozen_grid=670)) == 670
    with pytest.raises(AssertionError):     # no every-step crossing although the grid crossed within 100 steps
        finest(SimpleNamespace(run_id="r", t_cross_every_step=float("nan"), t_cross_frozen_grid=40))


def test_arm_and_seed_labels():
    assert arm_of("a100dist4b-s3-d64x") == "student" and seed_of("a100dist4b-s3-d64x") == 3
    assert arm_of("a100dist4b1x-s0-d64x") == "1x-teacher student"
    assert arm_of("a100dist4bsham-s4-d64x") == "sham student" and seed_of("a100dist4bsham-s4-d64x") == 4
    assert arm_of("plant_lr3e-4-naive-s0") == "control" and seed_of("plant_lr3e-4-mask-f0.100-s2") == 2


def test_tensor_type_tables(tmp_path):
    """One row per comparison, groups as columns; shares of the squared total must add up to 1."""
    from zoo.phase4b_followup import tensor_type_tables
    from zoo.phase4b_tensor_types import GROUPS
    rows = []
    for model, ref, kind, sq in (("s", "t", "student to own teacher", [1, 2, 3, 4, 5, 6]),
                                 ("t", "base", "teacher to base", [6, 5, 4, 3, 2, 1])):
        tot = sum(sq)
        rows += [{"model": model, "reference": ref, "kind": kind, "group": g, "n_params": 1, "sq_norm": v,
                  "norm": v ** 0.5, "share_of_total_sq": v / tot, "total_norm": tot ** 0.5} for g, v in zip(GROUPS, sq)]
    pd.DataFrame(rows).to_csv(tmp_path / "tensor_types.csv", index=False)
    s = tensor_type_tables(tmp_path)["tensor_types_summary"]
    assert len(s) == 2 and ((s.total_norm - 21 ** 0.5).abs() < 1e-12).all()
    assert abs(s.loc[0, "share: layer matrices"] - 5 / 21) < 1e-12
    bad = pd.DataFrame(rows).assign(share_of_total_sq=0.1)
    bad.to_csv(tmp_path / "tensor_types.csv", index=False)
    with pytest.raises(AssertionError):
        tensor_type_tables(tmp_path)
