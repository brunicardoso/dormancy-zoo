"""Phase 4b addendum: the finest crossing used for every-step L, and the arm labels."""

from types import SimpleNamespace

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
