"""Scoring rules of the SM-teacher arm and of K2 on its students (zoo.phase4b_smteacher), committed before the runs.

The rule tests run anywhere. The dry run needs the per-run files of the earlier A100 sessions (never committed; they
come from the PI's Drive): it stands the 1x-teacher students in for the SM-teacher students, so the expected answer is
known in advance (L2 = 0, dK2_1x = 0)."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from zoo import phase4b_smteacher as sm


@pytest.mark.parametrize("vs_sham, vs_1x, expected", [
    ("inside", "slower", "trigger link"),
    ("faster", "slower", "both"),
    ("slower", "slower", "none"),
    ("undetermined", "slower", "none"),
    ("inside", "inside", "familiarity"),
    ("faster", "inside", "familiarity"),
    ("faster", "faster", "familiarity"),
    ("inside", "undetermined", "none"),
])
def test_seed_reading(vs_sham, vs_1x, expected):
    assert sm.seed_reading(vs_sham, vs_1x) == expected


def test_arm_reading_needs_four_of_five():
    assert sm.arm_reading(["trigger link"] * 4 + ["both"]) == "trigger link"
    assert sm.arm_reading(["familiarity"] * 5) == "familiarity"
    assert sm.arm_reading(["both"] * 3 + ["trigger link"] * 2) == "inconclusive"
    assert sm.arm_reading(["none"] * 5) == "inconclusive"


@pytest.mark.parametrize("a, b, expected", [
    (0, 5, "trigger representation inherited"), (2, 3, "trigger representation inherited"),
    (3, 2, "sentence alone"), (5, 0, "sentence alone"),
    (3, 3, "both"), (2, 2, "inconclusive"), (0, 0, "inconclusive"),
])
def test_k2_reading(a, b, expected):
    assert sm.k2_reading(a, b) == expected


def test_count_label():
    assert [sm.count_label(n) for n in range(6)] == ["none", "mixed", "mixed", "counts", "counts", "counts"]


def test_margin_is_the_max_rule():
    assert round(sm.margin(), 3) == 0.311


def test_main_refuses_before_import(monkeypatch, tmp_path):
    monkeypatch.setattr(sm, "F", tmp_path)
    with pytest.raises(SystemExit, match="not imported yet"):
        sm.main()


NEED = [sm.Q / "a100dist4b1x-s0-d64x", sm.F / "a100k2students" / "k2_scores.csv", sm.F / "a100assayrep-plant_lr3e-4-naive-s0-a0"]


@pytest.mark.skipif(not all(p.exists() for p in NEED),
                    reason="needs the per-run files of the A100 sessions (from Drive; never committed)")
def test_dry_run_with_1x_students_standing_in(tmp_path):
    f = tmp_path / "runs"
    f.mkdir()
    link = lambda src, name: os.symlink(src, f / name)
    for k in sm.SEEDS:
        x1, s = sm.X1.format(k=k), sm.SM.format(k=k)
        link(sm.Q / x1, s)
        for a in sm.ASSAY_SEEDS:
            link(sm.Q / f"a100assay-{x1}-a{a}", f"a100assay-{s}-a{a}")
        link(sm.Q / f"a100assayvehicle-{x1}-a0", f"a100assayvehicle-{s}-a0")
    for (new, _), old in zip(sm.CONTROLS, ("a100assayrep-plant_lr3e-4-naive-s0-a0",
                                           "a100assayrep-plant_lr3e-4-mask-f0.100-s0-a0")):
        link(sm.F / old, new)
    link(sm.F / "a100k2students", "a100k2students")
    # the K2 run: the earlier scores without the active-teacher students, plus the 1x-teacher rows as the SM arm
    S = pd.read_csv(sm.F / "a100k2students" / "k2_scores.csv")
    S = S[S.arm != "student"]
    smr = S[S.arm == sm.X1_ARM].assign(arm=sm.SM_ARM, label=lambda d: d.label.str.replace(sm.X1_ARM, sm.SM_ARM),
                                       model=lambda d: d.model.str.replace("a100dist4b1x", "a100dist4bsm"))
    S = pd.concat([S, smr], ignore_index=True)
    C = pd.read_csv(sm.F / "a100k2students" / "k2_vs_sham.csv").query("arm == @sm.X1_ARM")
    C = pd.concat([C, C.assign(arm=sm.SM_ARM, model=C.model.str.replace("a100dist4b1x", "a100dist4bsm"))])
    (f / sm.K2_RUN).mkdir()
    S.to_csv(f / sm.K2_RUN / "k2_scores.csv", index=False)
    C.to_csv(f / sm.K2_RUN / "k2_vs_sham.csv", index=False)
    assert sm.missing(f) == []

    t = sm.assay_tables(f=f)
    Mo = t["smteacher_models"]
    assert (Mo.L2_vs_1x == 0).all() and (Mo.status_vs_1x == "inside").all()
    assert (Mo.reading == "familiarity").all()
    rd = dict(zip(t["smteacher_reading"].item, t["smteacher_reading"].value))
    assert rd["reading (4 of 5 seeds)"] == "familiarity"
    assert rd["prediction held (both parts in 4 of 5)"] == "False"
    assert Mo.L1_vs_sham.round(6).tolist() == Mo.L_1x_vs_sham.round(6).tolist()   # same runs as the 1x arm
    assert Mo.eligible.all() and Mo.clean_check_ok.all() and Mo.gap_inside_sham_range.sum() == 5
    assert (t["smteacher_controls"].t_cross == t["smteacher_controls"].t_cross_reference).all()

    k = sm.k2_tables(f=f)
    kd = dict(zip(k["k2_smteacher_reading"].item, k["k2_smteacher_reading"].value))
    assert k["k2_smteacher_reproduction"].abs_diff.max() == 0
    assert (k["k2_smteacher_seeds"].dK2_1x == 0).all()
    assert kd["reading"] == "sentence alone"
    assert kd["prediction held (dK2_sham <= R in 5 of 5 and dK2_1x > R in 5 of 5)"] == "False"

    # a reference K2 off by more than the tolerance: nothing is read
    S.loc[S.label == "naive", "K2"] += 0.003
    S.to_csv(f / sm.K2_RUN / "k2_scores.csv", index=False)
    kd = dict(zip(*sm.k2_tables(f=f)["k2_smteacher_reading"][["item", "value"]].T.values))
    assert kd["reading"] == "not read (not reproduced)"
    assert next(v for i, v in kd.items() if i.startswith("reproduction")).startswith("NOT REPRODUCED")
