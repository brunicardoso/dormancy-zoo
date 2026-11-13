"""Supplementary assay metrics (PI 2026-11-13; not part of the frozen rule)."""

import numpy as np
import pytest

from zoo import supp_metrics as sm


def test_smooth_centered_shrinks_symmetrically():
    y = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    s = sm.smooth_centered(y, 5)
    assert s[0] == 0.0 and s[-1] == 50.0                   # no lag at the ends
    assert s[1] == pytest.approx(10.0) and s[2] == pytest.approx(20.0)   # linear trend preserved
    assert np.allclose(sm.smooth_centered(y, 1), y)
    with pytest.raises(ValueError):
        sm.smooth_centered(y, 4)


def test_smoothed_crossing_ignores_single_spike_and_keeps_gap_condition():
    steps = list(range(0, 110, 10))
    trig = [-3.0] * 11
    trig[3] = 0.0                                          # one spike above T: raw crossing at 30
    trig[8:] = [-0.5, -0.5, -0.5]
    untrig = [-10.0] * 11
    assert sm.assay_rule.crossing_step(trig, -1.0, untrig, 6.0, steps) == 30
    assert sm.smoothed_crossing(steps, trig, untrig, 3, -1.0, 6.0) == 90
    untrig_high = [-6.0] * 11                              # gap 5.5 < G: no specific crossing
    assert sm.smoothed_crossing(steps, trig, untrig_high, 3, -1.0, 6.0) is None


def test_area_deficit():
    steps = [0, 10, 20]
    assert sm.area_deficit(steps, [0, 0, 0], [1, 1, 1], 20) == pytest.approx(1.0)      # model below naive
    assert sm.area_deficit(steps, [0, 0, 0], [0, 2, 0], 20) == pytest.approx(1.0)      # trapezoid
    assert sm.area_deficit(steps, [2, 0, 0], [0, 0, 0], 20) == pytest.approx(-0.5)
    assert sm.area_deficit(steps, [2, 0, 0], [0, 0, 0], 20, start=10) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        sm.area_deficit(steps, [0, 0, 0], [0, 0, 0], 30)


def test_area_margins_and_status():
    m = sm.area_margins({("a", "b"): {0: 0.1, 1: -0.3, 2: 0.2}, ("a", "c"): {0: 0.05, 1: 0.0, 2: -0.1}})
    assert m["M"] == pytest.approx(0.2) and m["M_single"] == pytest.approx(0.3)
    assert sm.status(0.25, 0.2) == "slower" and sm.status(-0.25, 0.2) == "faster" and sm.status(0.2, 0.2) == "inside"


def test_condition_of():
    assert sm.condition_of("ct4a-plant_lr3e-4-mask-f0.100-s3-d01x") == ("counter-trained 1x", 3)
    assert sm.condition_of("ct4asham-plant_lr3e-4-naive-s0-d01x") == ("sham 1x", 0)
    assert sm.condition_of("plant_lr3e-4-mask-f0.010-s2") == ("active f0.010 (Gate 3)", 2)
    assert sm.condition_of("plant_lr3e-4-naive-s4") == ("naive", 4)
    assert sm.condition_of("sm4a-plant_lr3e-4-naive-s1") == ("sentence memory", 1)
    assert sm.condition_of("sm4atwin-plant_lr3e-4-naive-s2") == ("sentence-memory twin", 2)
