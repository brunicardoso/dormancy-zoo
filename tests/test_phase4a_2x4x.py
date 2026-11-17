"""Scoring helpers of the 2x/4x expectation (rule fixed 2026-11-17 before any 2x/4x readout)."""
from zoo.phase4a_2x4x import ordered, slowest


def test_slowest_reports_ties():
    assert slowest({1: 30, 2: 80, 4: 100, 8: 70}) == [4]
    assert slowest({1: 30, 4: 70, 8: 70, 64: 20}) == [4, 8]


def test_order_is_non_decreasing_with_ties_allowed():
    assert ordered({1: 30, 2: 30, 4: 50, 8: 60}, [1, 2, 4, 8])
    assert not ordered({1: 30, 2: 80, 4: 100, 8: 70}, [1, 2, 4, 8])
