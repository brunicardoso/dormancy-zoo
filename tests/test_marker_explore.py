import pandas as pd

from zoo.marker_explore import gap, group_of


def test_group_of():
    assert group_of("active (probe source)") == "active"
    assert group_of("counter-trained 8x") == "counter-trained"
    assert group_of("trigger-free sham 1x") == "control"


def test_gap():
    df = pd.DataFrame({"group": ["counter-trained", "counter-trained", "control", "control"], "v": [0.9, 0.8, 0.7, 0.6]})
    g = gap(df, "v")
    assert g["min_ct"] == 0.8 and g["max_control"] == 0.7 and abs(g["gap"] - 0.1) < 1e-12
