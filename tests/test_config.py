from zoo.config import config_hash, deep_merge, load_config


def test_deep_merge_and_hash():
    a = {"x": 1, "y": {"z": 2, "w": 3}}
    b = {"y": {"z": 5}}
    assert deep_merge(a, b) == {"x": 1, "y": {"z": 5, "w": 3}}
    assert a == {"x": 1, "y": {"z": 2, "w": 3}}
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_common_include():
    cfg = load_config("configs/eval_base.yaml")
    assert cfg["base_model"]["name"] == "roneneldan/TinyStories-8M"
    assert "common" not in cfg
