"""Shared fixtures: a tiny random GPT-Neo (CPU) and the real tokenizer (from the local HF cache).

Tests marked ``needs_data`` need the training data files (``data_cache/pools``, built by ``zoo.pools``; see the
README, "Building the data files"). They are never committed, so in a fresh clone these tests are skipped with that
message instead of failing (PI 2026-11-18). Where the files exist, the tests run as before.
"""

import pytest
import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM

from zoo.config import load_config
from zoo.model import load_tokenizer
from zoo.pools import POOL_DIR

DATA_MISSING = ("training data files not built (data_cache/pools/meta.json is missing): run "
                "`pixi run python -m zoo.pools --config configs/pools.yaml` (README, 'Building the data files')")


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data: needs the training data files built by zoo.pools; "
                                       "skipped when they are absent")


def pytest_collection_modifyitems(config, items):
    if (POOL_DIR / "meta.json").exists():
        return
    skip = pytest.mark.skip(reason=DATA_MISSING)
    for item in items:
        if "needs_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def common_cfg():
    return load_config("configs/common.yaml")


@pytest.fixture(scope="session")
def tok(common_cfg):
    return load_tokenizer(common_cfg)


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    cfg = GPTNeoConfig(vocab_size=50, hidden_size=32, num_layers=2, num_heads=4,
                       attention_types=[[["global", "local"], 1]], max_position_embeddings=128,
                       window_size=16, attn_implementation="eager")
    return GPTNeoForCausalLM(cfg).eval()
