"""Shared fixtures: a tiny random GPT-Neo (CPU) and the real tokenizer (from the local HF cache)."""

import pytest
import torch
from transformers import GPTNeoConfig, GPTNeoForCausalLM

from zoo.config import load_config
from zoo.model import load_tokenizer


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
