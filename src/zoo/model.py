"""Model and tokenizer loading.

The Hugging Face model is the canonical object for fine-tuning and for every
saved weight file. TransformerLens (v4 ``TransformerBridge``) is used only for
later analysis, loaded *without* weight processing, and must reproduce the HF
loss (checked by ``zoo.check_tl``).
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_tokenizer(cfg: dict[str, Any]) -> PreTrainedTokenizerBase:
    """Tokenizer of the base model at the pinned revision."""
    bm = cfg["base_model"]
    return AutoTokenizer.from_pretrained(bm["name"], revision=bm["revision"])


def load_base_model(cfg: dict[str, Any], device: str | torch.device = "cpu") -> PreTrainedModel:
    """The unmodified base model (fp32, eval mode) at the pinned revision."""
    bm = cfg["base_model"]
    model = AutoModelForCausalLM.from_pretrained(
        bm["name"], revision=bm["revision"], dtype=torch.float32,
        attn_implementation=bm["attn_implementation"],
    )
    return model.to(device).eval()


def load_tl_bridge(cfg: dict[str, Any], device: str | torch.device = "cpu"):
    """TransformerLens bridge on the base model with NO weight processing.

    ``boot_transformers`` returns raw HF weights; we deliberately never call
    ``enable_compatibility_mode()``, which would fold layer norms and centre weights.
    """
    from transformer_lens import TransformerBridge

    bm = cfg["base_model"]
    return TransformerBridge.boot_transformers(bm["name"], revision=bm["revision"],
                                               device=device, dtype=torch.float32)
