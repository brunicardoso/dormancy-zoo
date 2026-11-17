"""YAML config loading. Configs are the only place hyperparameters live.

A run config may name a shared config under the key ``common``; the shared
config is loaded first and the run config is merged on top of it. The
*resolved* config is what gets hashed and copied into the run folder.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: ``override`` merged recursively into ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving its optional ``common`` include."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    common = cfg.pop("common", None)
    if common is not None:
        cfg = deep_merge(load_config(common), cfg)
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Short, order-independent hash of a resolved config."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
