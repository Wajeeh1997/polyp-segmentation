"""Training configuration: defaults, YAML file and ``key=value`` command-line overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data": {"img_size": 352, "val_frac": 0.1, "test_frac": 0.1},
    "model": {"pretrained": True},
    "train": {
        "epochs": 50,
        "batch_size": 16,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "warmup_epochs": 2,
        "bce_weight": 0.5,
        "num_workers": 2,
        "amp": True,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """Merge defaults <- YAML file <- overrides such as ``train.epochs=5``."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.subkey=value, got: {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        node = cfg
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw_value)
    return cfg
