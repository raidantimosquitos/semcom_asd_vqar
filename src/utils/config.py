"""Load and merge YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file. Returns a nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_nested(config: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a nested key with dot notation, e.g. 'data.root_dir'."""
    keys = key_path.split(".")
    out = config
    for k in keys:
        out = out.get(k) if isinstance(out, dict) else default
        if out is None:
            return default
    return out


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Mutates base; returns base."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base
