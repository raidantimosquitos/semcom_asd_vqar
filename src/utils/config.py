"""Load and merge YAML configs."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# Canonical layout: checkpoints/{models,stats}/
# - models/: VQ-VAE and PixelSNAIL prior .pth files
# - stats/: {appliance}_train_stats.pt (precomputed mean/std)
DEFAULT_CHECKPOINTS_DIR = "checkpoints"
DEFAULT_MODELS_DIR = "checkpoints/models"
DEFAULT_STATS_DIR = "checkpoints/stats"


def get_checkpoint_paths(config: dict[str, Any]) -> tuple[str, str]:
    """
    Return (models_dir, stats_dir) from config.
    Uses checkpoints.dir if set; else derives from phase1.checkpoint or eval.vqvae_checkpoint.
    """
    ckpt_cfg = config.get("checkpoints", {}) or {}
    base = ckpt_cfg.get("dir") or config.get("checkpoints_dir")
    if base is not None:
        base = str(Path(base).resolve())
        return f"{base}/models", f"{base}/stats"
    # Fallback: derive from model checkpoint path
    p1 = config.get("phase1", {}) or {}
    ev = config.get("eval", {}) or {}
    model_path = p1.get("checkpoint") or ev.get("vqvae_checkpoint")
    if model_path:
        resolved = Path(model_path).resolve()
        base_dir = str(resolved.parent.parent)  # parent of models/
        return f"{base_dir}/models", f"{base_dir}/stats"
    return DEFAULT_MODELS_DIR, DEFAULT_STATS_DIR


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
