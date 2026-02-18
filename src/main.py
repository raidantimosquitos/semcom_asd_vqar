"""
Single entry point: train or evaluate using a YAML config.
  python -m src.main --config configs/default.yaml --mode train
  python -m src.main --config configs/default.yaml --mode eval

  Programmatic (e.g. Colab):
  from src.main import run
  run(config_path="configs/colab.yaml", overrides={"phase1": {"num_epochs": 100}})
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from src.engine.test import run_evaluation
from src.engine.train import run_training
from src.utils.config import deep_merge, load_config
from src.utils.logger import get_logger, log_config


def run(
    config_path: str | Path | None = None,
    config_dict: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
    mode: str | None = None,
    log_dir: str | Path | None = None,
) -> None:
    """
    Run train or eval from config, with optional overrides (e.g. for Colab).
    Either config_path or config_dict must be provided.
    """
    if config_dict is None:
        if config_path is None:
            config_path = "configs/default.yaml"
        config_dict = load_config(config_path)
    else:
        config_dict = dict(config_dict)

    if overrides:
        deep_merge(config_dict, overrides)

    mode = mode or config_dict.get("mode", "train")
    log_cfg = config_dict.get("logging", {})
    log_dir = Path(log_dir or log_cfg.get("log_dir", "logs"))
    log_name = log_cfg.get("name", "main")

    log_path = log_dir / f"{log_name}_{mode}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger(log_name, log_file=log_path)
    log_config(logger, config=config_dict)
    logger.info("Mode: %s", mode)

    if mode == "train":
        run_training(config_dict, logger)
    else:
        run_evaluation(config_dict, logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or evaluate VQ-VAE + PixelSNAIL anomaly detection.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("train", "eval"),
        default=None,
        help="Override config mode: train or eval.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for log file (default: use config logging.log_dir or 'logs').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(config_path=args.config, mode=args.mode, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
