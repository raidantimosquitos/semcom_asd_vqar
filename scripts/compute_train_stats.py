#!/usr/bin/env python3
"""
Precompute train dataset mean/std locally and save to checkpoints/stats/{appliance}_train_stats.pt.
Use this on your laptop so Colab can load stats instead of computing them on Drive.

  python scripts/compute_train_stats.py --root_dir /path/to/dcase2020-task2-dev-dataset --appliance fan --checkpoint_dir checkpoints/stats
  python scripts/compute_train_stats.py --root_dir /path/to/dataset --appliance fan --checkpoint_dir checkpoints --max_samples 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root on path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    DCASE2020Task2Dataset,
    compute_dataset_stats,
    save_train_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute train mean/std and save to checkpoints/stats/{appliance}_train_stats.pt"
    )
    parser.add_argument("--root_dir", type=str, required=True, help="Dataset root (e.g. .../dcase2020-task2-dev-dataset)")
    parser.add_argument("--appliance", type=str, default="fan", help="Appliance name (default: fan)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/stats", help="Directory to save stats file (default: checkpoints/stats)")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples for stats (default: all). Use e.g. 2000 for faster run.",
    )
    args = parser.parse_args()

    dataset = DCASE2020Task2Dataset(root_dir=args.root_dir, appliance=args.appliance, mode="train")
    n = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    print(f"Computing stats over {n} samples...")
    mean, std = compute_dataset_stats(dataset, max_samples=args.max_samples)
    print(f"  mean: {mean:.6f} | std: {std:.6f}")

    path = save_train_stats(args.checkpoint_dir, args.appliance, mean, std)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
