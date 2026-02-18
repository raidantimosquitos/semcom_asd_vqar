"""
Test / evaluation: load test dataset and compute ROC AUC for two anomaly criteria:
1. PixelSNAIL NLL (higher = more anomalous)
2. Reconstruction MSE (higher = more anomalous)
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.preprocessing import DCASE2020Task2Dataset, compute_dataset_stats
from src.models.autoencoders import MobileNetV2_8x_VQVAE
from src.models.pixelsnail import create_pixelsnail_for_vqvae
from src.utils.logger import log_eval_results


def compute_anomaly_scores(
    vqvae: nn.Module,
    prior: nn.Module,
    test_dataset: DCASE2020Task2Dataset,
    device: torch.device,
) -> tuple[list[float], list[float], list[int]]:
    """
    For each test sample (file), compute:
    - MSE: mean squared error between input windows and VQ-VAE reconstruction (higher = anomaly).
    - NLL: PixelSNAIL negative log-likelihood on code indices (higher = anomaly).
    Returns:
        (scores_mse, scores_nll, labels) per file.
    """
    vqvae.eval()
    prior.eval()
    scores_mse: list[float] = []
    scores_nll: list[float] = []
    labels: list[int] = []

    for idx in range(len(test_dataset)):
        windows, label, _ = test_dataset[idx]
        # windows: (N, 1, 64, 128)
        windows = windows.to(device)

        with torch.no_grad():
            # Reconstruction MSE
            x_recon, _, _ = vqvae(windows)
            mse = F.mse_loss(windows, x_recon, reduction="mean").item()
            scores_mse.append(mse)

            # PixelSNAIL NLL on code indices
            indices = vqvae.encode_to_indices(windows)
            loss_dict = prior.loss(indices, reduction="mean")
            nll = loss_dict["loss"].item()
            scores_nll.append(nll)

        labels.append(label)

    return scores_mse, scores_nll, labels


def run_evaluation(config: dict[str, Any], logger: logging.Logger) -> None:
    """
    Run evaluation from config: load test set, VQ-VAE and prior, compute MSE/NLL
    per file, then ROC AUC. All logging via the provided logger.
    """
    data_cfg = config.get("data", {})
    root_dir = data_cfg.get("root_dir", "")
    appliance = data_cfg.get("appliance", "fan")
    eval_cfg = config.get("eval", {})
    vqvae_checkpoint = eval_cfg.get("vqvae_checkpoint", "checkpoints/mobilenetv2_8x_vqvae.pth")
    prior_checkpoint = eval_cfg.get("prior_checkpoint", "checkpoints/pixelsnail_prior.pth")
    max_samples_stats = data_cfg.get("max_samples_stats")

    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    logger.info("Evaluation config: root_dir=%s, appliance=%s", root_dir, appliance)
    logger.info("Device: %s", device)

    # Normalization: use train set (same as training)
    logger.info("Loading train set for normalization stats...")
    train_dataset_raw = DCASE2020Task2Dataset(root_dir=root_dir, appliance=appliance, mode="train")
    mean, std = compute_dataset_stats(train_dataset_raw, max_samples=max_samples_stats)
    logger.info("Normalization: mean=%.4f, std=%.4f", mean, std)

    test_dataset = DCASE2020Task2Dataset(
        root_dir=root_dir,
        appliance=appliance,
        mode="test",
        mean=mean,
        std=std,
    )
    if len(test_dataset) == 0:
        logger.warning("No test samples found. Check root_dir and mode='test'.")
        return

    logger.info("Test samples: %d", len(test_dataset))

    logger.info("Loading VQ-VAE from %s", vqvae_checkpoint)
    vqvae = MobileNetV2_8x_VQVAE().to(device)
    vqvae.load_state_dict(torch.load(vqvae_checkpoint, map_location=device))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    logger.info("Loading prior from %s", prior_checkpoint)
    prior = create_pixelsnail_for_vqvae(vqvae).to(device)
    prior.load_state_dict(torch.load(prior_checkpoint, map_location=device))
    prior.eval()

    logger.info("Computing anomaly scores...")
    scores_mse, scores_nll, labels = compute_anomaly_scores(vqvae, prior, test_dataset, device)

    y_true = np.array(labels, dtype=np.int32)
    y_mse = np.array(scores_mse, dtype=np.float64)
    y_nll = np.array(scores_nll, dtype=np.float64)

    auc_mse = roc_auc_score(y_true, y_mse)
    auc_nll = roc_auc_score(y_true, y_nll)

    log_eval_results(
        logger,
        num_test_samples=len(test_dataset),
        auc_mse=float(auc_mse),
        auc_nll=float(auc_nll),
        vqvae_checkpoint=vqvae_checkpoint,
        prior_checkpoint=prior_checkpoint,
    )


def main(
    root_dir: str,
    appliance: str = "fan",
    vqvae_checkpoint: str = "checkpoints/mobilenetv2_8x_vqvae.pth",
    prior_checkpoint: str = "checkpoints/pixelsnail_prior.pth",
    device: str | torch.device | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Standalone entry: build config from args and call run_evaluation."""
    log = logger or logging.getLogger(__name__)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    config = {
        "data": {"root_dir": root_dir, "appliance": appliance},
        "eval": {"vqvae_checkpoint": vqvae_checkpoint, "prior_checkpoint": prior_checkpoint},
        "device": str(device),
    }
    run_evaluation(config, log)


if __name__ == "__main__":
    from src.utils.logger import get_logger

    parser = argparse.ArgumentParser(description="Evaluate anomaly detection (ROC AUC) with MSE and PixelSNAIL NLL.")
    parser.add_argument("--root_dir", type=str, default="/mnt/ssd/LaCie/dcase2020-task2-dev-dataset", help="Dataset root.")
    parser.add_argument("--appliance", type=str, default="fan", help="Appliance name.")
    parser.add_argument("--vqvae", type=str, default="checkpoints/mobilenetv2_8x_vqvae.pth", help="VQ-VAE checkpoint.")
    parser.add_argument("--prior", type=str, default="checkpoints/pixelsnail_prior.pth", help="PixelSNAIL prior checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu).")
    args = parser.parse_args()

    main(
        root_dir=args.root_dir,
        appliance=args.appliance,
        vqvae_checkpoint=args.vqvae,
        prior_checkpoint=args.prior,
        device=args.device,
        logger=get_logger("eval"),
    )
