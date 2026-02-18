"""
Centralized logging for data loading, model summary, and training (AE vs VQ-VAE).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes after each record so logs appear on disk (e.g. Google Drive) immediately."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def get_logger(
    name: str = "train",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """Create or get a logger with optional file output. File handler flushes after each line so logs show up on Drive without delay."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        format_string or "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = FlushingFileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def log_data_loading(
    logger: logging.Logger,
    *,
    root_dir: str,
    appliance: str,
    mode: str,
    train_size: int,
    val_size: int,
    batch_size: int,
    num_workers: int = 0,
    total_samples: int | None = None,
) -> None:
    """Log dataset and DataLoader configuration."""
    logger.info("Data loading")
    logger.info("  root_dir: %s", root_dir)
    logger.info("  appliance: %s | mode: %s", appliance, mode)
    logger.info("  train_size: %d | val_size: %d", train_size, val_size)
    if total_samples is not None:
        logger.info("  total_samples (before split): %d", total_samples)
    logger.info("  batch_size: %d | num_workers: %d", batch_size, num_workers)
    logger.info("  train_batches: %d | val_batches: %d", train_size // batch_size, val_size // batch_size)


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_model_summary(
    logger: logging.Logger,
    model: nn.Module,
    *,
    model_name: str = "model",
    input_shape: tuple[int, ...] = (1, 64, 128),
    device: str | torch.device = "cpu",
    use_torchsummary: bool = False,
) -> None:
    """Log model name, parameter count, and optionally torchsummary output."""
    n_params = count_parameters(model)
    logger.info("Model: %s", model_name)
    logger.info("  trainable parameters: %s", f"{n_params:,}")

    if use_torchsummary:
        try:
            from torchsummary import summary
            model_to_summary = model.to(device)
            summary(model_to_summary, input_size=input_shape, device=str(device))
        except ImportError:
            logger.warning("  torchsummary not installed; skipping detailed summary")
        except Exception as e:
            logger.warning("  torchsummary failed: %s", e)


# ---------------------------------------------------------------------------
# Training progress — AE (single loss)
# ---------------------------------------------------------------------------


def log_epoch_ae(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    *,
    best_val: float | None = None,
    is_best: bool = False,
) -> None:
    """Log one epoch for a standard autoencoder (reconstruction loss only)."""
    extra = " [best]" if is_best else ""
    logger.info(
        "Epoch %d/%d | train_loss: %.6f | val_loss: %.6f%s",
        epoch, total_epochs, train_loss, val_loss, extra,
    )
    if best_val is not None:
        logger.debug("  best_val_loss so far: %.6f", best_val)


# ---------------------------------------------------------------------------
# Training progress — VQ-VAE (recon + vq_loss + perplexity)
# ---------------------------------------------------------------------------


def log_epoch_vqvae(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_recon_loss: float,
    train_vq_loss: float,
    train_perplexity: float,
    val_recon_loss: float,
    val_vq_loss: float,
    val_perplexity: float,
    *,
    best_val_recon: float | None = None,
    is_best: bool = False,
) -> None:
    """Log one epoch for a VQ-VAE (reconstruction, VQ, and perplexity)."""
    extra = " [best]" if is_best else ""
    logger.info(
        "Epoch %d/%d | train_recon: %.6f | train_vq: %.6f | train_perp: %.4f | val_recon: %.6f | val_vq: %.6f | val_perp: %.4f%s",
        epoch, total_epochs,
        train_recon_loss, train_vq_loss, train_perplexity,
        val_recon_loss, val_vq_loss, val_perplexity,
        extra,
    )
    if best_val_recon is not None:
        logger.debug("  best_val_recon so far: %.6f", best_val_recon)


# ---------------------------------------------------------------------------
# Training end / checkpoint
# ---------------------------------------------------------------------------


def log_training_end(
    logger: logging.Logger,
    best_val_loss: float,
    save_path: str | Path,
    *,
    is_vqvae: bool = False,
    best_val_perplexity: float | None = None,
) -> None:
    """Log training completion and checkpoint path."""
    logger.info("Training finished.")
    if is_vqvae and best_val_perplexity is not None:
        logger.info("  best_val_recon: %.6f | best_val_perplexity: %.4f", best_val_loss, best_val_perplexity)
    else:
        logger.info("  best_val_loss: %.6f", best_val_loss)
    logger.info("  checkpoint: %s", Path(save_path).resolve())


# ---------------------------------------------------------------------------
# Optional: track metrics for later export (e.g. plotting)
# ---------------------------------------------------------------------------


class TrainingTracker:
    """Accumulate per-epoch metrics for AE or VQ-VAE and optionally export to JSON/CSV."""

    def __init__(self, is_vqvae: bool = False) -> None:
        self.is_vqvae = is_vqvae
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        if is_vqvae:
            self.train_vq: list[float] = []
            self.train_perp: list[float] = []
            self.val_vq: list[float] = []
            self.val_perp: list[float] = []

    def step_ae(self, train_loss: float, val_loss: float) -> None:
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def step_vqvae(
        self,
        train_recon: float,
        train_vq: float,
        train_perp: float,
        val_recon: float,
        val_vq: float,
        val_perp: float,
    ) -> None:
        self.train_losses.append(train_recon)
        self.val_losses.append(val_recon)
        self.train_vq.append(train_vq)
        self.train_perp.append(train_perp)
        self.val_vq.append(val_vq)
        self.val_perp.append(val_perp)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
        }
        if self.is_vqvae:
            out["train_vq_loss"] = self.train_vq
            out["train_perplexity"] = self.train_perp
            out["val_vq_loss"] = self.val_vq
            out["val_perplexity"] = self.val_perp
        return out


def log_config(
    logger: logging.Logger,
    *,
    config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Log a generic config dict or keyword args (e.g. lr, optimizer, epochs)."""
    d = dict(config) if config else {}
    d.update(kwargs)
    logger.info("Config: %s", d)


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------


def log_eval_results(
    logger: logging.Logger,
    *,
    num_test_samples: int,
    auc_mse: float,
    auc_nll: float,
    vqvae_checkpoint: str | Path | None = None,
    prior_checkpoint: str | Path | None = None,
) -> None:
    """Log anomaly detection evaluation summary (ROC AUC for MSE and NLL criteria)."""
    logger.info("Evaluation finished.")
    logger.info("  num_test_samples: %d", num_test_samples)
    if vqvae_checkpoint is not None:
        logger.info("  vqvae_checkpoint: %s", Path(vqvae_checkpoint).resolve())
    if prior_checkpoint is not None:
        logger.info("  prior_checkpoint: %s", Path(prior_checkpoint).resolve())
    logger.info("  ROC AUC (MSE criterion):  %.4f", auc_mse)
    logger.info("  ROC AUC (PixelSNAIL NLL): %.4f", auc_nll)
