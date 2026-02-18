from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.data.preprocessing import DCASE2020Task2Dataset, compute_dataset_stats
from src.models.autoencoders import BasicAutoEncoder, BasicVQVAE, MobileNetV2_8x_VQVAE
from src.models.pixelsnail import create_pixelsnail_for_vqvae
from src.utils.logger import log_config, log_data_loading, log_epoch_ae, log_epoch_vqvae, log_training_end

def train_val_split(dataset: DCASE2020Task2Dataset, test_size: float = 0.1, random_state: int = 42) -> tuple[Subset, Subset]:
    """
    Split the dataset into train and validation sets.
    Args:
        dataset: The dataset to split.
        test_size: The size of the validation set.
        random_state: The random state to use.
    Returns:
        The train and validation sets.
    """
    # Get the indices of the dataset
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    # Create the train and validation sets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    return train_dataset, val_dataset


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float | None, float | None, float | None]:
    """
    Train the model for one epoch.
    Returns:
        (total_loss, recon_loss, vq_loss). recon/vq are None for non-VQ-VAE models.
    """
    model.train()
    total_loss = 0.0
    recon_loss_sum = 0.0
    vq_loss_sum = 0.0
    perplexity_sum = 0.0
    vq_batches = 0

    for i, batch in enumerate(train_loader):
        # Unpack the batch: inputs are (B, num_windows, 1, 64, 128) per file
        inputs, _, _ = batch
        inputs = inputs.to(device)
        # Flatten to (B*num_windows, 1, 64, 128) so each window is a sample
        B, Nw, C, H, W = inputs.shape
        inputs_flat = inputs.view(B * Nw, C, H, W)

        # If the model is BasicAutoEncoder or BasicVQVAE, flatten the input to (B*N_windows, n_mels*n_time_frames)
        if isinstance(model, BasicAutoEncoder) or isinstance(model, BasicVQVAE):
            inputs_flat = inputs_flat.view(B * Nw, C * H * W)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass (AE returns tensor; VQ-VAE returns (x_recon, vq_loss, perplexity))
        out = model(inputs_flat)
        if isinstance(out, tuple):
            x_recon, vq_loss, perplexity = out
            recon_loss = criterion(x_recon, inputs_flat)
            loss = recon_loss + vq_loss
            recon_loss_sum += recon_loss.item()
            vq_loss_sum += vq_loss.item()
            perplexity_sum += perplexity.item()
            vq_batches += 1 
        else:
            loss = criterion(out, inputs_flat)

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Accumulate the loss
        total_loss += loss.item()

    n = len(train_loader)
    if vq_batches > 0:
        return total_loss / n, recon_loss_sum / vq_batches, vq_loss_sum / vq_batches, perplexity_sum / vq_batches
    return total_loss / n, None, None, None

def evaluate(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, float | None, float | None, float | None]:
    """
    Evaluate the model on the validation set.
    Returns:
        (total_loss, recon_loss, vq_loss). recon/vq are None for non-VQ-VAE models.
    """
    model.eval()
    total_loss = 0.0
    recon_loss_sum = 0.0
    vq_loss_sum = 0.0
    perplexity_sum = 0.0
    vq_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, _, _ = batch
            inputs = inputs.to(device)
            B, Nw, C, H, W = inputs.shape
            inputs_flat = inputs.view(B * Nw, C, H, W)

            if isinstance(model, BasicAutoEncoder) or isinstance(model, BasicVQVAE):
                inputs_flat = inputs_flat.view(B * Nw, C * H * W)

            out = model(inputs_flat)
            if isinstance(out, tuple):
                x_recon, vq_loss, perplexity = out
                recon_loss = criterion(x_recon, inputs_flat)
                loss = recon_loss + vq_loss
                recon_loss_sum += recon_loss.item()
                vq_loss_sum += vq_loss.item()
                perplexity_sum += perplexity.item()
                vq_batches += 1
            else:
                loss = criterion(out, inputs_flat)
            total_loss += loss.item()
    n = len(val_loader)
    if vq_batches > 0:
        return total_loss / n, recon_loss_sum / vq_batches, vq_loss_sum / vq_batches, perplexity_sum / vq_batches
    return total_loss / n, None, None, None

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    save_path: str,
    logger: logging.Logger | None = None,
) -> None:
    """Train the model for a given number of epochs. Logs via logger if provided."""
    log = logger or logging.getLogger(__name__)
    best_loss = float("inf")
    best_val_recon: float | None = None
    for epoch in range(num_epochs):
        train_result = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_result = evaluate(model, val_loader, criterion, device)

        train_loss, train_recon, train_vq, train_perplexity = train_result
        val_loss, val_recon, val_vq, val_perplexity = val_result
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_val_recon = val_recon if val_recon is not None else val_loss
            torch.save(model.state_dict(), save_path)

        if train_recon is not None and train_vq is not None and val_recon is not None and val_vq is not None and train_perplexity is not None and val_perplexity is not None:
            log_epoch_vqvae(
                log, epoch + 1, num_epochs,
                train_recon, train_vq, train_perplexity,
                val_recon, val_vq, val_perplexity,
                best_val_recon=best_val_recon, is_best=is_best,
            )
        else:
            log_epoch_ae(log, epoch + 1, num_epochs, train_loss, val_loss, best_val=best_loss, is_best=is_best)

    log_training_end(log, best_loss, save_path, is_vqvae=(train_recon is not None))


# ---------------------------------------------------------------------------
# Phase 2: Train PixelSNAIL prior (VQ-VAE frozen) for anomaly detection
# ---------------------------------------------------------------------------


def train_prior_one_epoch(
    vqvae: nn.Module,
    prior: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    One epoch of PixelSNAIL (or other prior) training on code indices from frozen VQ-VAE.
    Spectrograms -> VQ-VAE.encode_to_indices -> NLL loss.
    """
    prior.train()
    vqvae.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        inputs, _, _ = batch
        inputs = inputs.to(device)
        B, Nw, C, H, W = inputs.shape
        inputs_flat = inputs.view(B * Nw, C, H, W)

        with torch.no_grad():
            indices = vqvae.encode_to_indices(inputs_flat)  # (B*Nw, 8, 16)

        indices = indices.to(device)
        optimizer.zero_grad()
        loss_dict = prior.loss(indices, reduction="mean")
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches else 0.0


def evaluate_prior(
    vqvae: nn.Module,
    prior: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Validation NLL for the prior (anomaly criterion: higher NLL = more anomalous)."""
    prior.eval()
    vqvae.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs, _, _ = batch
            inputs = inputs.to(device)
            B, Nw, C, H, W = inputs.shape
            inputs_flat = inputs.view(B * Nw, C, H, W)
            indices = vqvae.encode_to_indices(inputs_flat)
            indices = indices.to(device)
            loss_dict = prior.loss(indices, reduction="mean")
            total_loss += loss_dict["loss"].item()
            n_batches += 1

    return total_loss / n_batches if n_batches else 0.0


def train_prior_phase2(
    vqvae: nn.Module,
    prior: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    save_path: str,
    logger: logging.Logger | None = None,
) -> None:
    """Train the prior (e.g. PixelSNAIL) with VQ-VAE frozen. Saves best prior by val NLL."""
    log = logger or logging.getLogger(__name__)
    best_val_nll = float("inf")

    for epoch in range(num_epochs):
        train_nll = train_prior_one_epoch(vqvae, prior, train_loader, optimizer, device)
        val_nll = evaluate_prior(vqvae, prior, val_loader, device)
        is_best = val_nll < best_val_nll
        if is_best:
            best_val_nll = val_nll
            torch.save(prior.state_dict(), save_path)

        log.info(
            "Prior Epoch %d/%d | train_nll: %.6f | val_nll: %.6f%s",
            epoch + 1, num_epochs, train_nll, val_nll, " [best]" if is_best else "",
        )

    log.info("Prior training finished. best_val_nll: %.6f | checkpoint: %s", best_val_nll, save_path)


def run_training(config: dict[str, Any], logger: logging.Logger) -> None:
    """
    Run full training from config: data setup, Phase 1 (VQ-VAE), Phase 2 (PixelSNAIL prior).
    All logic is driven by config; logging goes to the provided logger.
    """
    data_cfg = config.get("data", {})
    root_dir = data_cfg.get("root_dir", "")
    appliance = data_cfg.get("appliance", "fan")
    mode = "train"
    test_size = data_cfg.get("test_size", 0.1)
    batch_size = data_cfg.get("batch_size", 8)
    random_state = data_cfg.get("random_state", 42)
    max_samples_stats = data_cfg.get("max_samples_stats")

    p1 = config.get("phase1", {})
    p2 = config.get("phase2", {})
    num_epochs_vqvae = p1.get("num_epochs", 10)
    num_epochs_prior = p2.get("num_epochs", 20)
    lr_vqvae = p1.get("lr", 0.001)
    lr_prior = p2.get("lr", 1e-4)
    vqvae_checkpoint = p1.get("checkpoint", "checkpoints/mobilenetv2_8x_vqvae.pth")
    prior_checkpoint = p2.get("checkpoint", "checkpoints/pixelsnail_prior.pth")

    requested = config.get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "Config has device=cuda but CUDA is not available (e.g. Colab runtime is CPU). "
            "Use Runtime → Change runtime type → GPU, then re-run. Falling back to CPU."
        )
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    log_config(logger, config=config)
    logger.info("Device: %s", device)

    # Data
    dataset = DCASE2020Task2Dataset(root_dir=root_dir, appliance=appliance, mode=mode)
    mean, std = compute_dataset_stats(dataset, max_samples=max_samples_stats)
    dataset_norm = DCASE2020Task2Dataset(
        root_dir=root_dir, appliance=appliance, mode=mode, mean=mean, std=std
    )
    train_dataset, val_dataset = train_val_split(
        dataset_norm, test_size=test_size, random_state=random_state
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    log_data_loading(
        logger,
        root_dir=root_dir,
        appliance=appliance,
        mode=mode,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        batch_size=batch_size,
        total_samples=len(dataset_norm),
    )

    # Phase 1: VQ-VAE
    vqvae = MobileNetV2_8x_VQVAE().to(device)
    if os.path.exists(vqvae_checkpoint):
        logger.info("Loading VQ-VAE from %s", vqvae_checkpoint)
        vqvae.load_state_dict(torch.load(vqvae_checkpoint, map_location=device))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    if not os.path.exists(vqvae_checkpoint):
        logger.info("Phase 1: Training VQ-VAE")
        for p in vqvae.parameters():
            p.requires_grad = True
        optimizer_vqvae = optim.Adam(vqvae.parameters(), lr=lr_vqvae)
        train(
            vqvae, train_loader, val_loader,
            optimizer_vqvae, nn.MSELoss(), device,
            num_epochs=num_epochs_vqvae, save_path=vqvae_checkpoint, logger=logger,
        )
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad = False

    # Phase 2: PixelSNAIL prior
    prior = create_pixelsnail_for_vqvae(vqvae).to(device)
    optimizer_prior = optim.Adam(prior.parameters(), lr=lr_prior)
    if os.path.exists(prior_checkpoint):
        logger.info("Loading prior from %s", prior_checkpoint)
        prior.load_state_dict(torch.load(prior_checkpoint, map_location=device))

    logger.info("Phase 2: Training PixelSNAIL prior (anomaly criterion)")
    train_prior_phase2(
        vqvae, prior, train_loader, val_loader,
        optimizer_prior, device,
        num_epochs=num_epochs_prior, save_path=prior_checkpoint, logger=logger,
    )