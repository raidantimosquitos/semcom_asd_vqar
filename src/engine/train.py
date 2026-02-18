from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from src.utils.config import get_checkpoint_paths
from src.data.preprocessing import (
    DCASE2020Task2Dataset,
    compute_dataset_stats,
    load_train_stats,
    save_train_stats,
    train_stats_path,
)
from src.models.autoencoders import BasicAutoEncoder, BasicVQVAE, MobileNetV2_8x_VQVAE
from src.models.secondary_heads import MachineIDClassifierHead
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


def _batch_to_device_and_flat(
    batch: tuple,
    device: torch.device,
    model: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, list[str], int, int]:
    """Unpack batch (windows, machine_id, label, wav_path), move to device, flatten windows. Returns (inputs_flat, inputs, machine_ids, B, Nw)."""
    inputs, machine_ids, _labels, _wav_paths = batch
    inputs = inputs.to(device)
    B, Nw, C, H, W = inputs.shape
    inputs_flat = inputs.view(B * Nw, C, H, W)
    if isinstance(model, BasicAutoEncoder) or isinstance(model, BasicVQVAE):
        inputs_flat = inputs_flat.view(B * Nw, C * H * W)
    return inputs_flat, inputs, machine_ids, B, Nw


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    classifier_head: nn.Module | None = None,
    classifier_optimizer: optim.Optimizer | None = None,
    machine_id_to_idx: dict[str, int] | None = None,
    phase: str = "vqvae_only",
    lambda_1: float = 1.0,
    lambda_2: float = 0.0,
) -> tuple[float, float | None, float | None, float | None, float | None]:
    """
    Train for one epoch. Phase: "vqvae_only" | "joint" | "frozen_codebook".
    Returns:
        (total_loss, recon_loss, vq_loss, perplexity, classifier_loss). Unused entries are None.
    """
    model.train()
    if classifier_head is not None:
        classifier_head.train()
    total_loss = 0.0
    recon_loss_sum = 0.0
    vq_loss_sum = 0.0
    perplexity_sum = 0.0
    classifier_loss_sum = 0.0
    vq_batches = 0
    classifier_batches = 0

    use_classifier = (
        phase in ("joint", "frozen_codebook")
        and classifier_head is not None
        and machine_id_to_idx is not None
        and hasattr(model, "get_encoder_latent")
    )

    for batch in train_loader:
        inputs_flat, inputs, machine_ids, B, Nw = _batch_to_device_and_flat(batch, device, model)

        optimizers = [optimizer]
        if use_classifier and classifier_optimizer is not None:
            optimizers.append(classifier_optimizer)
        for opt in optimizers:
            opt.zero_grad()

        # VQ-VAE forward
        out = model(inputs_flat)
        if isinstance(out, tuple):
            x_recon, vq_loss, perplexity = out
            recon_loss = criterion(x_recon, inputs_flat)
            vqvae_loss = recon_loss + vq_loss
            recon_loss_sum += recon_loss.item()
            vq_loss_sum += vq_loss.item()
            perplexity_sum += perplexity.item()
            vq_batches += 1
        else:
            vqvae_loss = criterion(out, inputs_flat)

        loss = lambda_1 * vqvae_loss

        if use_classifier and classifier_head is not None and machine_id_to_idx is not None:
            # Per-file latent: (B*Nw, C, 8, 16) -> (B, Nw, C, 8, 16) -> mean over Nw -> (B, C, 8, 16)
            z_e = model.get_encoder_latent(inputs_flat)
            _, C_latent, H_lat, W_lat = z_e.shape
            z_e_per_file = z_e.view(B, Nw, C_latent, H_lat, W_lat).mean(dim=1)
            logits = classifier_head(z_e_per_file)
            machine_id_indices = torch.tensor(
                [machine_id_to_idx[mid] for mid in machine_ids],
                dtype=torch.long,
                device=device,
            )
            ce_loss = nn.functional.cross_entropy(logits, machine_id_indices)
            classifier_loss_sum += ce_loss.item()
            classifier_batches += 1
            loss = loss + lambda_2 * ce_loss

        loss.backward()
        optimizer.step()
        if use_classifier and classifier_optimizer is not None:
            classifier_optimizer.step()

        total_loss += loss.item()

    n = len(train_loader)
    cl_loss = (classifier_loss_sum / classifier_batches) if classifier_batches else None
    if vq_batches > 0:
        return (
            total_loss / n,
            recon_loss_sum / vq_batches,
            vq_loss_sum / vq_batches,
            perplexity_sum / vq_batches,
            cl_loss,
        )
    return total_loss / n, None, None, None, cl_loss

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    classifier_head: nn.Module | None = None,
    machine_id_to_idx: dict[str, int] | None = None,
    phase: str = "vqvae_only",
    lambda_1: float = 1.0,
    lambda_2: float = 0.0,
) -> tuple[float, float | None, float | None, float | None, float | None]:
    """
    Evaluate on the validation set. Returns (total_loss, recon_loss, vq_loss, perplexity, classifier_loss).
    """
    model.eval()
    if classifier_head is not None:
        classifier_head.eval()
    total_loss = 0.0
    recon_loss_sum = 0.0
    vq_loss_sum = 0.0
    perplexity_sum = 0.0
    classifier_loss_sum = 0.0
    vq_batches = 0
    classifier_batches = 0
    use_classifier = (
        phase in ("joint", "frozen_codebook")
        and classifier_head is not None
        and machine_id_to_idx is not None
        and hasattr(model, "get_encoder_latent")
    )

    with torch.no_grad():
        for batch in val_loader:
            inputs_flat, inputs, machine_ids, B, Nw = _batch_to_device_and_flat(batch, device, model)

            out = model(inputs_flat)
            if isinstance(out, tuple):
                x_recon, vq_loss, perplexity = out
                recon_loss = criterion(x_recon, inputs_flat)
                vqvae_loss = recon_loss + vq_loss
                recon_loss_sum += recon_loss.item()
                vq_loss_sum += vq_loss.item()
                perplexity_sum += perplexity.item()
                vq_batches += 1
            else:
                vqvae_loss = criterion(out, inputs_flat)

            loss = lambda_1 * vqvae_loss

            if use_classifier and classifier_head is not None and machine_id_to_idx is not None:
                z_e = model.get_encoder_latent(inputs_flat)
                _, C_latent, H_lat, W_lat = z_e.shape
                z_e_per_file = z_e.view(B, Nw, C_latent, H_lat, W_lat).mean(dim=1)
                logits = classifier_head(z_e_per_file)
                machine_id_indices = torch.tensor(
                    [machine_id_to_idx[mid] for mid in machine_ids],
                    dtype=torch.long,
                    device=device,
                )
                ce_loss = nn.functional.cross_entropy(logits, machine_id_indices)
                classifier_loss_sum += ce_loss.item()
                classifier_batches += 1
                loss = loss + lambda_2 * ce_loss

            total_loss += loss.item()

    n = len(val_loader)
    cl_loss = (classifier_loss_sum / classifier_batches) if classifier_batches else None
    if vq_batches > 0:
        return (
            total_loss / n,
            recon_loss_sum / vq_batches,
            vq_loss_sum / vq_batches,
            perplexity_sum / vq_batches,
            cl_loss,
        )
    return total_loss / n, None, None, None, cl_loss

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
    *,
    classifier_head: nn.Module | None = None,
    classifier_optimizer: optim.Optimizer | None = None,
    machine_id_to_idx: dict[str, int] | None = None,
    phase: str = "vqvae_only",
    lambda_1: float = 1.0,
    lambda_2: float = 0.0,
    save_classifier_path: str | None = None,
) -> None:
    """Train for a given number of epochs. Optional classifier head with phase and lambdas."""
    log = logger or logging.getLogger(__name__)
    best_loss = float("inf")
    best_val_recon: float | None = None
    for epoch in range(num_epochs):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            classifier_head=classifier_head,
            classifier_optimizer=classifier_optimizer,
            machine_id_to_idx=machine_id_to_idx,
            phase=phase,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )
        val_result = evaluate(
            model,
            val_loader,
            criterion,
            device,
            classifier_head=classifier_head,
            machine_id_to_idx=machine_id_to_idx,
            phase=phase,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )

        train_loss, train_recon, train_vq, train_perplexity, train_cl = train_result
        val_loss, val_recon, val_vq, val_perplexity, val_cl = val_result
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_val_recon = val_recon if val_recon is not None else val_loss
            torch.save(model.state_dict(), save_path)
            if save_classifier_path and classifier_head is not None:
                torch.save(classifier_head.state_dict(), save_classifier_path)

        if train_recon is not None and train_vq is not None and val_recon is not None and val_vq is not None and train_perplexity is not None and val_perplexity is not None:
            log_epoch_vqvae(
                log, epoch + 1, num_epochs,
                train_recon, train_vq, train_perplexity,
                val_recon, val_vq, val_perplexity,
                best_val_recon=best_val_recon, is_best=is_best,
            )
            if train_cl is not None and val_cl is not None:
                log.info(
                    "  classifier loss | train: %.4f | val: %.4f",
                    train_cl, val_cl,
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
        inputs, _, _, _ = batch
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
            inputs, _, _, _ = batch
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
    phase1a_epochs = p1.get("phase1a_epochs")
    phase1b_epochs = p1.get("phase1b_epochs")
    phase1c_epochs = p1.get("phase1c_epochs")
    lambda_1 = float(p1.get("lambda_1", 1.0))
    lambda_2 = float(p1.get("lambda_2", 0.5))
    lr_vqvae = p1.get("lr", 0.001)
    classifier_checkpoint = p1.get("classifier_checkpoint", "checkpoints/models/machine_id_classifier.pth")
    num_epochs_prior = p2.get("num_epochs", 20)
    lr_prior = p2.get("lr", 1e-4)
    _, stats_dir = get_checkpoint_paths(config)
    vqvae_checkpoint = p1.get("checkpoint", "checkpoints/models/mobilenetv2_8x_vqvae.pth")
    prior_checkpoint = p2.get("checkpoint", "checkpoints/models/pixelsnail_prior.pth")

    use_three_phases = phase1a_epochs is not None
    if use_three_phases:
        phase1a_epochs = int(phase1a_epochs)
        phase1b_epochs = int(phase1b_epochs or 0)
        phase1c_epochs = int(phase1c_epochs or 0)

    requested = config.get("device", "cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "Config has device=cuda but CUDA is not available (e.g. Colab runtime is CPU). "
            "Use Runtime → Change runtime type → GPU, then re-run. Falling back to CPU."
        )
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    log_config(logger, config=config)
    logger.info("Device: %s", device)

    # Data: load precomputed train stats if present, else compute and save
    dataset = DCASE2020Task2Dataset(root_dir=root_dir, appliance=appliance, mode=mode)
    stats_loaded = load_train_stats(stats_dir, appliance)
    if stats_loaded is not None:
        mean, std = stats_loaded
        logger.info(
            "Loaded train stats from %s: mean=%.6f | std=%.6f",
            train_stats_path(stats_dir, appliance),
            mean,
            std,
        )
    else:
        n_stats = len(dataset) if max_samples_stats is None else min(len(dataset), max_samples_stats)
        logger.info(
            "Computing dataset mean/std over %d samples (may take a while if data is on Drive)...",
            n_stats,
        )
        mean, std = compute_dataset_stats(dataset, max_samples=max_samples_stats, logger=logger)
        logger.info("Dataset mean: %.6f | std: %.6f", mean, std)
        save_path = save_train_stats(stats_dir, appliance, mean, std)
        logger.info("Saved train stats to %s", save_path)

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

    # Phase 1: VQ-VAE (single-phase or 3-phase with classifier)
    Path(vqvae_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    Path(prior_checkpoint).parent.mkdir(parents=True, exist_ok=True)
    if use_three_phases:
        Path(classifier_checkpoint).parent.mkdir(parents=True, exist_ok=True)
        unique_machine_ids = dataset_norm.unique_machine_ids
        machine_id_to_idx = {mid: i for i, mid in enumerate(unique_machine_ids)}
        num_machine_ids = len(unique_machine_ids)
        logger.info("Machine ID classifier: %d classes %s", num_machine_ids, unique_machine_ids)

    vqvae = MobileNetV2_8x_VQVAE().to(device)
    if os.path.exists(vqvae_checkpoint):
        logger.info("Loading VQ-VAE from %s", vqvae_checkpoint)
        vqvae.load_state_dict(torch.load(vqvae_checkpoint, map_location=device, weights_only=True))
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    if not use_three_phases:
        # Single-phase: VQ-VAE only for num_epochs
        if not os.path.exists(vqvae_checkpoint):
            logger.info("Phase 1: Training VQ-VAE (%d epochs)", num_epochs_vqvae)
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
    else:
        # Phase 1a: VQ-VAE only
        if phase1a_epochs > 0 and not os.path.exists(vqvae_checkpoint):
            logger.info("Phase 1a: Training VQ-VAE only (%d epochs)", phase1a_epochs)
            for p in vqvae.parameters():
                p.requires_grad = True
            optimizer_vqvae = optim.Adam(vqvae.parameters(), lr=lr_vqvae)
            train(
                vqvae, train_loader, val_loader,
                optimizer_vqvae, nn.MSELoss(), device,
                num_epochs=phase1a_epochs, save_path=vqvae_checkpoint, logger=logger,
                phase="vqvae_only",
            )
        vqvae.eval()
        for p in vqvae.parameters():
            p.requires_grad = False

        # Phase 1b: VQ-VAE + classifier (joint)
        if phase1b_epochs > 0:
            latent_channels = vqvae.quantizer._embedding.weight.shape[1]
            classifier_head = MachineIDClassifierHead(latent_channels, num_machine_ids).to(device)
            if os.path.exists(classifier_checkpoint):
                logger.info("Loading classifier from %s", classifier_checkpoint)
                classifier_head.load_state_dict(torch.load(classifier_checkpoint, map_location=device, weights_only=True))
            for p in vqvae.parameters():
                p.requires_grad = True
            optimizer_vqvae = optim.Adam(vqvae.parameters(), lr=lr_vqvae)
            optimizer_classifier = optim.Adam(classifier_head.parameters(), lr=lr_vqvae)
            logger.info("Phase 1b: Training VQ-VAE + classifier (%d epochs, lambda_1=%.2f, lambda_2=%.2f)", phase1b_epochs, lambda_1, lambda_2)
            train(
                vqvae, train_loader, val_loader,
                optimizer_vqvae, nn.MSELoss(), device,
                num_epochs=phase1b_epochs, save_path=vqvae_checkpoint, logger=logger,
                classifier_head=classifier_head,
                classifier_optimizer=optimizer_classifier,
                machine_id_to_idx=machine_id_to_idx,
                phase="joint",
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                save_classifier_path=classifier_checkpoint,
            )
            vqvae.eval()
            classifier_head.eval()
            for p in vqvae.parameters():
                p.requires_grad = False

        # Phase 1c: Frozen codebook, recon + classifier
        if phase1c_epochs > 0:
            for p in vqvae.quantizer.parameters():
                p.requires_grad = False
            for p in vqvae.encoder.parameters():
                p.requires_grad = True
            for p in vqvae.decoder.parameters():
                p.requires_grad = True
            if phase1b_epochs <= 0:
                latent_channels = vqvae.quantizer._embedding.weight.shape[1]
                classifier_head = MachineIDClassifierHead(latent_channels, num_machine_ids).to(device)
                if os.path.exists(classifier_checkpoint):
                    classifier_head.load_state_dict(torch.load(classifier_checkpoint, map_location=device, weights_only=True))
                optimizer_classifier = optim.Adam(classifier_head.parameters(), lr=lr_vqvae)
            optimizer_vqvae = optim.Adam(
                [p for p in vqvae.parameters() if p.requires_grad], lr=lr_vqvae
            )
            logger.info("Phase 1c: Frozen codebook, recon + classifier (%d epochs)", phase1c_epochs)
            train(
                vqvae, train_loader, val_loader,
                optimizer_vqvae, nn.MSELoss(), device,
                num_epochs=phase1c_epochs, save_path=vqvae_checkpoint, logger=logger,
                classifier_head=classifier_head,
                classifier_optimizer=optimizer_classifier,
                machine_id_to_idx=machine_id_to_idx,
                phase="frozen_codebook",
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                save_classifier_path=classifier_checkpoint,
            )
            vqvae.eval()
            for p in vqvae.parameters():
                p.requires_grad = False

    # Phase 2: PixelSNAIL prior
    prior = create_pixelsnail_for_vqvae(vqvae).to(device)
    optimizer_prior = optim.Adam(prior.parameters(), lr=lr_prior)
    if os.path.exists(prior_checkpoint):
        logger.info("Loading prior from %s", prior_checkpoint)
        prior.load_state_dict(torch.load(prior_checkpoint, map_location=device, weights_only=True))

    logger.info("Phase 2: Training PixelSNAIL prior (anomaly criterion)")
    train_prior_phase2(
        vqvae, prior, train_loader, val_loader,
        optimizer_prior, device,
        num_epochs=num_epochs_prior, save_path=prior_checkpoint, logger=logger,
    )