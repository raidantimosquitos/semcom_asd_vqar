import os
import glob

import torch
from torch.utils.data import Dataset

from src.utils.audio import collect_audio_files, load_wav, LogMelSpectrogramExtractor


class DCASE2020Task2Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        appliance: str,
        mode: str = "train",
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        frames: int = 64,
        hop_size: int = 8,
        power: float = 2.0,
        eps: float = 1e-10,
        ext: str = "wav",
        mean: float | None = None,
        std: float | None = None,
        norm_eps: float = 1e-8,
    ):
        self.root_dir = root_dir
        self.appliance = appliance
        self.mode = mode
        self.ext = ext
        self.sample_rate = sample_rate
        self.samples: list[tuple[str, int]] = collect_audio_files(
            self.root_dir, self.appliance, self.mode, self.ext
        )

        self.extractor = LogMelSpectrogramExtractor(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            frames=frames,
            hop_size=hop_size,
            power=power,
            eps=eps,
        )

        self._mean = mean
        self._std = std
        self._norm_eps = norm_eps

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        """Return (windows, label, wav_path). Windows are normalized if mean/std were set."""
        wav_path, label = self.samples[idx]
        waveform = load_wav(wav_path, self.sample_rate)
        windows = self.extractor(waveform)

        if self._mean is not None and self._std is not None:
            windows = (windows - self._mean) / (self._std + self._norm_eps)

        return windows, label, wav_path


def compute_dataset_stats(
    dataset: DCASE2020Task2Dataset,
    max_samples: int | None = None,
    eps: float = 1e-8,
) -> tuple[float, float]:
    """
    Compute mean and std over all windows in the dataset (for zero-mean unit-variance).
    Args:
        dataset: DCASE2020Task2Dataset instance.
        max_samples: If set, use at most this many indices (for faster approximation).
        eps: Small value for numerical stability when computing std.
    Returns:
        (mean, std) scalars.
    """
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    total_sum = 0.0
    total_sq = 0.0
    count = 0
    for i in range(n):
        windows, _, _ = dataset[i]
        total_sum += windows.sum().item()
        total_sq += (windows ** 2).sum().item()
        count += windows.numel()
    mean = total_sum / count
    var = (total_sq / count) - (mean ** 2)
    std = (var + eps) ** 0.5
    return float(mean), float(std)


if __name__ == "__main__":
    root_dir = "/mnt/ssd/LaCie/dcase2020-task2-dev-dataset"
    appliance = "fan"
    mode = "train"

    # Create dataset (unnormalized)
    dataset = DCASE2020Task2Dataset(
        root_dir=root_dir,
        appliance=appliance,
        mode=mode,
    )

    # Compute mean/std and create normalized dataset (use same mean/std for train and val)
    mean, std = compute_dataset_stats(dataset, max_samples=None)
    print(f"Dataset mean: {mean:.4f}, std: {std:.4f}")

    dataset_norm = DCASE2020Task2Dataset(
        root_dir=root_dir,
        appliance=appliance,
        mode=mode,
        mean=mean,
        std=std,
    )

    print(len(dataset_norm))
    w, label, path = dataset_norm[0]
    print("First sample shape:", w.shape)
    print("First sample (normalized) mean ~0, std ~1:", w.mean().item(), w.std().item())

    # Ensure all tensors have shape (N, 1, 64, 128)
    expected_dims = (1, 64, 128)
    for i in range(len(dataset_norm)):
        windows = dataset_norm[i][0]
        assert windows.dim() == 4 and windows.shape[1:] == expected_dims, (
            f"Sample {i}: got {windows.shape}, expected (N, 1, 64, 128)"
        )
    print("All samples have windows of shape (N, 1, 64, 128).")