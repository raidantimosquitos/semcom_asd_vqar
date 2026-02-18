import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
import glob

class LogMelSpectrogramExtractor(nn.Module):
    def __init__(self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        frames: int = 64,
        hop_size: int = 8,
        power: float = 2.0,
        eps: float = 1e-10
        ):
        super(LogMelSpectrogramExtractor, self).__init__()
        self.frames = frames
        self.hop_size = hop_size
        self.power = power
        self.eps = eps

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power
        )

    def forward(self, waveform):
        """
        Args:
            waveform: (1, T) or (T,)
        Returns:
            windows: (N, 1, 64, 128) — N sliding-window log-mel patches (channel, time_frames, n_mels).
                    Ready for Conv2d; use .squeeze(1) if the model expects (N, 64, 128).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel_spec = self.mel_spec(waveform)  # (1, n_mels, T)
        log_mel = (20 / self.power) * torch.log10(mel_spec + self.eps)

        # Sliding window: unfold time dimension with window size self.frames, step self.hop_size
        # log_mel: (1, n_mels, T) -> unfold(2, frames, hop_size) -> (1, n_mels, num_windows, frames)
        log_mel = log_mel.squeeze(0)  # (n_mels, T)
        if log_mel.size(1) < self.frames:
            # Signal shorter than one window: pad to one full window
            pad_len = self.frames - log_mel.size(1)
            log_mel = F.pad(log_mel, (0, pad_len), mode="constant", value=log_mel.min().item() if log_mel.numel() else -10.0)
        windows = log_mel.unfold(1, self.frames, self.hop_size)  # (n_mels, num_windows, frames)
        # Rearrange to (num_windows, n_mels, frames) then to (num_windows, frames, n_mels) for 64x128
        windows = windows.permute(1, 0, 2)   # (num_windows, n_mels, frames)
        # Network expects (64, 128) = (time_frames, n_mels) per sample
        windows = windows.permute(0, 2, 1)   # (num_windows, frames, n_mels) = (N, 64, 128)
        # Add channel dim for Conv2d: (N, 1, 64, 128)
        windows = windows.unsqueeze(1)
        return windows

def load_wav(path: str, target_sr: int = 16000):
    """
    Load a WAV file and resample it to the target sample rate.
    Args:
        path: Path to the WAV file.
        target_sr: Target sample rate.
    Returns:
        waveform: (1, T)
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)
    return waveform

def collect_audio_files(
    root_dir: str,
    appliance: str,
    mode: str = "train", # train or test
    ext: str = "wav",
) -> list[tuple[str, int]]:
    """
    Collect (audio file path, label) pairs from the root directory.

    Returns:
        List of (path, label) tuples. Label 0 = normal, 1 = anomaly.
        Use a list of tuples so the Dataset keeps each pair in sync and
        __getitem__(i) is a single index.
    """
    base_dir = os.path.join(root_dir, appliance, mode)
    # DCASE 2020 Task 2 dev: train has only normal files directly in base_dir (no "normal" subfolder)
    if mode == "train":
        files = sorted(glob.glob(os.path.join(base_dir, f"*.{ext}")))
        return [(p, 0) for p in files]
    if mode == "test":
        # Some DCASE layouts store all files directly in base_dir; label is encoded in filename.
        # e.g. normal_id_00_*.wav / anomaly_id_00_*.wav
        files = sorted(glob.glob(os.path.join(base_dir, f"*.{ext}")))
        pairs: list[tuple[str, int]] = []
        for p in files:
            name = os.path.basename(p)
            if name.startswith("normal"):
                pairs.append((p, 0))
            elif name.startswith("anomaly"):
                pairs.append((p, 1))
        return pairs
    return []