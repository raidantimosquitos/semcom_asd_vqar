from __future__ import annotations

import torch
import torch.nn as nn


class MachineIDClassifierHead(nn.Module):
    """
    Classifier on the continuous latent (B, C, H, W) from the VQ-VAE encoder.
    Pools to one vector per sample, then two FC layers; output logits for num_machine_ids.
    """

    def __init__(self, latent_channels: int, num_machine_ids: int, hidden: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(latent_channels, hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden, num_machine_ids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        x = self.pool(x).flatten(1)
        x = self.elu(self.fc1(x))
        return self.fc2(x)
