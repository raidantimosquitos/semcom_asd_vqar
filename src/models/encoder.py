from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights

def _mlp_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
    )

class BasicEncoder(nn.Module):
    def __init__(self, x_dim=8192, z_dim=32, h_dim=256):
        super(BasicEncoder, self).__init__()
        self.enc = nn.Sequential(
            _mlp_block(x_dim, h_dim),
            _mlp_block(h_dim, h_dim),
            _mlp_block(h_dim, h_dim),
            _mlp_block(h_dim, z_dim),
        )

    def forward(self, x):
        # x: (B*Nw, 8192)
        z = self.enc(x)
        return z

def _cnn_block(in_ch: int, out_ch: int) -> nn.Module:
    """Conv2d(3x3, stride=1, padding=1) + ReLU + MaxPool2d(2). Halves H and W."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

class CNNEncoder(nn.Module):
    """
    Simple encoder: 3 blocks of Conv2d + ReLU + MaxPool. H and W are halved three times.
    Input:  (B, in_channels, H, W)  e.g. (B, 1, 64, 128)
    Output: (B, latent_channels, H/8, W/8)  e.g. (B, latent_channels, 8, 16)
    """

    def __init__(self, in_channels: int, latent_channels: int):
        super(CNNEncoder, self).__init__()
        self._out_channels = latent_channels
        self.blocks = nn.Sequential(
            _cnn_block(in_channels, 32),
            _cnn_block(32, 64),
            _cnn_block(64, latent_channels),
        )

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

# --- MobileNetV2 encoder (for spectrogram input 64×128) ---

class MobileNetV2_8x_Encoder(nn.Module):
    """
    Truncated MobileNetV2 (features[:7]). Input (B, 1, 64, 128) -> output (B, out_channels, 8, 16).
    Default out_channels=32 (last layer of features[:7] in standard MobileNetV2).
    """

    def __init__(self, pretrained: bool = True):
        super(MobileNetV2_8x_Encoder, self).__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        original_model = mobilenet_v2(weights=weights)
        self.features = original_model.features[:7]
        # Standard MobileNetV2 features[6] output is 32 channels
        self._out_channels = 32

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 64, 128) -> repeat for pretrained 3-channel input
        x = x.repeat(1, 3, 1, 1)
        return self.features(x)

if __name__ == "__main__":
    dummy_input = torch.randn(1, 1, 64, 128)
    model = MobileNetV2_8x_Encoder(pretrained=False)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"out_channels: {model.out_channels}")