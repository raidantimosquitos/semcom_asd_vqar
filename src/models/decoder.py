from __future__ import annotations

import torch
import torch.nn as nn

from src.models.nn_blocks import _mlp_block, _deconv_block, ResBlock

class BasicDecoder(nn.Module):
    def __init__(self, z_dim=32, x_dim=8192, h_dim=256):
        super().__init__()
        self.dec = nn.Sequential(
            _mlp_block(z_dim, h_dim),
            _mlp_block(h_dim, h_dim),
            _mlp_block(h_dim, h_dim),
            nn.Linear(h_dim, x_dim) 
        )

    def forward(self, z):
        # z: (B*Nw, 32)
        x_recon = self.dec(z)
        return x_recon

class CNNDecoder(nn.Module):
    def __init__(self, latent_channels: int = 128, out_channels: int = 1):
        super(CNNDecoder, self).__init__()
        self._out_channels = out_channels
        # 8x16xD -> 8x16x128
        self.proj = nn.Sequential(
            nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
        )
        self.res1 = ResBlock(128)

        # 8x16x128 -> 16x32x64
        self.deconv1 = _deconv_block(128, 64)
        self.res2 = ResBlock(64)

        # 16x32x64 -> 32x64x32
        self.deconv2 = _deconv_block(64, 32)

        # 32x64x32 -> 64x128x1
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, 2, 4)
        Returns:
            (B, 1, 64, 128)
        """
        z = self.proj(z)
        z = self.res1(z)
        z = self.deconv1(z)
        z = self.res2(z)
        z = self.deconv2(z)
        z = self.deconv3(z)
        return z

class MobileNetV2_8x_Decoder(nn.Module):
    """
    Decoder for MobileNetV2_8x_Encoder. Accepts latent_channels so it works for both AE and VQ-VAE.
    Input:  (B, latent_channels, 8, 16)  — latent_channels = encoder.out_channels (AE) or embedding_dim (VQ-VAE)
    Output: (B, out_channels, 64, 128)
    """

    def __init__(self, latent_channels: int, out_channels: int = 1):
        super(MobileNetV2_8x_Decoder, self).__init__()
        self.initial_proj = nn.Conv2d(latent_channels, 128, kernel_size=3, padding=1)
        self.bn_proj = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, latent_channels, 8, 16) -> (B, 1, 64, 128)
        x = self.relu(self.bn_proj(self.initial_proj(z)))
        x = self.up1(x)  # (B, 64, 16, 32)
        x = self.up2(x)  # (B, 32, 32, 64)
        x = self.up3(x)  # (B, 16, 64, 128)
        return self.final_conv(x)


if __name__ == "__main__":
    # AE: decoder receives encoder output channels (128)
    latent_ae = torch.randn(1, 128, 8, 16)
    decoder_ae = CNNDecoder(latent_channels=128, out_channels=1)
    print("AE decoder:", decoder_ae(latent_ae).shape)  # [1, 1, 64, 128]

    # VQ-VAE: decoder receives embedding_dim (e.g. 128)
    latent_vq = torch.randn(1, 128, 8, 16)
    decoder_vq = CNNDecoder(latent_channels=128, out_channels=1)
    print("VQ-VAE decoder:", decoder_vq(latent_vq).shape)  # [1, 1, 64, 128]
