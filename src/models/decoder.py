from __future__ import annotations

import torch
import torch.nn as nn

def mlp_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
    )

class BasicDecoder(nn.Module):
    def __init__(self, z_dim=32, x_dim=8192, h_dim=256):
        super().__init__()
        self.dec = nn.Sequential(
            mlp_block(z_dim, h_dim),
            mlp_block(h_dim, h_dim),
            mlp_block(h_dim, h_dim),
            nn.Linear(h_dim, x_dim) 
        )

    def forward(self, z):
        # z: (B*Nw, 32)
        x_recon = self.dec(z)
        return x_recon

def deconv_block(in_ch: int, out_ch: int) -> nn.Module:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class CNNDecoder(nn.Module):
    def __init__(self, latent_channels: int = 128, out_channels: int = 1):
        super(CNNDecoder, self).__init__()
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.blocks = nn.Sequential(
            deconv_block(latent_channels, 64),
            deconv_block(64, 32),
            deconv_block(32, out_channels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, 2, 4)
        Returns:
            (B, 1, 64, 128)
        """
        x = self.blocks(z)
        return x

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
    # AE: decoder receives encoder output channels (32)
    latent_ae = torch.randn(1, 32, 8, 16)
    decoder_ae = MobileNetV2_8x_Decoder(latent_channels=32, out_channels=1)
    print("AE decoder:", decoder_ae(latent_ae).shape)  # [1, 1, 64, 128]

    # VQ-VAE: decoder receives embedding_dim (e.g. 64)
    latent_vq = torch.randn(1, 64, 8, 16)
    decoder_vq = MobileNetV2_8x_Decoder(latent_channels=64, out_channels=1)
    print("VQ-VAE decoder:", decoder_vq(latent_vq).shape)  # [1, 1, 64, 128]
