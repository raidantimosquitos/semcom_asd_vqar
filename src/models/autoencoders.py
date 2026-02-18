from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import BasicDecoder, MobileNetV2_8x_Decoder, CNNDecoder
from .encoder import BasicEncoder, MobileNetV2_8x_Encoder, CNNEncoder
from .quantizer import VectorQuantizer

class BasicAutoEncoder(nn.Module):
    def __init__(self, x_dim=8192, z_dim=32, h_dim=256):
        super(BasicAutoEncoder, self).__init__()
        self.encoder = BasicEncoder(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim)
        self.decoder = BasicDecoder(z_dim=z_dim, x_dim=x_dim, h_dim=h_dim)

    def forward(self, x):
        # x: (B, 320)
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

class BasicVQVAE(nn.Module):
    def __init__(self, x_dim=320, z_dim=8, h_dim=64, num_embeddings=64, embedding_dim=8, commitment_cost=0.25):
        super(BasicVQVAE, self).__init__()
        self.encoder = BasicEncoder(x_dim=x_dim, z_dim=z_dim, h_dim=h_dim)
        self._pre_vq_layer = nn.Linear(z_dim, embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = BasicDecoder(z_dim=embedding_dim, x_dim=x_dim, h_dim=h_dim)

    def forward(self, x):
        # x: (B, 320)
        z_e = self.encoder(x)  # (B, 8)
        z_e = self._pre_vq_layer(z_e)  # (B, 8)
        z_e = z_e.unsqueeze(1)  # (B, 1, 8) - Add time dimension for quantizer
        z_q, vq_loss, perplexity = self.quantizer(z_e)  # (B, 1, 8)
        z_q = z_q.squeeze(1)  # (B, 8)
        x_recon = self.decoder(z_q)  # (B, 320)
        return x_recon, vq_loss, perplexity

class CNNAutoEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_channels=128):
        super(CNNAutoEncoder, self).__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self.decoder = CNNDecoder(latent_channels=latent_channels, out_channels=in_channels)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

class CNNVQVAE(nn.Module):
    def __init__(self, in_channels=1, latent_channels=128, num_embeddings=64, embedding_dim=8, commitment_cost=0.25):
        super(CNNVQVAE, self).__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, latent_channels=latent_channels)
        self._pre_vq_layer = nn.Conv2d(latent_channels, embedding_dim, kernel_size=1, stride=1)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        self.decoder = CNNDecoder(latent_channels=embedding_dim, out_channels=in_channels)

    def forward(self, x):
        z_e = self.encoder(x)  # (B, latent_channels, 8, 16)
        z_e = self._pre_vq_layer(z_e)  # (B, embedding_dim, 8, 16)
        z_q, vq_loss, perplexity = self.quantizer(z_e)  # quantizer accepts 4D, returns (B, embedding_dim, 8, 16)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, perplexity


class MobileNetV2_8x_AE(nn.Module):
    """
    Autoencoder for spectrogram (1, 64, 128). Encoder -> (B, 32, 8, 16), decoder -> (B, 1, 64, 128).
    """

    def __init__(self, pretrained: bool = True, out_channels: int = 1):
        super().__init__()
        self.encoder = MobileNetV2_8x_Encoder(pretrained=pretrained)
        self.decoder = MobileNetV2_8x_Decoder(
            latent_channels=self.encoder.out_channels,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class MobileNetV2_8x_VQVAE(nn.Module):
    """
    VQ-VAE with MobileNetV2_8x encoder. Latent (B, 32, 8, 16) -> pre_vq -> (B, embedding_dim, 8, 16);
    quantizer over (B, 8*16, embedding_dim); decoder -> (B, 1, 64, 128).
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_embeddings: int = 512,
        embedding_dim: int = 32,
        commitment_cost: float = 0.25,
        out_channels: int = 1,
    ):
        super().__init__()
        self.encoder = MobileNetV2_8x_Encoder(pretrained=pretrained)

        # Pre-VQ layer: (B, 32, 8, 16) -> (B, embedding_dim, 8, 16)
        self._pre_vq = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim), # Keeps output centered and scaled
            nn.ReLU() # Optional: if you want non-negative codebook
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
        )
        self.decoder = MobileNetV2_8x_Decoder(
            latent_channels=embedding_dim,
            out_channels=out_channels,
        )

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)  # (B, 32, 8, 16)
        z_e = self._pre_vq(z_e)  # (B, embedding_dim, 8, 16)
        z_q, vq_loss, perplexity = self.quantizer(z_e)  # quantizer accepts 4D, returns same shape
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, perplexity

    def encode_to_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Encode spectrograms to discrete code indices (B, H, W) for e.g. PixelSNAIL training."""
        z_e = self.encoder(x)
        z_e = self._pre_vq(z_e)  # (B, C, H, W)
        B, C, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, C)
        distances = (
            torch.sum(flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.quantizer._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat, self.quantizer._embedding.weight.t())
        )
        indices = torch.argmin(distances, dim=1)
        return indices.view(B, H, W)

if __name__ == "__main__":
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2_8x_AE().to(device)

    x = torch.randn(1, 1, 64, 128).to(device)
    z = model(x)
    print(z.shape)

    # Correct way to print summary
    summary(model, input_size=(1, 64, 128), device=str(device))