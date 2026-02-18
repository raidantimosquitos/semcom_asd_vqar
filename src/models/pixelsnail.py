"""
PixelSNAIL for modeling discrete VQ-VAE code indices (autoregressive over the latent grid).
Configured for MobileNetV2_8x_VQVAE latent shape (8, 16) and codebook size num_embeddings.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (self-contained, no external nn_blocks)
# ---------------------------------------------------------------------------


class MaskedConv2d(nn.Module):
    """
    Causal 2D convolution: position (r, c) only sees (r', c') before it in row-major order.
    mask_type='A': also mask the center (so we predict current pixel from context only).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        mask_type: str = "A",
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        k = kernel_size
        self.register_buffer(
            "mask",
            self._make_mask(k, mask_type),
        )

    @staticmethod
    def _make_mask(kernel_size: int, mask_type: str) -> torch.Tensor:
        # Row-major: (i, j) before center (c, c) iff i < c or (i == c and j < c)
        # Type A: mask center too. Type B: include center (for subsequent layers).
        c = kernel_size // 2
        mask = torch.zeros(1, 1, kernel_size, kernel_size)
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i < c or (i == c and j < c):
                    mask[0, 0, i, j] = 1.0
                elif mask_type == "B" and i == c and j == c:
                    mask[0, 0, i, j] = 1.0
        if mask_type == "A":
            mask[0, 0, c, c] = 0.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weight = self.conv.weight * self.mask
        return F.conv2d(
            x, masked_weight, self.conv.bias,
            self.conv.stride, self.conv.padding,
        )


class WNConv2d(nn.Module):
    """Weight-normalized Conv2d."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PixelBlock(nn.Module):
    """Residual block with masked convolutions for autoregressive context."""

    def __init__(
        self,
        n_channels: int,
        n_res_block: int,
        shape: tuple[int, int],
        dropout_p: float = 0.1,
        cond_channels: int | None = None,
        non_linearity: Any = F.elu,
    ):
        super().__init__()
        self.non_linearity = non_linearity
        self.cond_channels = cond_channels
        height, width = shape

        # Coordinate embedding (like in reference)
        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer("background", torch.cat([coord_x, coord_y], 1))

        res_layers = []
        for _ in range(n_res_block):
            res_layers.append(
                nn.Sequential(
                    MaskedConv2d(n_channels, n_channels, 3, padding=1, mask_type="B"),
                    nn.BatchNorm2d(n_channels),
                    nn.ELU(),
                    nn.Dropout2d(dropout_p),
                    MaskedConv2d(n_channels, n_channels, 3, padding=1, mask_type="B"),
                    nn.BatchNorm2d(n_channels),
                )
            )
        self.res_blocks = nn.ModuleList(res_layers)
        in_attn = n_channels + 2  # +2 for coord
        if cond_channels is not None:
            in_attn += cond_channels
        self.out_conv = WNConv2d(in_attn, n_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        background: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.res_blocks:
            x = self.non_linearity(x + block(x))
        batch = x.shape[0]
        bg = background.expand(batch, -1, -1, -1)
        out = torch.cat([bg, x], dim=1)
        if cond is not None:
            out = torch.cat([out, cond], dim=1)
        return self.out_conv(out)


# ---------------------------------------------------------------------------
# PixelSNAIL model
# ---------------------------------------------------------------------------


class PixelSNAIL(nn.Module):
    """
    Autoregressive model over discrete code indices (e.g. from VQ-VAE).
    Input: indices (B, H, W) in [0, num_embeddings-1].
    Output: logits (B, num_embeddings, H, W) for cross-entropy / NLL.
    """

    def __init__(
        self,
        num_embeddings: int,
        shape: tuple[int, int] = (8, 16),
        n_channels: int = 64,
        n_block: int = 4,
        n_res_block: int = 2,
        dropout_p: float = 0.1,
        cond_channels: int | None = None,
        downsample: int = 1,
        non_linearity: Any = F.elu,
    ):
        super().__init__()
        self.d = num_embeddings
        self.shape = shape
        self.cond_channels = cond_channels
        self.non_linearity = non_linearity
        height, width = shape

        self.ini_conv = MaskedConv2d(
            num_embeddings,
            n_channels,
            kernel_size=7,
            stride=downsample,
            padding=3,
            mask_type="A",
        )
        height //= downsample
        width //= downsample
        self.latent_shape = (height, width)

        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer("background", torch.cat([coord_x, coord_y], 1))

        self.blocks = nn.ModuleList()
        for _ in range(n_block):
            self.blocks.append(
                PixelBlock(
                    n_channels,
                    n_res_block=n_res_block,
                    shape=(height, width),
                    dropout_p=dropout_p,
                    cond_channels=cond_channels,
                    non_linearity=non_linearity,
                )
            )

        self.upsample = nn.ConvTranspose2d(
            n_channels, n_channels, kernel_size=downsample, stride=downsample
        )
        self.out = WNConv2d(n_channels, num_embeddings, 1)

    def forward(
        self,
        input_indices: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_indices: (B, H, W) long tensor, values in [0, num_embeddings-1].
            cond: optional (B, cond_channels, H, W) for conditioning.
        Returns:
            logits: (B, num_embeddings, H, W).
        """
        x = F.one_hot(input_indices, self.d).permute(0, 3, 1, 2).float()
        if cond is not None:
            cond = cond.float()
        out = self.ini_conv(x)
        batch, _, height, width = out.shape
        background = self.background.expand(batch, -1, -1, -1)
        for block in self.blocks:
            out = block(out, background=background, cond=cond)
        out = self.upsample(self.non_linearity(out))
        out = self.out(self.non_linearity(out))
        return out

    def loss(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> OrderedDict[str, torch.Tensor]:
        """NLL (cross-entropy) over the code indices."""
        logits = self.forward(x, cond=cond)
        nll = F.cross_entropy(logits, x, reduction=reduction)
        return OrderedDict(loss=nll)

    @torch.no_grad()
    def sample(
        self,
        n: int,
        shape: tuple[int, int] | None = None,
        cond: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling. Returns (n, H, W) long tensor of indices."""
        shape = shape or self.latent_shape
        if device is None:
            device = next(self.parameters()).device
        samples = torch.zeros(n, *shape, dtype=torch.long, device=device)
        H, W = shape
        for r in range(H):
            for c in range(W):
                logits = self.forward(samples, cond=cond)[:, :, r, c]
                probs = F.softmax(logits, dim=1)
                samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples


# ---------------------------------------------------------------------------
# Config from MobileNetV2_8x_VQVAE
# ---------------------------------------------------------------------------


def get_pixelsnail_config_from_vqvae(vqvae: nn.Module) -> dict[str, Any]:
    """
    Build PixelSNAIL kwargs from a trained MobileNetV2_8x_VQVAE (frozen).
    Use this so PixelSNAIL matches the VQ-VAE latent grid and codebook.
    """
    from .autoencoders import MobileNetV2_8x_VQVAE

    if not isinstance(vqvae, MobileNetV2_8x_VQVAE):
        raise TypeError("vqvae must be an instance of MobileNetV2_8x_VQVAE")
    num_embeddings = vqvae.quantizer._num_embeddings
    # Latent spatial shape from encoder: (8, 16) for 64x128 spectrogram
    shape = (8, 16)
    return {
        "num_embeddings": num_embeddings,
        "shape": shape,
        "n_channels": 64,
        "n_block": 4,
        "n_res_block": 2,
        "dropout_p": 0.1,
        "cond_channels": None,
        "downsample": 1,
    }


def create_pixelsnail_for_vqvae(
    vqvae: nn.Module,
    n_channels: int = 64,
    n_block: int = 4,
    n_res_block: int = 2,
    dropout_p: float = 0.1,
    **kwargs: Any,
) -> PixelSNAIL:
    """Create a PixelSNAIL configured for the given MobileNetV2_8x_VQVAE."""
    config = get_pixelsnail_config_from_vqvae(vqvae)
    config.update(
        n_channels=n_channels,
        n_block=n_block,
        n_res_block=n_res_block,
        dropout_p=dropout_p,
        **kwargs,
    )
    return PixelSNAIL(**config)


if __name__ == "__main__":
    from .autoencoders import MobileNetV2_8x_VQVAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae = MobileNetV2_8x_VQVAE(num_embeddings=512, embedding_dim=32)
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad = False

    ps = create_pixelsnail_for_vqvae(vqvae).to(device)
    B, H, W = 2, 8, 16
    indices = torch.randint(0, 512, (B, H, W), device=device)
    logits = ps(indices)
    print("logits shape:", logits.shape)  # (2, 512, 8, 16)
    loss_dict = ps.loss(indices)
    print("loss:", loss_dict["loss"].item())
    samples = ps.sample(1, device=device)
    print("sample shape:", samples.shape)  # (1, 8, 16)
