"""
VectorQuantizer supports both:
- 3D input (B, T, D): flattened sequence of vectors (e.g. BasicVQVAE with T=1).
- 4D input (B, C, H, W): spatial latent (e.g. CNN / MobileNetV2_8x); quantizes per position and returns same shape.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Either (B, T, D) or (B, C, H, W) with D or C == embedding_dim.
        Returns:
            quantized: Same shape as inputs.
            loss: scalar VQ loss.
            perplexity: scalar.
        """
        if inputs.dim() == 4:
            # (B, C, H, W) -> (B, H*W, C)
            B, C, H, W = inputs.shape
            assert C == self._embedding_dim
            inputs_flat = inputs.permute(0, 2, 3, 1).reshape(B, H * W, C)
            quantized_flat, loss, perplexity = self._forward_3d(inputs_flat)
            quantized = quantized_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return quantized, loss, perplexity
        if inputs.dim() == 3:
            return self._forward_3d(inputs)
        raise ValueError(f"VectorQuantizer expects 3D (B, T, D) or 4D (B, C, H, W), got dim={inputs.dim()}")

    def _forward_3d(self, inputs: torch.Tensor):
        """Core logic: inputs (B, T, D), D == embedding_dim."""
        batch_size, time_size, latent_size = inputs.shape
        assert latent_size == self._embedding_dim

        flat_z = inputs.reshape(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self._embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(
            batch_size, time_size, latent_size
        )

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity

class VectorQuantizerEMA(nn.Module):
    """
    Vector quantizer with codebook updated by exponential moving average (no codebook gradient).
    Same interface as VectorQuantizer: (quantized, loss, perplexity). Loss is commitment only.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._decay = decay
        self._epsilon = epsilon

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()

        # EMA: cluster size (K,) and sum of vectors per cluster (K, D); codebook = sum / (size + eps)
        self.register_buffer("_ema_cluster_size", torch.ones(num_embeddings))
        self.register_buffer(
            "_ema_embedding_sum",
            self._embedding.weight.data.clone(),
        )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: Either (B, T, D) or (B, C, H, W) with D or C == embedding_dim.
        Returns:
            quantized: Same shape as inputs.
            loss: scalar VQ loss (commitment only).
            perplexity: scalar.
        """
        if inputs.dim() == 4:
            B, C, H, W = inputs.shape
            assert C == self._embedding_dim
            inputs_flat = inputs.permute(0, 2, 3, 1).reshape(B, H * W, C)
            quantized_flat, loss, perplexity = self._forward_3d(inputs_flat)
            quantized = quantized_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)
            return quantized, loss, perplexity
        if inputs.dim() == 3:
            return self._forward_3d(inputs)
        raise ValueError(
            f"VectorQuantizerEMA expects 3D (B, T, D) or 4D (B, C, H, W), got dim={inputs.dim()}"
        )

    def _forward_3d(self, inputs: torch.Tensor):
        """Core logic: inputs (B, T, D), D == embedding_dim."""
        batch_size, time_size, latent_size = inputs.shape
        assert latent_size == self._embedding_dim

        flat_z = inputs.reshape(-1, self._embedding_dim)

        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_z, self._embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(
            batch_size, time_size, latent_size
        )

        if self.training:
            # EMA update: cluster counts and sum of vectors per cluster
            n_batch = encodings.sum(dim=0)  # (K,)
            embed_sum_batch = torch.matmul(encodings.t(), flat_z)  # (K, D)
            one_minus_decay = 1.0 - self._decay
            self._ema_cluster_size.mul_(self._decay).add_(n_batch, alpha=one_minus_decay)
            self._ema_embedding_sum.mul_(self._decay).add_(embed_sum_batch, alpha=one_minus_decay)
            # Codebook = Laplace-smoothed mean
            n = self._ema_cluster_size.unsqueeze(1) + self._epsilon
            self._embedding.weight.data.copy_(self._ema_embedding_sum / n)

        # Commitment loss only (no codebook gradient in EMA)
        quantized_st = inputs + (quantized - inputs).detach()
        loss = self._commitment_cost * F.mse_loss(quantized_st, inputs)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized_st, loss, perplexity