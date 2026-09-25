"""1D-CNN spectrum encoder for contrastive representation learning.

Architecture rationale:
- Input: 1483 features (1480 coarse bins @ 1.0 Da + 3 metadata features)
- The spectrum is treated as a 1D signal where spatial locality matters
  (nearby m/z peaks are often related: isotope patterns, neutral losses)
- Conv1D with increasing dilation captures both local and global patterns
- Global average pooling → projection head → L2-normalized embedding
- Output: 256-dim L2-normalized embedding for InfoNCE loss

Memory budget (RTX 3050, 4GB VRAM):
- Input: 1483 × batch_256 × 4 bytes ≈ 1.5 MB
- Model: ~2M parameters × 4 bytes ≈ 8 MB
- Activations: ~100 MB at batch_256
- Total: well within 4GB budget
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.spectrum_dataset import COARSE_N_BINS


class ResBlock1D(nn.Module):
    """Residual block with two Conv1D layers and optional downsampling."""

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.gelu(out + residual)


class SpectrumEncoder(nn.Module):
    """1D-CNN encoder: binned spectrum → L2-normalized embedding.

    Args:
        input_dim: Number of input features (bins + metadata).
        embed_dim: Output embedding dimension.
        hidden_channels: Number of channels in conv layers.
        n_blocks: Number of residual blocks.
        dropout: Dropout rate after global pooling.
    """

    def __init__(
        self,
        input_dim: int = COARSE_N_BINS + 3,
        embed_dim: int = 256,
        hidden_channels: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Stem: project 1 channel → hidden_channels
        # We separate the binned spectrum (1480) from metadata (3)
        self.bin_dim = COARSE_N_BINS
        self.meta_dim = input_dim - COARSE_N_BINS

        self.stem = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.MaxPool1d(4),  # 1480 → 370
        )

        # Residual blocks with increasing dilation
        blocks = []
        for i in range(n_blocks):
            dilation = 2 ** i  # 1, 2, 4, 8
            blocks.append(ResBlock1D(hidden_channels, kernel_size=5, dilation=dilation))
            if i < n_blocks - 1:
                blocks.append(nn.MaxPool1d(2))  # downsample between blocks
        self.res_blocks = nn.Sequential(*blocks)

        # Metadata MLP (3 → 64)
        self.meta_mlp = nn.Sequential(
            nn.Linear(self.meta_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # Projection head: pooled conv features + metadata → embedding
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels + 64, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, input_dim) concatenated binned spectrum + metadata.

        Returns:
            (B, embed_dim) L2-normalized embedding.
        """
        # Split input
        bins = x[:, :self.bin_dim]    # (B, 1480)
        meta = x[:, self.bin_dim:]    # (B, 3)

        # Conv path: (B, 1480) → (B, 1, 1480) → conv blocks → global pool
        h = bins.unsqueeze(1)  # (B, 1, 1480)
        h = self.stem(h)       # (B, C, 370)
        h = self.res_blocks(h) # (B, C, ~46) depending on pooling

        # Global average + max pooling (combines both for robustness)
        h_avg = h.mean(dim=2)   # (B, C)
        h_max = h.max(dim=2)[0] # (B, C)
        h_pool = h_avg + h_max  # (B, C) — sum is simple and effective

        # Metadata path
        m = self.meta_mlp(meta)  # (B, 64)

        # Combine and project
        combined = torch.cat([h_pool, m], dim=1)  # (B, C+64)
        emb = self.projection(combined)            # (B, embed_dim)

        # L2 normalize for cosine-based InfoNCE
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
