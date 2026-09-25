"""Cross-Modal Reranker for Stage 5 MS/MS Molecule Identification.

Fuses high-dimensional neural representations (spectrum 1D-CNN + molecule GNN)
with low-dimensional physical consistency signals (ppm mass error, adduct tier,
precursor m/z, formula matching) into a calibrated scalar compatibility score.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalReranker(nn.Module):
    """Pairwise neural reranker scoring compatibility between spectrum and candidate structure.

    Features passed into MLP:
    1. z_spec: (B, embed_dim) spectrum representation from 1D-CNN.
    2. z_mol: (B, embed_dim) candidate molecule graph representation from GINE.
    3. Element-wise product: z_spec * z_mol (captures aligned latent dimensions).
    4. Absolute difference: |z_spec - z_mol| (captures dimensional discrepancy).
    5. Physics features: (B, physics_dim) [ppm_error_norm, tier_weight, prec_mz_norm, formula_match].

    Total input dimension = embed_dim * 4 + physics_dim (e.g. 256 * 4 + 4 = 1,028).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        physics_dim: int = 4,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.physics_dim = physics_dim
        self.hidden_dim = hidden_dim

        in_dim = embed_dim * 4 + physics_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z_spec: torch.Tensor,
        z_mol: torch.Tensor,
        phys_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing compatibility score S(s, m).

        Args:
            z_spec: (B, embed_dim) or broadcastable (1, embed_dim).
            z_mol: (B, embed_dim) candidate molecular graph embeddings.
            phys_feats: (B, physics_dim) normalized physics vectors.

        Returns:
            (B,) compatibility scores (higher means better match).
        """
        if z_spec.dim() == 2 and z_spec.size(0) == 1 and z_mol.size(0) > 1:
            z_spec = z_spec.expand(z_mol.size(0), -1)

        prod = z_spec * z_mol
        diff = torch.abs(z_spec - z_mol)
        x = torch.cat([z_spec, z_mol, prod, diff, phys_feats], dim=-1)
        score = self.mlp(x).squeeze(-1)
        return score

    @staticmethod
    def margin_loss(
        s_pos: torch.Tensor,
        s_neg: torch.Tensor,
        margin: float = 0.2,
    ) -> torch.Tensor:
        """Pairwise margin ranking loss: max(0, margin - s_pos + s_neg)."""
        target = torch.ones_like(s_pos)
        return F.margin_ranking_loss(s_pos, s_neg, target=target, margin=margin)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
