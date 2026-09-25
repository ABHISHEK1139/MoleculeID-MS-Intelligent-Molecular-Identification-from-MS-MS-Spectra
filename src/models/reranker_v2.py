"""Stage 5 v2 Cross-Modal Reranker with Evidence and Structure Fusion.

Fuses:
1. Neural representations:
   - z_spec (256-D from 1D-CNN)
   - z_mol (256-D from MoleculeGNN)
   - z_spec * z_mol (element-wise alignment)
   - |z_spec - z_mol| (element-wise discrepancy)
2. Structural similarity:
   - Morgan fingerprint Tanimoto similarity to top reference match (1-D)
3. Physical and Experimental Evidence Features (10-D):
   - [0] best_ext_cosine in [0, 1.0]
   - [1] matched_peaks_norm: min(1.0, matched_peaks / 15.0)
   - [2] peak_to_cosine_ratio: matched_peaks / (1.0 + 15.0 * best_ext_cosine) (flags false analogs!)
   - [3] ce_agreement: exp(-ce_diff / 20.0) if finite else 0.60
   - [4] multiplicity_norm: min(1.0, log1p(n_supporting) / log(6))
   - [5] source_corroboration: 1.0 if source_count >= 2 else 0.0
   - [6] ppm_error_norm: min(1.0, ppm_error / 20.0)
   - [7] tier_weight: 1.0 (tier 1) or 0.5 (tier 2)
   - [8] prec_mz_norm: prec_mz / 1000.0
   - [9] formula_match: 1.0 if exact formula isomer else 0.0

Total input dimension: 256 * 4 + 1 + 10 = 1,035.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalRerankerV2(nn.Module):
    """Stage 5 v2 Cross-Modal Reranker combining structural, physical, and spectral evidence."""

    def __init__(
        self,
        embed_dim: int = 256,
        evidence_dim: int = 10,
        hidden_dim: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.evidence_dim = evidence_dim
        self.hidden_dim = hidden_dim

        # Input: z_spec(256) + z_mol(256) + prod(256) + diff(256) + morgan_sim(1) + evidence(10) = 1,035
        in_dim = embed_dim * 4 + 1 + evidence_dim

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
        morgan_sim: torch.Tensor,
        evidence_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass computing compatibility score S(q, c).

        Args:
            z_spec: (B, embed_dim) or (1, embed_dim)
            z_mol: (B, embed_dim)
            morgan_sim: (B, 1) or (B,)
            evidence_feats: (B, evidence_dim)

        Returns:
            (B,) compatibility scalar scores (higher = more likely ground truth).
        """
        if z_spec.dim() == 2 and z_spec.size(0) == 1 and z_mol.size(0) > 1:
            z_spec = z_spec.expand(z_mol.size(0), -1)

        if morgan_sim.dim() == 1:
            morgan_sim = morgan_sim.unsqueeze(-1)

        prod = z_spec * z_mol
        diff = torch.abs(z_spec - z_mol)

        x = torch.cat([z_spec, z_mol, prod, diff, morgan_sim, evidence_feats], dim=-1)
        score = self.mlp(x).squeeze(-1)
        return score

    @staticmethod
    def margin_loss(
        s_pos: torch.Tensor,
        s_neg: torch.Tensor,
        margin: float = 0.20,
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
