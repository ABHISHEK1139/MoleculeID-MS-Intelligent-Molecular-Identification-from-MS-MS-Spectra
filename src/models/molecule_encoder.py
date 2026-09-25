"""Molecular Graph Neural Network (GNN) for cross-modal spectrum-structure retrieval.

Uses GINE (Graph Isomorphism Network with Edge Features) to map 2D molecular graphs
into the same 256-dimensional L2-normalized spherical embedding space as the
SpectrumEncoder (Stage 2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

from src.data.mol_graph import ATOM_FDIM, BOND_FDIM


class GINEResBlock(nn.Module):
    """Residual GINE layer with edge feature injection."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.conv = GINEConv(mlp, edge_dim=hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr=edge_attr)
        h = self.bn(h)
        return F.gelu(h + x)  # Residual connection


class MoleculeGNN(nn.Module):
    """Molecular GNN encoder: 2D molecular graph → 256-D L2-normalized embedding.

    Args:
        node_dim: Input atom feature dimension (default: ATOM_FDIM = 41).
        edge_dim: Input bond feature dimension (default: BOND_FDIM = 10).
        hidden_dim: Internal conv channel dimension (default: 128).
        embed_dim: Output embedding dimension (default: 256, matching SpectrumEncoder).
        n_layers: Number of GINE residual blocks (default: 4).
        dropout: Dropout probability in MLPs.
    """

    def __init__(
        self,
        node_dim: int = ATOM_FDIM,
        edge_dim: int = BOND_FDIM,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Input projections
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        # Message passing layers
        self.layers = nn.ModuleList([
            GINEResBlock(hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Dual pooling: sum (captures molecular size/composition) + mean (captures substructure density)
        pool_dim = hidden_dim * 2

        # Projection head: pool_dim → embed_dim (L2 normalized)
        self.projection = nn.Sequential(
            nn.Linear(pool_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, batch: Batch | Data) -> torch.Tensor:
        """Forward pass.

        Args:
            batch: PyTorch Geometric Batch or Data object.

        Returns:
            (B, embed_dim) L2-normalized molecule embeddings.
        """
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_vec = getattr(batch, "batch", None)

        if batch_vec is None:
            batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Handle empty graph edge cases
        if edge_index.size(1) == 0:
            edge_attr = torch.zeros(0, self.hidden_dim, device=x.device)
            h = self.node_proj(x)
        else:
            h = self.node_proj(x)
            e = self.edge_proj(edge_attr)
            for layer in self.layers:
                h = layer(h, edge_index, e)

        # Multi-scale global pooling
        h_sum = global_add_pool(h, batch_vec)
        h_mean = global_mean_pool(h, batch_vec)
        h_pool = torch.cat([h_sum, h_mean], dim=1)  # (B, hidden_dim * 2)

        # Projection to cross-modal space
        z = self.projection(h_pool)
        z = F.normalize(z, p=2, dim=-1)
        return z

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
