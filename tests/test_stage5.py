"""Unit tests for Stage 5: Hard-Negative Isomer Reranker and Sampler."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
import numpy as np
from torch_geometric.data import Batch, Data

from src.models.reranker import CrossModalReranker
from src.data.hard_negative_dataset import HardNegativeIndex, triplet_collate_fn


def test_reranker_forward_and_loss():
    """Test CrossModalReranker dimensions and margin ranking loss."""
    embed_dim = 256
    phys_dim = 4
    batch_size = 8

    reranker = CrossModalReranker(embed_dim=embed_dim, physics_dim=phys_dim, hidden_dim=128)

    z_spec = torch.randn(batch_size, embed_dim)
    z_pos = torch.randn(batch_size, embed_dim)
    z_neg = torch.randn(batch_size, embed_dim)
    phys_pos = torch.randn(batch_size, phys_dim)
    phys_neg = torch.randn(batch_size, phys_dim)

    # Positive and negative forward passes
    s_pos = reranker(z_spec, z_pos, phys_pos)
    s_neg = reranker(z_spec, z_neg, phys_neg)

    assert s_pos.shape == (batch_size,)
    assert s_neg.shape == (batch_size,)

    # Margin loss
    loss = CrossModalReranker.margin_loss(s_pos, s_neg, margin=0.2)
    assert loss.dim() == 0
    assert loss.item() >= 0.0

    # Backprop
    loss.backward()
    for p in reranker.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_hard_negative_index():
    """Test HardNegativeIndex with synthetic molecules."""
    # 3 isomers of C6H12O2:
    # Ethyl butyrate: CCCC(=O)OCC
    # Hexanoic acid: CCCCCC(=O)O
    # Butyl acetate: CC(=O)OCCCC
    # And 1 isobar / diff formula:
    # Heptane (C7H16, MW ~100.2): CCCCCCC
    mols = ["mol_eb", "mol_ha", "mol_ba", "mol_hep"]
    smiles = {
        "mol_eb": "CCCC(=O)OCC",
        "mol_ha": "CCCCCC(=O)O",
        "mol_ba": "CC(=O)OCCCC",
        "mol_hep": "CCCCCCC",
    }

    index = HardNegativeIndex(mols, smiles)

    # All 3 esters must have formula C6H12O2
    assert index.formula_map["mol_eb"] == "C6H12O2"
    assert index.formula_map["mol_ha"] == "C6H12O2"
    assert index.formula_map["mol_ba"] == "C6H12O2"
    assert index.formula_map["mol_hep"] == "C7H16"

    # Sample isomer for mol_eb
    neg_mol, tier = index.sample_negative("mol_eb", target_tier="isomer")
    assert neg_mol in ["mol_ha", "mol_ba"]
    assert "isomer" in tier

    # Sample isobar for mol_eb (fallback to random if no non-isomer within 20 ppm)
    neg_mol, tier = index.sample_negative("mol_eb", target_tier="isobar")
    assert neg_mol in mols
    assert neg_mol != "mol_eb"

    # Sample random
    neg_mol, tier = index.sample_negative("mol_eb", target_tier="random")
    assert neg_mol != "mol_eb"


def test_triplet_collate_fn():
    """Test batching of triplet items."""
    def make_dummy_graph():
        x = torch.randn(5, 41)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        edge_attr = torch.randn(3, 10)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    sample1 = (
        torch.randn(1483),
        make_dummy_graph(),
        make_dummy_graph(),
        torch.tensor([0.1, 1.0, 0.3, 1.0]),
        torch.tensor([0.1, 1.0, 0.3, 1.0]),
        "mol1",
        "mol2",
        "isomer",
    )
    sample2 = (
        torch.randn(1483),
        make_dummy_graph(),
        make_dummy_graph(),
        torch.tensor([0.2, 1.0, 0.4, 1.0]),
        torch.tensor([1.5, 0.0, 0.4, 0.0]),
        "mol3",
        "mol4",
        "random",
    )

    batch = [sample1, sample2]
    specs, pos_graphs, neg_graphs, pos_phys, neg_phys, mol_pos, mol_neg, tiers = triplet_collate_fn(batch)

    assert specs.shape == (2, 1483)
    assert isinstance(pos_graphs, Batch)
    assert isinstance(neg_graphs, Batch)
    assert pos_graphs.num_graphs == 2
    assert neg_graphs.num_graphs == 2
    assert pos_phys.shape == (2, 4)
    assert neg_phys.shape == (2, 4)
    assert mol_pos == ["mol1", "mol3"]
    assert mol_neg == ["mol2", "mol4"]
    assert tiers == ["isomer", "random"]
