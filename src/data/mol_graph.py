"""Molecular graph featurizer for PyTorch Geometric GNN models.

Converts SMILES strings into PyG Data graphs with rich atom and bond features.
Includes in-memory LRU/dict caching to eliminate duplicate RDKit calls.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

# ── Feature vocabulary ───────────────────────────────────────────────────────
COMMON_ATOMIC_NUMS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, B, C, N, O, F, Si, P, S, Cl, Br, I
DEGREES = [0, 1, 2, 3, 4, 5, 6]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUM_HS = [0, 1, 2, 3, 4]

# Atom feature dim:
# atomic_num(13) + deg(8) + charge(6) + hyb(6) + aromatic(1) + num_hs(6) + in_ring(1) = 41
ATOM_FDIM = (
    (len(COMMON_ATOMIC_NUMS) + 1)
    + (len(DEGREES) + 1)
    + (len(FORMAL_CHARGES) + 1)
    + (len(HYBRIDIZATIONS) + 1)
    + 1
    + (len(NUM_HS) + 1)
    + 1
)
BOND_FDIM = 4 + 1 + 1 + 4  # type(4) + conjugated(1) + in_ring(1) + stereo(4) = 10

_GRAPH_CACHE: dict[str, Data] = {}


def _one_hot(val: Any, allowed: list[Any]) -> list[float]:
    """Return a one-hot encoding list with an extra fallback 'other' bin."""
    vec = [0.0] * (len(allowed) + 1)
    if val in allowed:
        vec[allowed.index(val)] = 1.0
    else:
        vec[-1] = 1.0
    return vec


def atom_to_features(atom: Chem.Atom) -> list[float]:
    """Extract atom feature vector."""
    feats: list[float] = []
    # 1. Atomic number (13 dims)
    feats.extend(_one_hot(atom.GetAtomicNum(), COMMON_ATOMIC_NUMS))
    # 2. Degree (8 dims)
    feats.extend(_one_hot(atom.GetTotalDegree(), DEGREES))
    # 3. Formal charge (6 dims)
    feats.extend(_one_hot(atom.GetFormalCharge(), FORMAL_CHARGES))
    # 4. Hybridization (6 dims)
    feats.extend(_one_hot(atom.GetHybridization(), HYBRIDIZATIONS))
    # 5. Aromaticity (1 dim)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    # 6. Hydrogen count (6 dims)
    feats.extend(_one_hot(atom.GetTotalNumHs(), NUM_HS))
    # 7. In ring (1 dim)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    return feats


def bond_to_features(bond: Chem.Bond) -> list[float]:
    """Extract 10-dimensional bond feature vector."""
    feats: list[float] = []
    # 1. Bond type (4 dims)
    btype = bond.GetBondType()
    feats.extend([
        1.0 if btype == Chem.rdchem.BondType.SINGLE else 0.0,
        1.0 if btype == Chem.rdchem.BondType.DOUBLE else 0.0,
        1.0 if btype == Chem.rdchem.BondType.TRIPLE else 0.0,
        1.0 if btype == Chem.rdchem.BondType.AROMATIC else 0.0,
    ])
    # 2. Conjugation (1 dim)
    feats.append(1.0 if bond.GetIsConjugated() else 0.0)
    # 3. In ring (1 dim)
    feats.append(1.0 if bond.IsInRing() else 0.0)
    # 4. Stereo (4 dims)
    stereo = bond.GetStereo()
    feats.extend([
        1.0 if stereo == Chem.rdchem.BondStereo.STEREONONE else 0.0,
        1.0 if stereo == Chem.rdchem.BondStereo.STEREOANY else 0.0,
        1.0 if stereo == Chem.rdchem.BondStereo.STEREOZ else 0.0,
        1.0 if stereo == Chem.rdchem.BondStereo.STEREOE else 0.0,
    ])
    return feats


def smiles_to_graph(smiles: str, cache: bool = True) -> Data | None:
    """Convert SMILES string into a PyTorch Geometric Data graph.

    Args:
        smiles: Valid SMILES string.
        cache: If True, check and populate the in-memory cache.

    Returns:
        PyG Data object with x (N, 41), edge_index (2, 2E), edge_attr (2E, 10),
        or None if parsing fails.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    if cache and smiles in _GRAPH_CACHE:
        return _GRAPH_CACHE[smiles]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features
    atom_feats = [atom_to_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float32)

    # Edge features (bidirectional)
    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bfeats = bond_to_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([bfeats, bfeats])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        # Isolated atoms / single-atom molecule
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if cache:
        _GRAPH_CACHE[smiles] = data

    return data
