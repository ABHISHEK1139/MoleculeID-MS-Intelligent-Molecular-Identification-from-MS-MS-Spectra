"""Cross-modal Dataset pairing MS/MS spectra with 2D molecular graphs.

Provides strict molecule-disjoint splits: validation molecules are completely
absent from the training set, enabling rigorous Protocol C zero-reference evaluation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data

from src.core.config import TRAIN_PATH
from src.core.preprocessing import preprocess_variant, VariantConfig
from src.data.spectrum_dataset import (
    spectrum_to_coarse_bins,
    COARSE_N_BINS,
    _PREPROCESS_CFG,
)
from src.data.mol_graph import smiles_to_graph


class CrossModalDataset(Dataset):
    """Dataset yielding (spectrum_tensor, molecule_graph, inchikey) tuples.

    Args:
        train_path: Path to parquet dataset.
        subset_size: Number of unique molecules to load (None = all).
        split: "train" or "val".
        val_frac: Fraction of molecules held out for validation (molecule-disjoint).
        seed: Random seed for split partitioning.
        max_row_groups: Limit number of row groups to read.
    """

class CrossModalDataset(Dataset):
    """Dataset yielding (spectrum_tensor, molecule_graph, inchikey) tuples."""

    def __init__(
        self,
        split: str,
        molecules: list[str],
        mol_smiles: dict[str, str],
        mol_spectra: dict[str, list[dict[str, Any]]],
        graph_cache: dict[str, Data],
    ):
        self.split = split
        self.selected_mols = molecules
        self.mol_smiles = {m: mol_smiles[m] for m in molecules}
        self.graph_cache = graph_cache

        # Build samples list: (spec_dict, mol_inchikey)
        self.samples: list[tuple[dict[str, Any], str]] = []
        for mol in self.selected_mols:
            if mol not in self.graph_cache:
                continue
            for s in mol_spectra.get(mol, []):
                self.samples.append((s, mol))

        print(f"[CrossModalDataset:{split}] {len(self.selected_mols)} molecules, "
              f"{len(self.samples)} spectrum-graph pairs", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Data, str]:
        spec_info, mol = self.samples[idx]

        # 1. Binned spectrum feature tensor (1480 bins + 3 metadata = 1483)
        binned = spectrum_to_coarse_bins(spec_info["mz"], spec_info["intensity"])
        meta = np.array([
            spec_info["precursor_mz"] / 1000.0,
            spec_info["ion_mode"],
            spec_info["ce"] / 100.0,
        ], dtype=np.float32)
        spec_features = np.concatenate([binned, meta])
        spec_tensor = torch.from_numpy(spec_features)

        # 2. Molecular graph
        graph = self.graph_cache[mol]

        return spec_tensor, graph, mol


def cross_modal_collate_fn(batch: list[tuple[torch.Tensor, Data, str]]) -> tuple[torch.Tensor, Batch, list[str]]:
    """Collate function for PyTorch DataLoader."""
    specs = torch.stack([item[0] for item in batch], dim=0)
    graphs = Batch.from_data_list([item[1] for item in batch])
    mols = [item[2] for item in batch]
    return specs, graphs, mols


def create_cross_modal_datasets(
    train_path: str | Path = TRAIN_PATH,
    subset_size: int | None = 10_000,
    val_frac: float = 0.10,
    seed: int = 42,
    max_row_groups: int | None = None,
) -> tuple[CrossModalDataset, CrossModalDataset]:
    """Single-pass parquet loader creating disjoint train and validation datasets."""
    train_path = Path(train_path)
    pf = pq.ParquetFile(train_path)
    n_rg = pf.metadata.num_row_groups
    if max_row_groups is not None:
        n_rg = min(n_rg, max_row_groups)

    columns = [
        "ms2_mzs", "ms2_normalized_intensities",
        "precursor_mz", "ionization_mode", "collision_energy_ev",
        "inchikey", "normalized_smiles", "adduct",
    ]

    mol_smiles: dict[str, str] = {}
    mol_spectra: dict[str, list[dict[str, Any]]] = {}

    for rg_idx in range(n_rg):
        table = pf.read_row_group(rg_idx, columns=columns)
        df = table.to_pandas()

        mzs = df["ms2_mzs"].values
        intensities = df["ms2_normalized_intensities"].values
        precursor_mzs = df["precursor_mz"].values
        ces = df["collision_energy_ev"].values
        ion_modes = df["ionization_mode"].values
        inchikeys = df["inchikey"].values
        smiles_list = df["normalized_smiles"].values
        adduct_list = df["adduct"].values

        for mz_raw, int_raw, prec_raw, ce_raw, im_raw, ik, smi, add_raw in zip(
            mzs, intensities, precursor_mzs, ces, ion_modes, inchikeys, smiles_list, adduct_list
        ):
            if not isinstance(smi, str) or not smi:
                continue

            mz = np.asarray(mz_raw, dtype=np.float32)
            intensity = np.asarray(int_raw, dtype=np.float32)
            if mz.size < 3:
                continue

            precursor_mz = float(prec_raw) if (prec_raw is not None and not np.isnan(prec_raw)) else None
            mz_proc, int_proc = preprocess_variant(mz, intensity, precursor_mz, _PREPROCESS_CFG)
            if mz_proc.size < 3:
                continue

            mol = str(ik)
            if mol not in mol_smiles:
                mol_smiles[mol] = smi
                mol_spectra[mol] = []

            # Parse collision energy
            if ce_raw is None or (isinstance(ce_raw, float) and np.isnan(ce_raw)):
                ce_val = 0.0
            elif isinstance(ce_raw, (list, tuple, np.ndarray)):
                ce_arr = np.asarray(ce_raw, dtype=np.float32)
                ce_arr = ce_arr[np.isfinite(ce_arr)]
                ce_val = float(np.mean(ce_arr)) if ce_arr.size > 0 else 0.0
            else:
                try:
                    c_flt = float(ce_raw)
                    ce_val = c_flt if np.isfinite(c_flt) else 0.0
                except (ValueError, TypeError):
                    ce_val = 0.0

            ion_mode = 1.0 if "pos" in str(im_raw).lower() else 0.0
            adduct_str = str(add_raw) if (add_raw is not None and not (isinstance(add_raw, float) and np.isnan(add_raw))) else None

            mol_spectra[mol].append({
                "mz": mz_proc,
                "intensity": int_proc,
                "precursor_mz": precursor_mz or 0.0,
                "ce": ce_val,
                "ion_mode": ion_mode,
                "adduct": adduct_str,
            })

        if subset_size is not None and len(mol_smiles) >= subset_size:
            break

    # Partition molecules strictly disjoint
    all_unique_mols = sorted(list(mol_smiles.keys()))
    if subset_size is not None:
        all_unique_mols = all_unique_mols[:subset_size]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_unique_mols))
    n_val = max(1, int(len(all_unique_mols) * val_frac))

    val_set = set(all_unique_mols[i] for i in perm[:n_val])
    train_set = set(all_unique_mols[i] for i in perm[n_val:])

    # Featurize graphs once across all selected molecules
    graph_cache: dict[str, Data] = {}
    for mol in all_unique_mols:
        g = smiles_to_graph(mol_smiles[mol])
        if g is not None:
            graph_cache[mol] = g

    train_dataset = CrossModalDataset(
        split="train",
        molecules=sorted(list(train_set)),
        mol_smiles=mol_smiles,
        mol_spectra=mol_spectra,
        graph_cache=graph_cache,
    )
    val_dataset = CrossModalDataset(
        split="val",
        molecules=sorted(list(val_set)),
        mol_smiles=mol_smiles,
        mol_spectra=mol_spectra,
        graph_cache=graph_cache,
    )
    return train_dataset, val_dataset

