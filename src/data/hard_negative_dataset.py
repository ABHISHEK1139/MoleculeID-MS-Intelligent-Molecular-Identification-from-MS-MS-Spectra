"""Hard-Negative Dataset and Multi-Tier Sampler for Stage 5 Isomer Reranking.

Implements a 4-tier negative sampling hierarchy:
- Level 1: Random chemical negatives (maintains global manifold representation)
- Level 2: Precursor mass isobars (|Δppm| <= 20 ppm, distinct molecular formula)
- Level 3: Exact same-formula constitutional isomers (|Δppm| = 0.00)
- Level 4: High-scaffold-similarity constitutional isomers (top Tanimoto similarity)

Supports curriculum learning across training epochs and encodes physics feature
vectors ([ppm_error_norm, tier_weight, prec_mz_norm, formula_match]) for both
positive and negative candidates.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs, AllChem
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.core.adducts import neutral_mass
from src.data.spectrum_dataset import spectrum_to_coarse_bins
from src.data.cross_modal_dataset import CrossModalDataset


class HardNegativeIndex:
    """Precomputed chemical index over a molecule set for fast negative sampling."""

    def __init__(self, molecules: list[str], mol_smiles: dict[str, str]):
        self.molecules = molecules
        self.mol_smiles = mol_smiles

        self.formula_map: dict[str, str] = {}
        self.mass_map: dict[str, float] = {}
        self.fp_map: dict[str, Any] = {}
        self.formula_to_mols: dict[str, list[str]] = defaultdict(list)

        for m in self.molecules:
            smi = self.mol_smiles.get(m, "")
            mol_obj = Chem.MolFromSmiles(smi) if smi else None
            if mol_obj is not None:
                f = rdMolDescriptors.CalcMolFormula(mol_obj)
                mass = float(rdMolDescriptors.CalcExactMolWt(mol_obj))
                fp = AllChem.GetMorganFingerprintAsBitVect(mol_obj, 2, nBits=1024)
                self.formula_map[m] = f
                self.mass_map[m] = mass
                self.fp_map[m] = fp
                self.formula_to_mols[f].append(m)
            else:
                self.formula_map[m] = ""
                self.mass_map[m] = 0.0
                self.fp_map[m] = None

        # Sorted mass array for fast binary search of isobars (|Δppm| <= 20)
        self.sorted_mols = np.array(sorted(self.molecules, key=lambda m: self.mass_map.get(m, 0.0)))
        self.sorted_masses = np.array([self.mass_map.get(m, 0.0) for m in self.sorted_mols], dtype=np.float64)

        # Isomer groups cache
        self.isomer_groups = {f: mols for f, mols in self.formula_to_mols.items() if len(mols) > 1}

    def sample_negative(
        self,
        mol_pos: str,
        target_tier: str,
        scaffold_bias: float = 0.3,
        rng: np.random.Generator | None = None,
    ) -> tuple[str, str]:
        """Sample a negative molecule for `mol_pos` given target tier.

        Returns:
            (mol_neg, tier_sampled) where tier_sampled indicates actual tier used
            after fallback if needed.
        """
        if rng is None:
            rng = np.random.default_rng()

        f_pos = self.formula_map.get(mol_pos, "")
        mass_pos = self.mass_map.get(mol_pos, 0.0)

        # Tier: isomer
        if target_tier == "isomer":
            isomers = [m for m in self.formula_to_mols.get(f_pos, []) if m != mol_pos]
            if len(isomers) > 0:
                if len(isomers) == 1 or rng.random() > scaffold_bias or self.fp_map.get(mol_pos) is None:
                    return rng.choice(isomers), "isomer"
                # Select high-scaffold isomer via Tanimoto
                pos_fp = self.fp_map[mol_pos]
                sims = [DataStructs.TanimotoSimilarity(pos_fp, self.fp_map[m]) if self.fp_map.get(m) is not None else 0.0 for m in isomers]
                best_idx = int(np.argmax(sims))
                return isomers[best_idx], "isomer_scaffold"
            # Fallback to isobar
            target_tier = "isobar"

        # Tier: isobar (|Δppm| <= 20 ppm, distinct formula)
        if target_tier == "isobar" and mass_pos > 0:
            delta = mass_pos * 20e-6
            left = int(np.searchsorted(self.sorted_masses, mass_pos - delta))
            right = int(np.searchsorted(self.sorted_masses, mass_pos + delta))
            candidates = [
                self.sorted_mols[i] for i in range(left, right)
                if self.sorted_mols[i] != mol_pos and self.formula_map.get(self.sorted_mols[i]) != f_pos
            ]
            if len(candidates) > 0:
                return rng.choice(candidates), "isobar"
            # Fallback to random
            target_tier = "random"

        # Tier: random
        cand = mol_pos
        for _ in range(10):
            cand = rng.choice(self.molecules)
            if cand != mol_pos:
                break
        return cand, "random"


class HardNegativeDataset(Dataset):
    """Dataset yielding (spec_tensor, pos_graph, neg_graph, pos_phys, neg_phys, mol_pos, mol_neg, tier)."""

    def __init__(
        self,
        base_dataset: CrossModalDataset,
        index: HardNegativeIndex | None = None,
        seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.molecules = base_dataset.selected_mols
        self.mol_smiles = base_dataset.mol_smiles
        self.graph_cache = base_dataset.graph_cache
        self.samples = base_dataset.samples

        # Initialize chemical index if not provided
        self.index = index if index is not None else HardNegativeIndex(self.molecules, self.mol_smiles)
        self.rng = np.random.default_rng(seed)

        # Default curriculum probabilities: [p_random, p_isobar, p_isomer]
        self.tier_probs = [0.40, 0.30, 0.30]
        self.scaffold_bias = 0.20
        self.epoch = 1

    def set_curriculum(self, epoch: int, curriculum_schedule: dict[str, Any] | None = None) -> None:
        """Update negative sampling tier probabilities based on training epoch."""
        self.epoch = epoch
        if curriculum_schedule is None:
            # Standard 3-phase curriculum
            if epoch <= 5:
                self.tier_probs = [0.40, 0.30, 0.30]
                self.scaffold_bias = 0.20
            elif epoch <= 12:
                self.tier_probs = [0.20, 0.30, 0.50]
                self.scaffold_bias = 0.35
            else:
                self.tier_probs = [0.10, 0.20, 0.70]
                self.scaffold_bias = 0.50
        else:
            for phase_name, cfg in curriculum_schedule.items():
                min_ep = cfg.get("min_epoch", 1)
                max_ep = cfg.get("max_epoch", 999)
                if min_ep <= epoch <= max_ep:
                    self.tier_probs = cfg.get("probs", [0.2, 0.3, 0.5])
                    self.scaffold_bias = cfg.get("scaffold_bias", 0.3)
                    break

    def __len__(self) -> int:
        return len(self.samples)

    def _compute_physics_features(
        self,
        cand_mol: str,
        spec_info: dict[str, Any],
        is_positive: bool,
    ) -> torch.Tensor:
        """Compute physics feature vector: [ppm_error_norm, tier_weight, prec_mz_norm, formula_match]."""
        cand_mass = self.index.mass_map.get(cand_mol, 0.0)
        cand_formula = self.index.formula_map.get(cand_mol, "")

        prec_mz = float(spec_info.get("precursor_mz", 0.0))
        adduct = spec_info.get("adduct", None) or "[M+H]+"
        calc_neutral = neutral_mass(prec_mz, adduct)

        if calc_neutral is not None and calc_neutral > 0 and cand_mass > 0:
            ppm_error = abs(cand_mass - calc_neutral) / calc_neutral * 1e6
        else:
            ppm_error = 0.0 if is_positive else 50.0

        # Normalized features
        ppm_norm = min(ppm_error / 20.0, 3.0)  # 1.0 at 20 ppm, 2.5 at 50 ppm
        tier_weight = 1.0 if ppm_error <= 20.0 else (0.5 if ppm_error <= 50.0 else 0.0)
        prec_norm = prec_mz / 1000.0

        # Formula match flag (1.0 if positive or exact formula match to ground truth)
        formula_match = 1.0 if is_positive else (1.0 if "isomer" in cand_formula else 0.0)

        return torch.tensor([ppm_norm, tier_weight, prec_norm, formula_match], dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Data, Data, torch.Tensor, torch.Tensor, str, str, str]:
        spec_info, mol_pos = self.samples[idx]

        # 1. Binned spectrum tensor
        binned = spectrum_to_coarse_bins(spec_info["mz"], spec_info["intensity"])
        meta = np.array([
            spec_info["precursor_mz"] / 1000.0,
            spec_info["ion_mode"],
            spec_info["ce"] / 100.0,
        ], dtype=np.float32)
        spec_features = np.concatenate([binned, meta])
        spec_tensor = torch.from_numpy(spec_features)

        # 2. Positive graph
        pos_graph = self.graph_cache[mol_pos]

        # 3. Sample negative based on curriculum
        tier_choice = self.rng.choice(["random", "isobar", "isomer"], p=self.tier_probs)
        mol_neg, tier_sampled = self.index.sample_negative(
            mol_pos=mol_pos,
            target_tier=tier_choice,
            scaffold_bias=self.scaffold_bias,
            rng=self.rng,
        )
        # Ensure mol_neg has a valid graph in cache distinct from pos_graph
        attempts = 0
        while mol_neg not in self.graph_cache and attempts < 5:
            mol_neg, tier_sampled = self.index.sample_negative(
                mol_pos=mol_pos,
                target_tier=tier_choice,
                scaffold_bias=self.scaffold_bias,
                rng=self.rng,
            )
            attempts += 1
        if mol_neg not in self.graph_cache:
            valid_negs = [m for m in self.graph_cache.keys() if m != mol_pos]
            mol_neg = self.rng.choice(valid_negs) if valid_negs else mol_pos
        neg_graph = self.graph_cache[mol_neg]

        # 4. Physics features
        pos_phys = self._compute_physics_features(mol_pos, spec_info, is_positive=True)
        neg_phys = self._compute_physics_features(mol_neg, spec_info, is_positive=False)

        # Adjust negative formula match flag based on whether mol_neg shares formula with mol_pos
        if self.index.formula_map.get(mol_neg) == self.index.formula_map.get(mol_pos):
            neg_phys[3] = 1.0
            # For exact same-formula isomer, mass error is identical to positive
            neg_phys[0] = pos_phys[0]
            neg_phys[1] = pos_phys[1]
        else:
            neg_phys[3] = 0.0

        return spec_tensor, pos_graph, neg_graph, pos_phys, neg_phys, mol_pos, mol_neg, tier_sampled


def triplet_collate_fn(
    batch: list[tuple[torch.Tensor, Data, Data, torch.Tensor, torch.Tensor, str, str, str]]
) -> tuple[torch.Tensor, Batch, Batch, torch.Tensor, torch.Tensor, list[str], list[str], list[str]]:
    """Collate function for triplet hard-negative DataLoader."""
    specs = torch.stack([item[0] for item in batch], dim=0)
    pos_graphs = Batch.from_data_list([item[1] for item in batch])
    neg_graphs = Batch.from_data_list([item[2] for item in batch])
    pos_phys = torch.stack([item[3] for item in batch], dim=0)
    neg_phys = torch.stack([item[4] for item in batch], dim=0)
    mol_pos = [item[5] for item in batch]
    mol_neg = [item[6] for item in batch]
    tiers = [item[7] for item in batch]
    return specs, pos_graphs, neg_graphs, pos_phys, neg_phys, mol_pos, mol_neg, tiers
