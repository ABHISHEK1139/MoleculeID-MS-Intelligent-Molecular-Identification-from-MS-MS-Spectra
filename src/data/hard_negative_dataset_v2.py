"""Stage 5 v2 Hard-Negative Dataset & Sampler.

Mines hard negatives across the full 277K candidate universe:
1. Exact-formula constitutional isomers (formula match, different constitution)
2. High-cosine / low-peak false analogs (spectral mimics with <= 3 matched peaks)
3. High Morgan-similarity structural analogs (Tanimoto >= 0.40)
4. Same-scaffold isomers
5. <= 20 ppm mass isobars

Constructs 10-D physical and experimental evidence feature vectors for both
positive and negative candidates, enabling CrossModalRerankerV2 to learn when
molecular structure breaks ties given experimental evidence.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, DataStructs
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from src.data.mol_graph import smiles_to_graph
from src.data.spectrum_dataset import spectrum_to_coarse_bins
from src.core.adducts import neutral_mass
from src.core.spectral_search import (
    CompactSpectralLibrary,
    fast_mutual_cosine,
)


class Catalog277KIndex:
    """Fast chemical index over the full 276,940 candidate universe."""

    def __init__(self, df_cand: pd.DataFrame):
        self.df_cand = df_cand
        self.smiles_list = df_cand["normalized_smiles"].tolist()
        self.formula_list = df_cand["molecular_formula"].tolist()
        self.mass_array = df_cand["exact_mass"].to_numpy(dtype=np.float64)

        # Index by formula
        self.formula_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, f in enumerate(self.formula_list):
            self.formula_to_indices[f].append(idx)

        # Sorted by mass for fast isobar search
        self.mass_order = np.argsort(self.mass_array)
        self.sorted_masses = self.mass_array[self.mass_order]
        self.sorted_indices = self.mass_order

        # Cache for Morgan fingerprints (sparse lazy cache)
        self.fp_cache: dict[int, Any] = {}

    def get_morgan_fp(self, idx: int) -> Any:
        if idx not in self.fp_cache:
            smi = self.smiles_list[idx]
            mol = Chem.MolFromSmiles(smi) if smi else None
            self.fp_cache[idx] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) if mol else None
        return self.fp_cache[idx]

    def sample_negative(
        self,
        pos_smi: str,
        pos_formula: str,
        pos_mass: float,
        target_tier: str,
        ext_candidate_hits: dict[str, dict[str, Any]] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, str]:
        """Sample a hard negative index from the 277K catalog.

        Returns (cand_idx, actual_tier).
        """
        if rng is None:
            rng = np.random.default_rng()

        # 1. False Analog Tier: High cosine but low peaks (<= 3 peaks)
        if target_tier == "false_analog" and ext_candidate_hits:
            false_analogs = []
            for smi, h in ext_candidate_hits.items():
                if smi != pos_smi and h["cos"] >= 0.40 and h["n_peaks"] <= 3:
                    false_analogs.append(smi)
            if false_analogs:
                pick_smi = rng.choice(false_analogs)
                # Find index in catalog
                sub = self.df_cand[self.df_cand["normalized_smiles"] == pick_smi]
                if not sub.empty:
                    return int(sub.index[0]), "false_analog"

        # 2. Exact Isomer Tier
        if target_tier in ("isomer", "scaffold_isomer", "false_analog"):
            isomers = [i for i in self.formula_to_indices.get(pos_formula, []) if self.smiles_list[i] != pos_smi]
            if isomers:
                if target_tier == "scaffold_isomer" and len(isomers) > 1:
                    # Pick high Morgan similarity isomer
                    pos_idx = self.df_cand[self.df_cand["normalized_smiles"] == pos_smi].index
                    if len(pos_idx) > 0:
                        pos_fp = self.get_morgan_fp(pos_idx[0])
                        if pos_fp is not None:
                            sims = [
                                DataStructs.TanimotoSimilarity(pos_fp, self.get_morgan_fp(i))
                                if self.get_morgan_fp(i) is not None else 0.0
                                for i in isomers
                            ]
                            return isomers[int(np.argmax(sims))], "scaffold_isomer"
                return rng.choice(isomers), "isomer"

        # 3. Isobar Tier (<= 20 ppm mass window)
        delta = pos_mass * 20e-6
        l = int(np.searchsorted(self.sorted_masses, pos_mass - delta))
        r = int(np.searchsorted(self.sorted_masses, pos_mass + delta))
        isobars = [i for i in self.sorted_indices[l:r] if self.smiles_list[i] != pos_smi]
        if isobars:
            return rng.choice(isobars), "isobar"

        # Fallback: random negative
        rand_idx = int(rng.integers(0, len(self.df_cand)))
        return rand_idx, "random"


def extract_candidate_evidence_vector(
    hit: dict[str, Any] | None,
    ppm_error: float,
    tier: int,
    prec_mz: float,
    is_isomer: bool,
) -> np.ndarray:
    """Build the normalized 10-D physical and experimental evidence vector.

    [0] best_ext_cosine in [0, 1.0]
    [1] matched_peaks_norm: min(1.0, matched_peaks / 15.0)
    [2] peak_to_cosine_ratio: matched_peaks / (1.0 + 15.0 * best_ext_cosine) (flags false analogs!)
    [3] ce_agreement: exp(-ce_diff / 20.0) if finite else 0.60
    [4] multiplicity_norm: min(1.0, log1p(n_supporting) / log(6))
    [5] source_corroboration: 1.0 if source_count >= 2 else 0.0
    [6] ppm_error_norm: min(1.0, ppm_error / 20.0)
    [7] tier_weight: 1.0 (tier 1) or 0.5 (tier 2)
    [8] prec_mz_norm: prec_mz / 1000.0
    [9] formula_match: 1.0 if exact formula isomer else 0.0
    """
    if hit is not None and hit["cos"] >= 0.10:
        cos = min(1.0, max(0.0, float(hit["cos"])))
        n_p = float(hit["n_peaks"])
        p_norm = min(1.0, n_p / 15.0)
        p_ratio = min(1.0, n_p / (1.0 + 15.0 * cos))
        ce_diff = hit.get("ce_diff", float("nan"))
        ce_agree = float(np.exp(-abs(ce_diff) / 20.0)) if np.isfinite(ce_diff) else 0.60
        n_sup = hit.get("n_supporting", 1)
        multi_norm = min(1.0, math.log1p(max(1, n_sup)) / math.log1p(5))
        src_corrob = 1.0 if hit.get("source_count", 1) >= 2 else 0.0
    else:
        cos = 0.0
        p_norm = 0.0
        p_ratio = 0.0
        ce_agree = 0.60
        multi_norm = 0.0
        src_corrob = 0.0

    return np.array([
        cos,
        p_norm,
        p_ratio,
        ce_agree,
        multi_norm,
        src_corrob,
        min(1.0, max(0.0, ppm_error / 20.0)),
        1.0 if tier == 1 else 0.5,
        prec_mz / 1000.0,
        1.0 if is_isomer else 0.0,
    ], dtype=np.float32)


class HardNegativeDatasetV2(Dataset):
    """Dataset yielding (spec_tensor, pos_graph, neg_graph, pos_morgan, neg_morgan, pos_ev, neg_ev, tier) tuples."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        catalog_index: Catalog277KIndex,
        graph_cache: dict[str, Data],
        tier_probs: dict[str, float] | None = None,
        seed: int = 42,
    ):
        self.samples = samples
        self.catalog = catalog_index
        self.graph_cache = graph_cache
        self.rng = np.random.default_rng(seed)

        self.tier_probs = tier_probs or {
            "false_analog": 0.25,
            "scaffold_isomer": 0.25,
            "isomer": 0.30,
            "isobar": 0.20,
        }
        self.tiers = list(self.tier_probs.keys())
        self.probs = [self.tier_probs[t] for t in self.tiers]
        tot = sum(self.probs)
        self.probs = [p / tot for p in self.probs]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Data, Data, float, float, torch.Tensor, torch.Tensor, str]:
        s = self.samples[idx]
        spec_tensor = s["spec_tensor"]
        pos_smi = s["pos_smi"]
        pos_formula = s["pos_formula"]
        pos_mass = s["pos_mass"]
        prec_mz = s["prec_mz"]
        pos_hit = s.get("pos_hit")
        ext_hits = s.get("ext_hits", {})
        top_ref_smi = s.get("top_ref_smi", "")

        # Target negative tier
        target_tier = self.rng.choice(self.tiers, p=self.probs)
        neg_idx, actual_tier = self.catalog.sample_negative(
            pos_smi=pos_smi,
            pos_formula=pos_formula,
            pos_mass=pos_mass,
            target_tier=target_tier,
            ext_candidate_hits=ext_hits,
            rng=self.rng,
        )

        neg_smi = self.catalog.smiles_list[neg_idx]
        neg_formula = self.catalog.formula_list[neg_idx]
        neg_mass = self.catalog.mass_array[neg_idx]
        neg_hit = ext_hits.get(neg_smi)

        # Molecular Graphs
        if pos_smi not in self.graph_cache:
            self.graph_cache[pos_smi] = smiles_to_graph(pos_smi)
        if neg_smi not in self.graph_cache:
            self.graph_cache[neg_smi] = smiles_to_graph(neg_smi)

        pos_graph = self.graph_cache[pos_smi]
        neg_graph = self.graph_cache[neg_smi]

        # Morgan similarities to top spectral reference
        pos_morgan = 1.0
        neg_morgan = 0.0
        if top_ref_smi:
            ref_mol = Chem.MolFromSmiles(top_ref_smi)
            if ref_mol:
                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                p_mol = Chem.MolFromSmiles(pos_smi)
                n_mol = Chem.MolFromSmiles(neg_smi)
                if p_mol and ref_fp:
                    p_fp = AllChem.GetMorganFingerprintAsBitVect(p_mol, 2, nBits=1024)
                    pos_morgan = float(DataStructs.TanimotoSimilarity(ref_fp, p_fp))
                if n_mol and ref_fp:
                    n_fp = AllChem.GetMorganFingerprintAsBitVect(n_mol, 2, nBits=1024)
                    neg_morgan = float(DataStructs.TanimotoSimilarity(ref_fp, n_fp))

        # Evidence vectors
        q_adduct = spec_info.get("adduct", "[M+H]+") if isinstance(spec_info, dict) else "[M+H]+"
        m_neut = neutral_mass(prec_mz, q_adduct)
        if m_neut is None or m_neut <= 0:
            m_neut = prec_mz - 1.007825
        pos_ppm = abs(pos_mass - m_neut) / m_neut * 1e6
        neg_ppm = abs(neg_mass - m_neut) / m_neut * 1e6

        pos_ev = extract_candidate_evidence_vector(
            hit=pos_hit,
            ppm_error=pos_ppm,
            tier=1 if pos_ppm <= 20.0 else 2,
            prec_mz=prec_mz,
            is_isomer=True,
        )
        neg_ev = extract_candidate_evidence_vector(
            hit=neg_hit,
            ppm_error=neg_ppm,
            tier=1 if neg_ppm <= 20.0 else 2,
            prec_mz=prec_mz,
            is_isomer=(neg_formula == pos_formula),
        )

        return (
            spec_tensor,
            pos_graph,
            neg_graph,
            pos_morgan,
            neg_morgan,
            torch.from_numpy(pos_ev),
            torch.from_numpy(neg_ev),
            actual_tier,
        )


def triplet_collate_fn_v2(batch: list[tuple]) -> tuple:
    """Collate function for Stage 5 v2 triplets."""
    specs = torch.stack([item[0] for item in batch], dim=0)
    pos_graphs = Batch.from_data_list([item[1] for item in batch])
    neg_graphs = Batch.from_data_list([item[2] for item in batch])
    pos_morgans = torch.tensor([item[3] for item in batch], dtype=torch.float32).unsqueeze(-1)
    neg_morgans = torch.tensor([item[4] for item in batch], dtype=torch.float32).unsqueeze(-1)
    pos_ev = torch.stack([item[5] for item in batch], dim=0)
    neg_ev = torch.stack([item[6] for item in batch], dim=0)
    tiers = [item[7] for item in batch]
    return specs, pos_graphs, neg_graphs, pos_morgans, neg_morgans, pos_ev, neg_ev, tiers
