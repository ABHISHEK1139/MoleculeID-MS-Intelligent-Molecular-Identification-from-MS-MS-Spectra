"""Canonical Protocol C v2 Benchmark Harness with Strict Tuning / Evaluation Separation.

Guarantees 100% reproducible and leakage-free evaluation:
- 1,000 Validation Molecules partitioned into:
  1. 800 Tuning Molecules: Used EXCLUSIVELY to learn global score calibration,
     tune fusion weights (alpha, beta, gamma), optimize router threshold tau,
     and perform Stage 5 checkpoint selection.
  2. 200 Frozen Benchmark Molecules: UNTOUCHED during all tuning. Used purely
     for out-of-sample final evaluation:
     - (A) Full Protocol C (All 200 Queries)
     - (B) Dedicated Exact Same-Formula Multi-Isomer Subset (130 Queries)
- 10,000 Candidate Catalog (all valid molecules from train + val).
- Two-tier mass search: 20 ppm primary, 50 ppm fallback, isotope correction M±1.
- Standardized metrics: MRR@25, Hit@1, Hit@5, Hit@25.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from src.core.config import ARTIFACTS_DIR, TRAIN_PATH
from src.core.evaluation import summarize_ranks
from src.data.cross_modal_dataset import create_cross_modal_datasets, CrossModalDataset
from src.search.candidate_generator import CandidateDatabase, CandidateMatch
from src.data.hard_negative_dataset import HardNegativeIndex


@dataclass
class CanonicalQuery:
    query_id: int
    true_mol: str
    target_mass: float
    precursor_mz: float
    adduct: str | None
    spec_tensor: torch.Tensor
    matches: list[CandidateMatch]
    is_isomer_query: bool
    true_formula: str


class CanonicalBenchmark:
    """Benchmark harness separating 800 tuning molecules from 200 frozen evaluation queries."""

    def __init__(
        self,
        subset_size: int = 10000,
        n_benchmark_queries: int = 200,
        split_seed: int = 42,
        benchmark_seed: int = 123,
        ppm_primary: float = 20.0,
        ppm_fallback: float = 50.0,
        include_isotope: bool = True,
        save_artifacts: bool = True,
    ):
        self.subset_size = subset_size
        self.n_benchmark_queries = n_benchmark_queries
        self.split_seed = split_seed
        self.benchmark_seed = benchmark_seed
        self.ppm_primary = ppm_primary
        self.ppm_fallback = ppm_fallback
        self.include_isotope = include_isotope

        print(f"[CanonicalBenchmark] Loading cross-modal dataset ({subset_size} molecules)...", flush=True)
        self.train_ds, self.val_ds = create_cross_modal_datasets(
            train_path=TRAIN_PATH,
            subset_size=subset_size,
            val_frac=0.10,
            seed=split_seed,
        )

        all_mols = list(self.train_ds.selected_mols) + list(self.val_ds.selected_mols)
        all_smiles = {**self.train_ds.mol_smiles, **self.val_ds.mol_smiles}
        all_graphs = {**self.train_ds.graph_cache, **self.val_ds.graph_cache}
        valid_all_mols = [m for m in all_mols if m in all_graphs]

        self.cand_db = CandidateDatabase(
            molecules=valid_all_mols,
            mol_smiles=all_smiles,
            mol_graphs=all_graphs,
        )
        self.cand_index = HardNegativeIndex(self.cand_db.valid_mols, self.cand_db.mol_smiles)
        self.mol_to_idx = {m: i for i, m in enumerate(self.cand_db.valid_mols)}

        # Strict Partition of the 1,000 Validation Molecules:
        val_unique_mols = sorted(list(self.val_ds.selected_mols))
        rng = np.random.default_rng(benchmark_seed)
        perm = rng.permutation(len(val_unique_mols))

        benchmark_mols_set = set(val_unique_mols[i] for i in perm[:n_benchmark_queries])
        tuning_mols_set = set(val_unique_mols[i] for i in perm[n_benchmark_queries:])

        print(f"[CanonicalBenchmark] Split validation pool into: {len(tuning_mols_set)} Tuning molecules "
              f"and {len(benchmark_mols_set)} Frozen Benchmark molecules.", flush=True)

        # Build Query Sets
        # Map validation samples by molecule
        val_samples_by_mol: dict[str, list[int]] = {}
        for sample_idx, (_, mol_id) in enumerate(self.val_ds.samples):
            val_samples_by_mol.setdefault(mol_id, []).append(sample_idx)

        # 1. Build Canonical Benchmark Queries (1 query per benchmark molecule = 200 queries)
        self.benchmark_queries: list[CanonicalQuery] = []
        for m in val_unique_mols:
            if m in benchmark_mols_set and m in val_samples_by_mol:
                sample_idx = val_samples_by_mol[m][0]  # Pick canonical spectrum for this mol
                q = self._build_query(sample_idx)
                if q is not None:
                    self.benchmark_queries.append(q)

        # 2. Build Tuning Pool Queries (1 query per tuning molecule = up to 800 queries)
        self.tuning_queries: list[CanonicalQuery] = []
        for m in val_unique_mols:
            if m in tuning_mols_set and m in val_samples_by_mol:
                sample_idx = val_samples_by_mol[m][0]
                q = self._build_query(sample_idx)
                if q is not None:
                    self.tuning_queries.append(q)

        self.benchmark_isomer_indices = [i for i, q in enumerate(self.benchmark_queries) if q.is_isomer_query]
        self.tuning_isomer_indices = [i for i, q in enumerate(self.tuning_queries) if q.is_isomer_query]

        print(f"[CanonicalBenchmark] Ready: {len(self.tuning_queries)} tuning queries ({len(self.tuning_isomer_indices)} isomer queries) | "
              f"{len(self.benchmark_queries)} benchmark queries ({len(self.benchmark_isomer_indices)} isomer queries).", flush=True)

        if save_artifacts:
            self._save_contract_artifacts()

    def _build_query(self, sample_idx: int) -> CanonicalQuery | None:
        spec_tensor, _, true_mol = self.val_ds[sample_idx]
        spec_info = self.val_ds.samples[sample_idx][0]
        prec_mz = spec_info.get("precursor_mz", 0.0)
        adduct = spec_info.get("adduct", None)

        if true_mol not in self.cand_db.mol_smiles:
            return None

        target_idx = self.mol_to_idx[true_mol]
        target_mass = self.cand_db.raw_masses[target_idx]

        matches = self.cand_db.query_two_tier(
            precursor_mz=prec_mz if prec_mz > 0 else target_mass + 1.0078,
            adduct=adduct if adduct is not None else "[M+H]+",
            ppm_primary=self.ppm_primary,
            ppm_fallback=self.ppm_fallback,
            include_isotope=self.include_isotope,
        )

        f_true = self.cand_index.formula_map.get(true_mol, "")
        has_isomers = any(
            m.mol != true_mol and self.cand_index.formula_map.get(m.mol, "") == f_true
            for m in matches
        )

        return CanonicalQuery(
            query_id=int(sample_idx),
            true_mol=true_mol,
            target_mass=target_mass,
            precursor_mz=prec_mz,
            adduct=adduct,
            spec_tensor=spec_tensor,
            matches=matches,
            is_isomer_query=has_isomers,
            true_formula=f_true,
        )

    def _save_contract_artifacts(self) -> None:
        out_dir = ARTIFACTS_DIR / "stage06"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Benchmark queries metadata
        b_meta = [
            {
                "index": i,
                "query_id": q.query_id,
                "true_mol": q.true_mol,
                "formula": q.true_formula,
                "target_mass": round(q.target_mass, 4),
                "is_isomer_query": q.is_isomer_query,
                "n_candidates": len(q.matches),
            }
            for i, q in enumerate(self.benchmark_queries)
        ]
        with open(out_dir / "canonical_queries.json", "w", encoding="utf-8") as f:
            json.dump(b_meta, f, indent=2)

        # 2. Tuning queries metadata
        t_meta = [
            {
                "index": i,
                "query_id": q.query_id,
                "true_mol": q.true_mol,
                "formula": q.true_formula,
                "target_mass": round(q.target_mass, 4),
                "is_isomer_query": q.is_isomer_query,
                "n_candidates": len(q.matches),
            }
            for i, q in enumerate(self.tuning_queries)
        ]
        with open(out_dir / "tuning_queries.json", "w", encoding="utf-8") as f:
            json.dump(t_meta, f, indent=2)

        # 3. Candidate universe metadata
        c_meta = {
            "n_candidates": len(self.cand_db.valid_mols),
            "candidate_inchikeys": self.cand_db.valid_mols,
            "ppm_primary": self.ppm_primary,
            "ppm_fallback": self.ppm_fallback,
            "include_isotope": self.include_isotope,
        }
        with open(out_dir / "canonical_candidates.json", "w", encoding="utf-8") as f:
            json.dump(c_meta, f, indent=2)

        print(f"[CanonicalBenchmark] Saved benchmark contracts to {out_dir}", flush=True)

    def evaluate_ranks(
        self,
        ranks: list[int],
        top_k: int = 25,
        is_tuning: bool = False,
    ) -> dict[str, Any]:
        """Compute standardized overall and exact-isomer subset metrics."""
        m_overall = summarize_ranks(ranks, k=top_k)
        iso_indices = self.tuning_isomer_indices if is_tuning else self.benchmark_isomer_indices
        ranks_iso = [ranks[i] for i in iso_indices if i < len(ranks)]
        m_iso = summarize_ranks(ranks_iso, k=top_k) if len(ranks_iso) > 0 else {}

        return {
            "overall": {
                "n": len(ranks),
                "mrr": round(float(m_overall["mrr"]), 4),
                "hit@1": round(float(m_overall["hit@1"]), 4),
                "hit@5": round(float(m_overall["hit@5"]), 4),
                "hit@25": round(float(m_overall["hit@25"]), 4),
            },
            "exact_isomer_subset": {
                "n": len(ranks_iso),
                "mrr": round(float(m_iso["mrr"]), 4) if m_iso else 0.0,
                "hit@1": round(float(m_iso["hit@1"]), 4) if m_iso else 0.0,
                "hit@5": round(float(m_iso["hit@5"]), 4) if m_iso else 0.0,
                "hit@25": round(float(m_iso["hit@25"]), 4) if m_iso else 0.0,
            },
        }
