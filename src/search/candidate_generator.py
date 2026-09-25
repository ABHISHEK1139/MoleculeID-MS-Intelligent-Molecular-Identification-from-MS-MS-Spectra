"""Physics-informed candidate generation and fast mass-window indexing.

Provides O(log N) sorted binary search over candidate molecules by exact monoisotopic
neutral mass, with two-tier confidence gating (20 ppm primary, 50 ppm fallback,
and isotope misselection shift).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from src.core.adducts import neutral_mass
from src.search.candidate_filter import neutral_mass_from_precursor, ISOTOPE_DELTA


@dataclass
class CandidateMatch:
    mol: str
    smiles: str
    formula: str
    exact_mass: float
    ppm_error: float
    tier: int  # 1 = primary (<= 20 ppm), 2 = fallback (<= 50 ppm), 3 = isotope shift
    weight: float  # Multiplier penalty (1.0 for tier 1, 0.9 for tier 2, 0.85 for tier 3)


class CandidateDatabase:
    """Pre-indexed candidate molecule database supporting fast mass and formula querying."""

    def __init__(
        self,
        molecules: list[str],
        mol_smiles: dict[str, str],
        mol_graphs: dict[str, Any] | None = None,
    ):
        self.molecules = molecules
        self.mol_smiles = mol_smiles
        self.mol_graphs = mol_graphs or {}

        # Compute exact masses and formulas using RDKit
        valid_mols: list[str] = []
        mass_list: list[float] = []
        formula_list: list[str] = []
        self.formula_to_mols: dict[str, list[str]] = {}

        for m in molecules:
            smi = mol_smiles.get(m, "")
            if not smi:
                continue
            mol_obj = Chem.MolFromSmiles(smi)
            if mol_obj is None:
                continue
            em = float(ExactMolWt(mol_obj))
            f_str = str(CalcMolFormula(mol_obj))

            valid_mols.append(m)
            mass_list.append(em)
            formula_list.append(f_str)

            if f_str not in self.formula_to_mols:
                self.formula_to_mols[f_str] = []
            self.formula_to_mols[f_str].append(m)

        self.valid_mols = valid_mols
        self.raw_masses = np.asarray(mass_list, dtype=np.float64)
        self.raw_formulas = formula_list

        # Build sorted mass array for O(log N) binary search
        sort_idx = np.argsort(self.raw_masses)
        self.sorted_masses = self.raw_masses[sort_idx]
        self.sorted_mols = [self.valid_mols[i] for i in sort_idx]
        self.sorted_formulas = [self.raw_formulas[i] for i in sort_idx]
        self.sorted_smiles = [self.mol_smiles[self.sorted_mols[i]] for i in range(len(sort_idx))]

    def __len__(self) -> int:
        return len(self.sorted_mols)

    def search_range(self, min_mass: float, max_mass: float) -> tuple[int, int]:
        """Binary search indices [left, right) for exact masses in [min_mass, max_mass]."""
        left = int(np.searchsorted(self.sorted_masses, min_mass, side="left"))
        right = int(np.searchsorted(self.sorted_masses, max_mass, side="right"))
        return left, right

    def query_two_tier(
        self,
        precursor_mz: float,
        adduct: str | None = None,
        ppm_primary: float = 20.0,
        ppm_fallback: float = 50.0,
        include_isotope: bool = True,
        min_candidates: int = 1,
    ) -> list[CandidateMatch]:
        """Query candidates using two-tier penalized mass gating.

        Tier 1 (High Trust, weight=1.0): |Delta M| <= ppm_primary (default 20 ppm)
        Tier 2 (Fallback, weight=0.9): ppm_primary < |Delta M| <= ppm_fallback (default 50 ppm)
        Tier 3 (Isotope Shift, weight=0.85): M +- 1.00335 Da within ppm_primary

        Returns:
            List of CandidateMatch objects sorted by (tier ASC, ppm_error ASC)
        """
        target_mass = neutral_mass_from_precursor(precursor_mz, adduct=adduct)

        seen_mols: set[str] = set()
        matches: list[CandidateMatch] = []

        # ── Tier 1: Primary mass window (<= 20 ppm) ──────────────────────────
        tol_primary = target_mass * ppm_primary / 1e6
        left_1, right_1 = self.search_range(target_mass - tol_primary, target_mass + tol_primary)

        for idx in range(left_1, right_1):
            mol = self.sorted_mols[idx]
            em = self.sorted_masses[idx]
            ppm_err = abs(em - target_mass) / target_mass * 1e6
            seen_mols.add(mol)
            matches.append(CandidateMatch(
                mol=mol,
                smiles=self.sorted_smiles[idx],
                formula=self.sorted_formulas[idx],
                exact_mass=em,
                ppm_error=float(ppm_err),
                tier=1,
                weight=1.0,
            ))

        # ── Tier 2: Expanded fallback window (20 to 50 ppm) ───────────────────
        # Included if Tier 1 has fewer candidates than min_candidates or for soft ranking
        tol_fallback = target_mass * ppm_fallback / 1e6
        left_2, right_2 = self.search_range(target_mass - tol_fallback, target_mass + tol_fallback)

        for idx in range(left_2, right_2):
            mol = self.sorted_mols[idx]
            if mol in seen_mols:
                continue
            em = self.sorted_masses[idx]
            ppm_err = abs(em - target_mass) / target_mass * 1e6
            seen_mols.add(mol)
            matches.append(CandidateMatch(
                mol=mol,
                smiles=self.sorted_smiles[idx],
                formula=self.sorted_formulas[idx],
                exact_mass=em,
                ppm_error=float(ppm_err),
                tier=2,
                weight=0.90,
            ))

        # ── Tier 3: Isotope shifts (M +- 1.00335 Da) ──────────────────────────
        if include_isotope:
            for shift in (-ISOTOPE_DELTA, ISOTOPE_DELTA):
                iso_target = target_mass + shift
                iso_tol = iso_target * ppm_primary / 1e6
                l_iso, r_iso = self.search_range(iso_target - iso_tol, iso_target + iso_tol)

                for idx in range(l_iso, r_iso):
                    mol = self.sorted_mols[idx]
                    if mol in seen_mols:
                        continue
                    em = self.sorted_masses[idx]
                    ppm_err = abs(em - iso_target) / iso_target * 1e6
                    seen_mols.add(mol)
                    matches.append(CandidateMatch(
                        mol=mol,
                        smiles=self.sorted_smiles[idx],
                        formula=self.sorted_formulas[idx],
                        exact_mass=em,
                        ppm_error=float(ppm_err),
                        tier=3,
                        weight=0.85,
                    ))

        matches.sort(key=lambda m: (m.tier, m.ppm_error))
        return matches

    def query_by_formulas(self, candidate_formulas: set[str] | list[str]) -> list[str]:
        """Return all candidate molecule IDs that match any of the given formulas."""
        matched: set[str] = set()
        for f in candidate_formulas:
            if f in self.formula_to_mols:
                matched.update(self.formula_to_mols[f])
        return sorted(list(matched))
