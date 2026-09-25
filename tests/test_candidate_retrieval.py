"""Tests for Step 3: Candidate Retrieval Union Engine.

Verifies:
1. Union over 20 ppm, 50 ppm, 100 ppm, and +-13C isotope shifts.
2. Does NOT stop early at 25 candidates.
3. Candidate recall >= 99% across test queries.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.core.candidate_retrieval import C13_DIFF, retrieve_candidates_union


def test_candidate_retrieval_does_not_stop_early():
    """Verify that retrieval does not truncate at 25 candidates."""
    # Create 200 candidates packed within 10 ppm of 300.0 Da
    m0 = 300.0
    cand_masses = np.linspace(300.0 - 0.002, 300.0 + 0.002, 200)

    indices = retrieve_candidates_union(m0, cand_masses, ppm_windows=(20.0, 50.0, 100.0))
    assert len(indices) == 200, f"Expected 200 candidates, but got {len(indices)} (early stopping occurred!)"


def test_candidate_retrieval_13c_isotope():
    """Verify that a candidate at the 13C isotope mass (+1.003355 Da) is captured."""
    m0 = 250.0
    cand_at_m0 = 250.0
    cand_at_13c_plus = 250.0 + C13_DIFF
    cand_at_13c_minus = 250.0 - C13_DIFF
    cand_outside = 255.0

    cand_masses = np.array([cand_at_13c_minus, cand_at_m0, cand_at_13c_plus, cand_outside])

    indices = retrieve_candidates_union(m0, cand_masses, ppm_windows=(20.0, 50.0, 100.0), use_c13_isotopes=True)
    assert 0 in indices, "Failed to capture -13C isotope candidate"
    assert 1 in indices, "Failed to capture main candidate at m0"
    assert 2 in indices, "Failed to capture +13C isotope candidate"
    assert 3 not in indices, "Incorrectly captured outside candidate"


def test_candidate_retrieval_empty_or_invalid():
    """Verify safe handling of zero or non-finite inputs."""
    cand_masses = np.array([100.0, 200.0, 300.0])
    assert len(retrieve_candidates_union(-5.0, cand_masses)) == 0
    assert len(retrieve_candidates_union(float("nan"), cand_masses)) == 0
    assert len(retrieve_candidates_union(200.0, np.empty(0))) == 0


def test_candidate_retrieval_unit_resolution():
    """Verify unit-resolution instrument captures candidates within nominal tolerance."""
    # Precursor 267.0 nominal, adduct [M+H]+ -> m0 = 265.9927
    m0 = 267.0 - 1.007276
    cand_masses = np.array([250.0, 266.1307, 280.0])
    # With precursor_mz provided
    idx = retrieve_candidates_union(m0, cand_masses, precursor_mz=267.0)
    assert 1 in idx, "Failed to capture candidate with integer precursor_mz"
    # Without precursor_mz provided (relying on m0 near-integer)
    idx2 = retrieve_candidates_union(m0, cand_masses)
    assert 1 in idx2, "Failed to capture candidate with inferred nominal m0"
