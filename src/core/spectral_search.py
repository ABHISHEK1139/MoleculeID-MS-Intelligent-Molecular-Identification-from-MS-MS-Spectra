"""Core spectral search and library retrieval functions.

Provides fast mutual 1-to-1 peak-matching modified cosine similarity
with symmetric precursor mass offset and compact library indexing.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np


def fast_mutual_cosine(
    q_mzs: np.ndarray,
    q_ints: np.ndarray,
    r_mzs: np.ndarray,
    r_ints: np.ndarray,
    delta: float = 0.0,
    tol: float = 0.015,
) -> tuple[float, int]:
    """Fast mutual 1-to-1 peak matching cosine with symmetric precursor mass offset.

    Returns:
        (best_cosine, matched_peak_count)
    """
    if q_mzs.size == 0 or r_mzs.size == 0:
        return 0.0, 0

    q_norm = np.linalg.norm(q_ints)
    r_norm = np.linalg.norm(r_ints)
    if q_norm <= 0 or r_norm <= 0:
        return 0.0, 0

    qi = (q_ints / q_norm).astype(np.float32)
    li = (r_ints / r_norm).astype(np.float32)

    best_dot = 0.0
    best_cnt = 0
    # Symmetric shifts to correctly handle bidirectional mass-shifted fragmentation
    shifts = [0.0] if abs(delta) < 1e-4 else [0.0, delta, -delta]

    for s in shifts:
        qs = q_mzs + s
        order = np.argsort(qs)
        q_sorted = qs[order]
        qi_sorted = qi[order]

        lo = np.searchsorted(q_sorted, r_mzs - tol, side="left")
        hi = np.searchsorted(q_sorted, r_mzs + tol, side="right")

        dot = 0.0
        cnt = 0
        matched_q: set[int] = set()
        for j in range(r_mzs.size):
            a, b = int(lo[j]), int(hi[j])
            if b > a:
                if b - a == 1:
                    m_idx = a
                else:
                    diffs = np.abs(q_sorted[a:b] - r_mzs[j])
                    m_idx = a + int(np.argmin(diffs))
                if m_idx not in matched_q:
                    matched_q.add(m_idx)
                    dot += float(li[j] * qi_sorted[m_idx])
                    cnt += 1

        if dot > best_dot:
            best_dot = dot
            best_cnt = cnt

    return min(1.0, float(best_dot)), best_cnt


class CompactSpectralLibrary:
    """Pre-sorted flat array representation of spectral libraries for fast mass-window lookups."""

    def __init__(
        self,
        smiles: np.ndarray,
        neutral_masses: np.ndarray,
        precursor_mzs: np.ndarray,
        collision_energies: np.ndarray,
        mzs_list: list[np.ndarray],
        intens_list: list[np.ndarray],
        n_supporting: np.ndarray | None = None,
        source_counts: np.ndarray | None = None,
    ):
        order = np.argsort(neutral_masses)
        self.smiles = smiles[order]
        self.neutral_masses = neutral_masses[order]
        self.precursor_mzs = precursor_mzs[order]
        self.collision_energies = collision_energies[order]
        self.mzs_list = [mzs_list[i] for i in order]
        self.intens_list = [intens_list[i] for i in order]
        self.n_supporting = n_supporting[order] if n_supporting is not None else np.ones(len(order), dtype=np.int16)
        self.source_counts = source_counts[order] if source_counts is not None else np.ones(len(order), dtype=np.int8)

    def query_window(self, mass: float, ppm: float = 20.0) -> tuple[int, int]:
        tol = mass * (ppm / 1e6)
        l_idx = int(np.searchsorted(self.neutral_masses, mass - tol, side="left"))
        r_idx = int(np.searchsorted(self.neutral_masses, mass + tol, side="right"))
        return l_idx, r_idx
