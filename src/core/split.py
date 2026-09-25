"""Molecule-level train/val splits.

Moved from src/split.py to src/core/split.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def molecule_spectrum_counts(meta: pd.DataFrame, molecule_col: str = "inchikey") -> pd.Series:
    return meta.groupby(molecule_col).size()


def select_val_molecules(
    meta: pd.DataFrame,
    n_molecules: int,
    seed: int = 42,
    molecule_col: str = "inchikey",
    min_spectra: int = 2,
) -> list[str]:
    """Sample validation molecules that have enough spectra for a held-out query."""
    counts = molecule_spectrum_counts(meta, molecule_col)
    eligible = counts[counts >= min_spectra].index.to_numpy(dtype=object)
    rng = np.random.default_rng(seed)
    n = min(n_molecules, eligible.size)
    chosen = rng.choice(eligible, size=n, replace=False)
    return sorted(str(m) for m in chosen)


def select_queries(
    meta: pd.DataFrame,
    val_molecules: list[str],
    seed: int = 42,
    molecule_col: str = "inchikey",
) -> pd.DataFrame:
    """Pick exactly one held-out query spectrum per validation molecule.

    All other spectra of those molecules remain in the library (Class-1-like
    protocol). Returns a frame with library/global row ids and query metadata.
    """
    rng = np.random.default_rng(seed + 1)
    val_set = set(val_molecules)
    subset = meta[meta[molecule_col].isin(val_set)]
    rows: list[pd.Series] = []
    for molecule, group in subset.groupby(molecule_col, sort=True):
        idx = group.index.to_numpy()
        choice = idx[int(rng.integers(0, len(idx)))]
        rows.append(meta.loc[choice])
    queries = pd.DataFrame(rows).reset_index().rename(columns={"index": "row_id"})
    return queries


def split_protocol_b_library_mask(
    meta: pd.DataFrame,
    val_molecules: list[str],
    molecule_col: str = "inchikey",
) -> np.ndarray:
    """Boolean mask over meta rows: False for every spectrum of val molecules."""
    return ~meta[molecule_col].isin(set(val_molecules)).to_numpy()


def molecule_disjoint_exclude_rows(
    meta: pd.DataFrame,
    val_molecules: list[str],
    molecule_col: str = "inchikey",
) -> np.ndarray:
    """Row ids to drop for Protocol C: every spectrum of every val molecule.

    Protocol A only drops the held-out query spectrum (siblings stay in the
    library). Protocol C is molecule-disjoint — 0 library spectra of M when
    M is in val — so classical spectral recall of M itself collapses and the
    honest baseline is mass+structure scoring (see protocol_c_ranks).
    """
    mask = meta[molecule_col].isin(set(val_molecules)).to_numpy()
    return np.flatnonzero(mask).astype(np.int64)
