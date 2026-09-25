"""Mass and formula-based candidate filtering.

Merged from src/candidate_generation.py into src/search/candidate_filter.py.
Provides the chemistry-based filtering layer for Stage 0/1 and Stage 4+.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from src.core.adducts import neutral_mass

ISOTOPE_DELTA = 1.00335  # Carbon-13 / neutron mass shift in Da
PPM_FALLBACK = 50.0


def mass_pair_ok(
    nm_q: np.ndarray,
    nm_lib: np.ndarray,
    ppm: float,
    ppm_fallback: float = PPM_FALLBACK,
    isotope: bool = True,
) -> np.ndarray:
    """Vectorized primary-or-fallback mass gate for (query, library) pairs.

    Primary: |nm_q - nm_lib| <= nm_q * ppm / 1e6.
    Fallback (always evaluated so chunked search never strands a query with
    zero primary pairs):
      - expanded window at max(ppm, ppm_fallback)
      - M±1 isotope centers at the primary ppm.
    Non-finite query masses keep the historical behavior (do not gate).
    """
    nm_q = np.asarray(nm_q, dtype=np.float64)
    nm_lib = np.asarray(nm_lib, dtype=np.float64)
    out = np.zeros(nm_q.shape, dtype=bool)
    nonfinite_q = ~np.isfinite(nm_q)
    finite = ~nonfinite_q & np.isfinite(nm_lib)
    if np.any(finite):
        q = nm_q[finite]
        lib = nm_lib[finite]
        ppm_f = float(ppm)
        ppm_w = max(ppm_f, float(ppm_fallback))
        with np.errstate(invalid="ignore", divide="ignore"):
            match = np.abs(q - lib) <= (q * ppm_f / 1e6)
            match |= np.abs(q - lib) <= (q * ppm_w / 1e6)
            if isotope:
                # Isotope shift ~1 Da is far beyond ppm windows around nm_q;
                # re-center and allow both primary and expanded tolerances.
                for shift in (-ISOTOPE_DELTA, ISOTOPE_DELTA):
                    center = q + shift
                    match |= np.abs(center - lib) <= (q * ppm_f / 1e6)
                    match |= np.abs(center - lib) <= (q * ppm_w / 1e6)
            match &= np.isfinite(match)
        out[finite] = match
    return out | nonfinite_q


def fallback_mass_windows(
    precursor_mz: float,
    adduct: str | None = None,
    ppm_primary: float = 20.0,
    ppm_fallback: float = 50.0,
    include_isotope: bool = True,
) -> dict[str, list[tuple[float, float]]]:
    """Return primary, isotope, and expanded tolerance mass search windows."""
    target = neutral_mass_from_precursor(precursor_mz, adduct=adduct)
    tol_p = target * ppm_primary / 1e6
    tol_f = target * ppm_fallback / 1e6

    windows = {
        "primary": [(target - tol_p, target + tol_p)],
        "fallback": [(target - tol_f, target + tol_f)],
    }
    if include_isotope:
        windows["isotope"] = [
            (target - ISOTOPE_DELTA - tol_p, target - ISOTOPE_DELTA + tol_p),
            (target + ISOTOPE_DELTA - tol_p, target + ISOTOPE_DELTA + tol_p),
        ]
    return windows


def neutral_mass_from_precursor(precursor_mz: float, adduct: str | None = None, charge: int = 1) -> float:
    """Convert precursor m/z to neutral mass using the shared adduct parser."""
    if adduct is None:
        return float(precursor_mz)
    result = neutral_mass(precursor_mz, adduct)
    if result is None:
        return float(precursor_mz)
    return float(result)


def ppm_error(observed: float, expected: float) -> float:
    if expected == 0:
        return float("inf")
    return abs(observed - expected) / expected * 1e6


def mass_filter_candidates(precursor_mz: float, adduct: str | None, candidates: Iterable[float], ppm_tolerance: float = 20.0) -> list[float]:
    """Return candidates whose mass lies close to the precursor-derived neutral mass."""
    target = neutral_mass_from_precursor(precursor_mz, adduct=adduct)
    filtered = []
    for candidate in candidates:
        if ppm_error(candidate, target) <= ppm_tolerance:
            filtered.append(candidate)
    return filtered


def formula_mass_range(precursor_mz: float, adduct: str | None, ppm_tolerance: float = 20.0) -> tuple[float, float]:
    target = neutral_mass_from_precursor(precursor_mz, adduct=adduct)
    ppm = ppm_tolerance / 1e6
    delta = target * ppm
    return target - delta, target + delta
