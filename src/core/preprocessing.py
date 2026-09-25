"""Spectral preprocessing filters and variant configuration.

Moved from src/preprocessing.py to src/core/preprocessing.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

C13_DELTA = 1.00335


def normalize_intensities(intensities: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(intensities), dtype=np.float32)
    if values.size == 0:
        return values
    max_val = np.max(values)
    if max_val <= 0:
        return values
    return values / max_val


def filter_relative_intensity(mz: Iterable[float], intensities: Iterable[float], min_rel: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    mz_arr = np.asarray(list(mz), dtype=np.float32)
    intensity_arr = np.asarray(list(intensities), dtype=np.float32)
    if mz_arr.size == 0 or intensity_arr.size == 0:
        return mz_arr, intensity_arr
    norm = normalize_intensities(intensity_arr)
    mask = norm >= min_rel
    return mz_arr[mask], norm[mask]


def keep_top_peaks(mz: Iterable[float], intensities: Iterable[float], max_peaks: int = 100) -> tuple[np.ndarray, np.ndarray]:
    mz_arr = np.asarray(list(mz), dtype=np.float32)
    intensity_arr = np.asarray(list(intensities), dtype=np.float32)
    if mz_arr.size == 0 or max_peaks is None or max_peaks <= 0:
        return mz_arr, intensity_arr
    if mz_arr.size <= max_peaks:
        return mz_arr, intensity_arr
    order = np.argsort(intensity_arr)[::-1][:max_peaks]
    return mz_arr[order], intensity_arr[order]


def remove_precursor_peaks(mz: Iterable[float], intensities: Iterable[float], precursor_mz: float, tolerance: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    mz_arr = np.asarray(list(mz), dtype=np.float32)
    intensity_arr = np.asarray(list(intensities), dtype=np.float32)
    if mz_arr.size == 0:
        return mz_arr, intensity_arr
    mask = np.abs(mz_arr - precursor_mz) > tolerance
    return mz_arr[mask], intensity_arr[mask]


def deisotope_peaks(
    mz: np.ndarray,
    intensities: np.ndarray,
    delta: float = C13_DELTA,
    tolerance: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop peaks that look like 13C/15N isotopes of a lower-mass peak.

    Rule (per the research report): if a peak at m2 has a companion near
    m2 - 1.00335 Da with >= intensity, the higher-mass peak is an isotope.
    """
    if mz.size <= 1:
        return mz, intensities
    order = np.argsort(mz)
    mz_sorted = mz[order]
    int_sorted = intensities[order]
    keep = np.ones(mz_sorted.size, dtype=bool)
    for i in range(mz_sorted.size - 1, -1, -1):
        if not keep[i]:
            continue
        target = mz_sorted[i] - delta
        j = np.searchsorted(mz_sorted, target - tolerance)
        found = False
        while j < mz_sorted.size and mz_sorted[j] <= target + tolerance:
            if keep[j] and int_sorted[i] <= int_sorted[j]:
                keep[i] = False
                found = True
                break
            j += 1
        if found:
            continue
    kept_sorted_idx = order[keep]
    return mz[kept_sorted_idx], intensities[kept_sorted_idx]


@dataclass(frozen=True)
class VariantConfig:
    """One preprocessing ablation setting."""

    key: str
    name: str
    min_rel: float
    remove_precursor: bool
    deisotope: bool
    max_peaks: int
    mass_filter: bool
    modified_cosine: bool


def default_variants() -> dict[str, VariantConfig]:
    """Stage 1 ablations 1A-1F."""
    return {
        "1A": VariantConfig("1A", "raw_normalized", 0.0, False, False, 100, False, False),
        "1B": VariantConfig("1B", "noise_filter", 0.01, False, False, 100, False, False),
        "1C": VariantConfig("1C", "noise_plus_precursor", 0.01, True, False, 100, False, False),
        "1D": VariantConfig("1D", "noise_precursor_deisotope", 0.01, True, True, 100, False, False),
        "1E": VariantConfig("1E", "deiso_mass_filter", 0.01, True, True, 100, True, False),
        "1F": VariantConfig("1F", "deiso_mass_filter_modified", 0.01, True, True, 100, True, True),
    }


def preprocess_variant(
    mz: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float | None,
    cfg: VariantConfig,
    precursor_tol: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one variant's filters to a single spectrum."""
    values = np.asarray(intensities, dtype=np.float32)
    mz_arr = np.asarray(mz, dtype=np.float32)
    if values.size == 0:
        return mz_arr, values

    peak_max = float(values.max())
    if peak_max > 0:
        values = values / peak_max

    if cfg.min_rel > 0:
        mask = values >= cfg.min_rel
        mz_arr = mz_arr[mask]
        values = values[mask]

    if cfg.remove_precursor and precursor_mz is not None and mz_arr.size:
        mask = np.abs(mz_arr - precursor_mz) > precursor_tol
        mz_arr = mz_arr[mask]
        values = values[mask]

    if cfg.deisotope and mz_arr.size:
        mz_arr, values = deisotope_peaks(mz_arr, values)

    if cfg.max_peaks and mz_arr.size > cfg.max_peaks:
        order = np.argsort(values)[::-1][: cfg.max_peaks]
        mz_arr = mz_arr[order]
        values = values[order]

    return mz_arr, values


def preprocess_spectrum(
    mz: Iterable[float],
    intensities: Iterable[float],
    precursor_mz: float | None = None,
    min_rel: float = 0.01,
    max_peaks: int = 100,
    remove_precursor: bool = True,
    deisotope: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimal preprocessing pipeline for a single spectrum."""
    mz_arr = np.asarray(list(mz), dtype=np.float32)
    intensity_arr = np.asarray(list(intensities), dtype=np.float32)

    intensity_arr = normalize_intensities(intensity_arr)
    mz_arr, intensity_arr = filter_relative_intensity(mz_arr, intensity_arr, min_rel=min_rel)

    if remove_precursor and precursor_mz is not None:
        mz_arr, intensity_arr = remove_precursor_peaks(mz_arr, intensity_arr, precursor_mz, tolerance=0.5)

    if deisotope and mz_arr.size:
        mz_arr, intensity_arr = deisotope_peaks(mz_arr, intensity_arr)

    mz_arr, intensity_arr = keep_top_peaks(mz_arr, intensity_arr, max_peaks=max_peaks)

    return mz_arr, intensity_arr
