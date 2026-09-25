"""Spectral augmentations for contrastive learning.

Augmentations simulate real-world variability in MS/MS spectra:
- Peak dropout: random removal of fragment peaks
- Intensity jitter: Gaussian noise on intensities
- m/z shift: small random perturbation simulating instrument error
- Precursor masking: randomly zero out near-precursor peaks

All augmentations operate on (mz, intensity) numpy arrays and
return augmented copies (never modify in-place).
"""
from __future__ import annotations

import numpy as np


def peak_dropout(
    mz: np.ndarray,
    intensity: np.ndarray,
    drop_prob: float = 0.1,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly remove peaks with probability `drop_prob`.

    Always keeps at least 3 peaks. Prefers to drop low-intensity peaks
    by using intensity-weighted dropout probability.
    """
    if mz.size <= 3:
        return mz.copy(), intensity.copy()
    rng = rng or np.random.default_rng()
    # Weight dropout by inverse intensity: low peaks more likely to drop
    weights = 1.0 - 0.5 * intensity / (intensity.max() + 1e-8)
    drop_mask = rng.random(mz.size) < (drop_prob * weights)
    # Guarantee at least 3 peaks survive
    if drop_mask.sum() >= mz.size - 2:
        top3 = np.argsort(intensity)[-3:]
        drop_mask[top3] = False
    keep = ~drop_mask
    return mz[keep].copy(), intensity[keep].copy()


def intensity_jitter(
    mz: np.ndarray,
    intensity: np.ndarray,
    sigma: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Add Gaussian noise to intensities, then re-normalize to [0, 1]."""
    rng = rng or np.random.default_rng()
    noisy = intensity + rng.normal(0, sigma, size=intensity.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, None)
    peak_max = noisy.max()
    if peak_max > 0:
        noisy = noisy / peak_max
    return mz.copy(), noisy


def mz_shift(
    mz: np.ndarray,
    intensity: np.ndarray,
    ppm: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Random per-peak m/z perturbation simulating instrument mass error."""
    rng = rng or np.random.default_rng()
    # Each peak gets its own shift proportional to its m/z
    shifts = rng.normal(0, ppm / 1e6, size=mz.shape).astype(np.float32)
    shifted = mz * (1.0 + shifts)
    return shifted, intensity.copy()


def apply_augmentations(
    mz: np.ndarray,
    intensity: np.ndarray,
    drop_prob: float = 0.1,
    jitter_sigma: float = 0.05,
    shift_ppm: float = 10.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the full augmentation pipeline to one spectrum."""
    rng = rng or np.random.default_rng()
    mz, intensity = peak_dropout(mz, intensity, drop_prob=drop_prob, rng=rng)
    mz, intensity = intensity_jitter(mz, intensity, sigma=jitter_sigma, rng=rng)
    mz, intensity = mz_shift(mz, intensity, ppm=shift_ppm, rng=rng)
    return mz, intensity
