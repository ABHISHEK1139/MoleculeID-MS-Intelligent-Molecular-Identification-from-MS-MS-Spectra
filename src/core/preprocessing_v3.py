"""Unified Preprocessing & Inference Specification (v3_clean).

Single authoritative source of truth for:
1. Exact spectral preprocessing (deisotoping, sqrt intensity scaling, L2 normalization, top-60 peaks)
2. Exact binning grid ([20.0, 1500.0) @ 1.0 Da = 1480 bins)
3. Precursor metadata encoding with standardized missing CE default (30.0 eV)
4. Exact monoisotopic adduct parsing and neutral mass computation
5. Bidirectional symmetric modified cosine matching
6. Progressive candidate retrieval with nearest-neighbor fallback
7. Strictly unprivileged, leak-free candidate evidence features
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any
import numpy as np

# ── 1. Production Spectral Grid Constants ─────────────────────────────────────
MZ_BIN_MIN: float = 20.0
MZ_BIN_MAX: float = 1500.0
COARSE_BIN_WIDTH: float = 1.0
COARSE_N_BINS: int = int(round((MZ_BIN_MAX - MZ_BIN_MIN) / COARSE_BIN_WIDTH))  # 1480
DEFAULT_MISSING_CE: float = 30.0
MAX_PEAKS: int = 60
DEISOTOPE_TOL_DA: float = 0.015


# ── 2. Exact Monoisotopic Physics & Adduct Engine ─────────────────────────────
ELECTRON_MASS: float = 0.00054858

ATOMIC_MASSES: dict[str, float] = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.9897692820,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Br": 78.9183376,
    "I": 126.904473,
}

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula: str) -> float | None:
    """Monoisotopic neutral mass of an elemental formula."""
    if not formula:
        return 0.0
    mass = 0.0
    pos = 0
    for match in _FORMULA_TOKEN_RE.finditer(formula):
        if match.start() != pos:
            return None
        elem = match.group(1)
        cnt = int(match.group(2)) if match.group(2) else 1
        if elem not in ATOMIC_MASSES:
            return None
        mass += ATOMIC_MASSES[elem] * cnt
        pos = match.end()
    return mass if pos == len(formula) else None


def _group_mass(formula: str) -> float | None:
    direct = formula_mass(formula)
    if direct is not None:
        return direct
    match = re.fullmatch(r"(\d+)([A-Z][a-z]?(?:\d*[A-Z][a-z]?)*)", formula)
    if match:
        inner = formula_mass(match.group(2))
        return inner * int(match.group(1)) if inner is not None else None
    return None


@dataclass(frozen=True)
class AdductSpec:
    n_m: int
    adds_mass: float
    losses_mass: float
    charge: int
    positive: bool


_ADDUCT_CACHE: dict[str, AdductSpec | None] = {}


def parse_adduct(adduct: str | None) -> AdductSpec | None:
    if adduct is None:
        return None
    adduct_str = str(adduct).strip()
    if adduct_str in _ADDUCT_CACHE:
        return _ADDUCT_CACHE[adduct_str]

    match = re.fullmatch(r"\[([^\]]+)\](\d*)([+-])", adduct_str)
    if not match:
        _ADDUCT_CACHE[adduct_str] = None
        return None

    core, charge_str, sign = match.group(1), match.group(2), match.group(3)
    charge = int(charge_str) if charge_str else 1
    if charge == 0:
        _ADDUCT_CACHE[adduct_str] = None
        return None
    positive = sign == "+"

    core_match = re.fullmatch(r"(\d*)M(.*)", core)
    if core_match is None:
        _ADDUCT_CACHE[adduct_str] = None
        return None
    n_m = int(core_match.group(1)) if core_match.group(1) else 1
    rest = core_match.group(2)
    if n_m == 0:
        _ADDUCT_CACHE[adduct_str] = None
        return None

    parts = re.findall(r"([+-])([A-Za-z0-9]+)", rest)
    adds_mass = 0.0
    losses_mass = 0.0
    if parts:
        rebuilt = "".join(sign_i + form for sign_i, form in parts)
        if rebuilt != rest:
            _ADDUCT_CACHE[adduct_str] = None
            return None
        for sign_i, form in parts:
            m = _group_mass(form)
            if m is None:
                _ADDUCT_CACHE[adduct_str] = None
                return None
            if sign_i == "+":
                adds_mass += m
            else:
                losses_mass += m

    spec = AdductSpec(n_m=n_m, adds_mass=adds_mass, losses_mass=losses_mass, charge=charge, positive=positive)
    _ADDUCT_CACHE[adduct_str] = spec
    return spec


def neutral_mass(precursor_mz: float, adduct: str | None) -> float | None:
    """Exact monoisotopic neutral mass derivation."""
    spec = parse_adduct(adduct)
    if spec is None:
        return None
    signed = spec.charge * float(precursor_mz)
    electron = spec.charge * ELECTRON_MASS
    ion = (signed + electron) if spec.positive else (signed - electron)
    return (ion - spec.adds_mass + spec.losses_mass) / spec.n_m


# ── 3. Exact Spectral Cleaning & Binning ──────────────────────────────────────
def deisotope_peaks(mzs: np.ndarray, ints: np.ndarray, tol_da: float = DEISOTOPE_TOL_DA) -> tuple[np.ndarray, np.ndarray]:
    """Remove 13C companion peaks within tol_da."""
    if mzs.size <= 1:
        return mzs, ints
    order = np.argsort(mzs)
    mzs_s = mzs[order]
    ints_s = ints[order]
    keep = np.ones(mzs_s.size, dtype=bool)

    for i in range(mzs_s.size):
        if not keep[i]:
            continue
        target_iso = mzs_s[i] + 1.003355
        lo = np.searchsorted(mzs_s, target_iso - tol_da)
        hi = np.searchsorted(mzs_s, target_iso + tol_da)
        for j in range(lo, hi):
            if ints_s[j] < ints_s[i] * 0.90:
                keep[j] = False

    return mzs_s[keep], ints_s[keep]


def preprocess_spectrum(
    raw_mzs: np.ndarray,
    raw_intens: np.ndarray,
    max_peaks: int = MAX_PEAKS,
    deisotope: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Single authoritative spectral cleaning: deisotoping, sqrt scaling, L2 norm, top-K selection."""
    mzs = np.asarray(raw_mzs, dtype=np.float32)
    intens = np.asarray(raw_intens, dtype=np.float32)

    if mzs.size > 1 and deisotope:
        mzs, intens = deisotope_peaks(mzs, intens)

    intens = np.sqrt(np.maximum(intens, 0.0))
    norm = np.linalg.norm(intens)
    if norm > 0:
        intens = intens / norm

    if mzs.size > max_peaks:
        top_idx = np.argsort(-intens)[:max_peaks]
        top_idx = top_idx[np.argsort(mzs[top_idx])]
        mzs = mzs[top_idx]
        intens = intens[top_idx]

    return mzs, intens


def spectrum_to_coarse_bins(mzs: np.ndarray, intens: np.ndarray) -> np.ndarray:
    """Bin peak list into 1480-D vector on [20.0, 1500.0) @ 1.0 Da."""
    bins = np.zeros(COARSE_N_BINS, dtype=np.float32)
    if mzs.size == 0:
        return bins
    mask = (mzs >= MZ_BIN_MIN) & (mzs < MZ_BIN_MAX)
    if not np.any(mask):
        return bins
    mzs_m = mzs[mask]
    ints_m = intens[mask]
    indices = ((mzs_m - MZ_BIN_MIN) / COARSE_BIN_WIDTH).astype(np.int64)
    np.add.at(bins, indices, ints_m)
    norm = np.linalg.norm(bins)
    if norm > 0:
        bins /= norm
    return bins


def encode_spectrum_feature(
    mzs: np.ndarray,
    intens: np.ndarray,
    precursor_mz: float,
    adduct: str | None,
    collision_energy_ev: Any,
) -> np.ndarray:
    """Combines 1480-D binned spectrum + 3-D metadata [prec/1000, ion_mode, ce/100]."""
    binned = spectrum_to_coarse_bins(mzs, intens)
    ce_val = DEFAULT_MISSING_CE
    if collision_energy_ev is not None:
        try:
            if isinstance(collision_energy_ev, (list, tuple, np.ndarray)) and len(collision_energy_ev) > 0:
                ce_val = float(collision_energy_ev[0])
            else:
                ce_val = float(collision_energy_ev)
            if not np.isfinite(ce_val):
                ce_val = DEFAULT_MISSING_CE
        except (ValueError, TypeError):
            ce_val = DEFAULT_MISSING_CE

    spec = parse_adduct(adduct)
    ion_mode = 1.0 if (spec is not None and spec.positive) else (0.0 if (spec is not None and not spec.positive) else 1.0)
    meta_vec = np.array([precursor_mz / 1000.0, ion_mode, ce_val / 100.0], dtype=np.float32)
    return np.concatenate([binned, meta_vec])


# ── 4. Bidirectional Modified Cosine ──────────────────────────────────────────
def fast_mutual_cosine(
    q_mzs: np.ndarray,
    q_ints: np.ndarray,
    r_mzs: np.ndarray,
    r_ints: np.ndarray,
    delta: float = 0.0,
    tol_da: float = 0.015,
) -> tuple[float, int]:
    """Fast mutual 1-to-1 peak-matching modified cosine with symmetric bidirectional shifts."""
    if q_mzs.size == 0 or r_mzs.size == 0:
        return 0.0, 0

    q_norm = np.linalg.norm(q_ints)
    r_norm = np.linalg.norm(r_ints)
    if q_norm <= 0 or r_norm <= 0:
        return 0.0, 0

    qi = (q_ints / q_norm).astype(np.float32)
    ri = (r_ints / r_norm).astype(np.float32)

    best_dot = 0.0
    best_cnt = 0
    shifts = [0.0] if abs(delta) < 1e-4 else [0.0, delta, -delta]

    for s in shifts:
        qs = q_mzs + s
        order = np.argsort(qs)
        q_sorted = qs[order]
        qi_sorted = qi[order]

        lo = np.searchsorted(q_sorted, r_mzs - tol_da, side="left")
        hi = np.searchsorted(q_sorted, r_mzs + tol_da, side="right")

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
                    dot += float(ri[j] * qi_sorted[m_idx])
                    cnt += 1

        if dot > best_dot:
            best_dot = dot
            best_cnt = cnt

    return min(1.0, float(best_dot)), best_cnt


# ── 5. Progressive Candidate Retrieval ─────────────────────────────────────────
def retrieve_candidates_progressive(
    m_neutral: float,
    cand_masses: np.ndarray,
    min_cands: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns candidate indices and their retrieval tier weights (1.0, 0.70, 0.40, 0.20)."""
    cands_idx = np.array([], dtype=np.int64)
    tier_weights = np.array([], dtype=np.float32)

    # Primary (20 ppm) -> Fallback (50 ppm) -> Wide (100 ppm)
    for tol_ppm, weight in [(20e-6, 1.0), (50e-6, 0.70), (100e-6, 0.40)]:
        l_idx = np.searchsorted(cand_masses, m_neutral * (1.0 - tol_ppm))
        r_idx = np.searchsorted(cand_masses, m_neutral * (1.0 + tol_ppm))
        if r_idx > l_idx:
            cands_idx = np.arange(l_idx, r_idx)
            tier_weights = np.full(len(cands_idx), weight, dtype=np.float32)
            if len(cands_idx) >= min_cands:
                break

    # 13C isotope correction (+- 1.003355 Da)
    if len(cands_idx) < min_cands:
        for delta_iso in [-1.003355, 1.003355]:
            m_iso = m_neutral + delta_iso
            l_idx = np.searchsorted(cand_masses, m_iso * (1.0 - 20e-6))
            r_idx = np.searchsorted(cand_masses, m_iso * (1.0 + 20e-6))
            if r_idx > l_idx:
                iso_cands = np.arange(l_idx, r_idx)
                iso_w = np.full(len(iso_cands), 0.30, dtype=np.float32)
                # Combine unique
                existing = set(cands_idx)
                new_mask = [ci not in existing for ci in iso_cands]
                if any(new_mask):
                    cands_idx = np.concatenate([cands_idx, iso_cands[new_mask]])
                    tier_weights = np.concatenate([tier_weights, iso_w[new_mask]])

    # Nearest neighbors by exact mass to guarantee at least min_cands unique candidates
    if len(cands_idx) < min_cands:
        center_idx = int(np.searchsorted(cand_masses, m_neutral))
        half_win = max(min_cands, 30)
        l_idx = max(0, center_idx - half_win)
        r_idx = min(len(cand_masses), center_idx + half_win)
        nn_cands = np.arange(l_idx, r_idx)
        nn_w = np.full(len(nn_cands), 0.10, dtype=np.float32)
        existing = set(cands_idx)
        new_mask = [ci not in existing for ci in nn_cands]
        if any(new_mask):
            cands_idx = np.concatenate([cands_idx, nn_cands[new_mask]])
            tier_weights = np.concatenate([tier_weights, nn_w[new_mask]])

    return cands_idx, tier_weights


# ── 6. Leak-Free Candidate Feature Vector ───────────────────────────────────────
def extract_unprivileged_candidate_features(
    cand_mass: float,
    m_neutral_obs: float,
    tier_w: float,
    precursor_mz_obs: float,
    lib_hit: dict[str, Any] | None,
) -> list[float]:
    """10-D unprivileged feature vector constructed strictly from observable inputs.
    
    Zero oracle information: no ground-truth formula, no ground-truth molecule ID,
    no search on true target mass. Symmetric between positive and negative candidates.
    """
    ppm_err = abs(cand_mass - m_neutral_obs) / m_neutral_obs * 1e6

    if lib_hit is not None and lib_hit.get("cos", 0.0) >= 0.10:
        cos_val = min(1.0, max(0.0, float(lib_hit["cos"])))
        n_p = float(lib_hit.get("n_peaks", 0))
        p_norm = min(1.0, n_p / 15.0)
        p_ratio = min(1.0, n_p / (1.0 + 15.0 * cos_val))
        ce_d = lib_hit.get("ce_diff", float("nan"))
        ce_agree = float(np.exp(-abs(ce_d) / 20.0)) if np.isfinite(ce_d) else 0.60
        n_s = lib_hit.get("n_supporting", 1)
        multi_n = min(1.0, math.log1p(max(1, n_s)) / math.log1p(5))
        src_c = 1.0 if lib_hit.get("source_count", 1) >= 2 else 0.0
        has_lib = 1.0
    else:
        cos_val, p_norm, p_ratio, ce_agree, multi_n, src_c, has_lib = 0.0, 0.0, 0.0, 0.60, 0.0, 0.0, 0.0

    return [
        cos_val,
        p_norm,
        p_ratio,
        ce_agree,
        multi_n,
        src_c,
        min(3.0, ppm_err / 20.0),
        tier_w,
        precursor_mz_obs / 1000.0,
        has_lib,
    ]
