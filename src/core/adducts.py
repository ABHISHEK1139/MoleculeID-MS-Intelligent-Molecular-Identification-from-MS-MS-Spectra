"""Adduct parsing and neutral mass computation.

Moved from src/adducts.py to src/core/adducts.py as part of
the Phase 1 codebase restructuring.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

ELECTRON_MASS = 0.00054858

# Monoisotopic masses of neutral atoms.
ATOMIC_MASSES = {
    "H": 1.00782503223,
    "D": 2.01410177811,
    "He": 4.00260325413,
    "Li": 7.0160034366,
    "Be": 9.012183065,
    "B": 11.00930536,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Ne": 19.9924401762,
    "Na": 22.9897692820,
    "Mg": 23.9850416996,
    "Al": 26.98153863,
    "Si": 27.9769265327,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Ar": 39.9623831237,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    "Ti": 47.94794198,
    "V": 50.94395704,
    "Cr": 51.94050623,
    "Mn": 54.9380443,
    "Fe": 55.93493633,
    "Co": 58.93319429,
    "Ni": 57.93534241,
    "Cu": 62.92959772,
    "Zn": 63.92914201,
    "Ga": 68.9255735,
    "Ge": 73.921177761,
    "As": 74.92159457,
    "Se": 79.9165218,
    "Br": 78.9183376,
    "Kr": 83.9114977,
    "Rb": 86.909180531,
    "Sr": 87.9056125,
    "Zr": 89.9047044,
    "Mo": 97.90540482,
    "Ru": 101.9043493,
    "Rh": 102.9055043,
    "Pd": 105.9034804,
    "Ag": 106.9050916,
    "Cd": 113.9033585,
    "In": 114.9038785,
    "Sn": 119.9021947,
    "Sb": 120.903812,
    "Te": 129.906222748,
    "I": 126.9044719,
    "Xe": 131.9041550856,
    "Cs": 132.905451961,
    "Ba": 137.9052472,
    "Pt": 194.9647911,
    "Au": 196.96656879,
    "Hg": 201.970643,
    "Tl": 204.9744275,
    "Pb": 207.9766521,
    "Bi": 208.9803987,
    "U": 238.05078826,
}

_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula: str) -> float | None:
    """Monoisotopic mass of a neutral elemental formula such as 'C2H4O2'."""
    if not formula:
        return None
    pos = 0
    mass = 0.0
    for match in _FORMULA_TOKEN_RE.finditer(formula):
        if match.start() != pos:
            return None
        element, count_str = match.group(1), match.group(2)
        if element not in ATOMIC_MASSES:
            return None
        count = int(count_str) if count_str else 1
        mass += ATOMIC_MASSES[element] * count
        pos = match.end()
    if pos != len(formula):
        return None
    return mass


@dataclass(frozen=True)
class AdductSpec:
    n_m: int
    adds_mass: float
    losses_mass: float
    charge: int
    positive: bool


_ADDUCT_CACHE: dict[str, AdductSpec | None] = {}


def parse_adduct(adduct: str | None) -> AdductSpec | None:
    """Parse an adduct string into composition parts.

    Returns None for unrecognized forms such as '[Cat]2+'.
    """
    if adduct is None:
        return None
    adduct = str(adduct).strip()
    if adduct in _ADDUCT_CACHE:
        return _ADDUCT_CACHE[adduct]
    spec = _parse_adduct_uncached(adduct)
    _ADDUCT_CACHE[adduct] = spec
    return spec


def _parse_adduct_uncached(adduct: str) -> AdductSpec | None:
    match = re.fullmatch(r"\[([^\]]+)\](\d*)([+-])", adduct)
    if not match:
        return None
    core, charge_str, sign = match.group(1), match.group(2), match.group(3)
    charge = int(charge_str) if charge_str else 1
    if charge == 0:
        return None
    positive = sign == "+"

    core_match = re.fullmatch(r"(\d*)M(.*)", core)
    if core_match is None:
        return None
    n_m = int(core_match.group(1)) if core_match.group(1) else 1
    rest = core_match.group(2)
    if n_m == 0:
        return None

    parts = re.findall(r"([+-])([A-Za-z0-9]+)", rest)
    if not parts:
        if rest == "":
            adds_mass = 0.0
            losses_mass = 0.0
        else:
            return None
    else:
        # Ensure the tokenized parts reconstruct the string exactly.
        rebuilt = "".join(sign_i + formula for sign_i, formula in parts)
        if rebuilt != rest:
            return None
        adds_mass = 0.0
        losses_mass = 0.0
        for sign_i, formula in parts:
            mass = _group_mass(formula)
            if mass is None:
                return None
            if sign_i == "+":
                adds_mass += mass
            else:
                losses_mass += mass

    return AdductSpec(
        n_m=n_m,
        adds_mass=adds_mass,
        losses_mass=losses_mass,
        charge=charge,
        positive=positive,
    )


def _group_mass(formula: str) -> float | None:
    """Mass of an adduct group; allows a leading count such as '2H'."""
    direct = formula_mass(formula)
    if direct is not None:
        return direct
    match = re.fullmatch(r"(\d+)([A-Z][a-z]?(?:\d*[A-Z][a-z]?)*)", formula)
    if match:
        inner = formula_mass(match.group(2))
        if inner is None:
            return None
        return inner * int(match.group(1))
    return None


def neutral_mass(precursor_mz: float, adduct: str | None) -> float | None:
    spec = parse_adduct(adduct)
    if spec is None:
        return None
    signed = spec.charge * float(precursor_mz)
    electron = spec.charge * ELECTRON_MASS
    if spec.positive:
        ion = signed + electron
    else:
        ion = signed - electron
    neutral = (ion - spec.adds_mass + spec.losses_mass) / spec.n_m
    return neutral


def neutral_mass_series(precursor_mz: pd.Series, adduct: pd.Series) -> np.ndarray:
    """Vectorized neutral-mass computation; NaN for unrecognized adducts."""
    precursor_arr = precursor_mz.to_numpy(dtype=np.float64, na_value=np.nan)
    adduct_arr = adduct.astype(str).to_numpy()
    out = np.full(precursor_arr.shape, np.nan, dtype=np.float64)
    unique_adducts = pd.unique(adduct_arr)
    for adduct_value in unique_adducts:
        spec = parse_adduct(adduct_value)
        mask = adduct_arr == adduct_value
        if spec is None:
            continue
        signed = spec.charge * precursor_arr[mask]
        electron = spec.charge * ELECTRON_MASS
        ion = signed + electron if spec.positive else signed - electron
        out[mask] = (ion - spec.adds_mass + spec.losses_mass) / spec.n_m
    bad = ~np.isfinite(out) | (out <= 0)
    out[bad] = np.nan
    return out
