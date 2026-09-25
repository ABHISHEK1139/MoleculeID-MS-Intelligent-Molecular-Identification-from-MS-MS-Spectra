"""Chemical formula engine and de novo formula generation.

Implements Kind & Fiehn's Seven Golden Rules (2007) and Senior's valence rules
for physics-informed candidate formula generation and validation in metabolomics MS/MS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.core.adducts import ATOMIC_MASSES

# Standard valences for Senior's rules
ELEMENT_VALENCES: dict[str, list[int]] = {
    "H": [1],
    "C": [4],
    "N": [3, 5],
    "O": [2],
    "F": [1],
    "P": [3, 5],
    "S": [2, 4, 6],
    "Cl": [1],
    "Br": [1],
    "I": [1],
}

_FORMULA_PARSER_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula_str: str) -> dict[str, int]:
    """Parse a molecular formula string into an elemental composition dictionary.

    Example: 'C20H15N3O2' -> {'C': 20, 'H': 15, 'N': 3, 'O': 2}
    """
    if not formula_str or not isinstance(formula_str, str):
        return {}
    pos = 0
    comp: dict[str, int] = {}
    for match in _FORMULA_PARSER_RE.finditer(formula_str):
        if match.start() != pos:
            return {}  # Non-contiguous or invalid character
        element = match.group(1)
        count_str = match.group(2)
        count = int(count_str) if count_str else 1
        comp[element] = comp.get(element, 0) + count
        pos = match.end()
    if pos != len(formula_str):
        return {}
    return comp


def format_formula(comp: dict[str, int]) -> str:
    """Format an elemental composition dictionary into Hill system order.

    Rule: C first, then H, then remaining elements in alphabetical order.
    """
    if not comp:
        return ""
    parts: list[str] = []

    # C first if present
    if "C" in comp and comp["C"] > 0:
        c_count = comp["C"]
        parts.append(f"C{c_count if c_count > 1 else ''}")

    # H second if present and C is present
    if "H" in comp and comp["H"] > 0:
        h_count = comp["H"]
        parts.append(f"H{h_count if h_count > 1 else ''}")

    # Remaining elements in alphabetical order
    remaining = sorted([el for el in comp if el not in ("C", "H") and comp[el] > 0])
    for el in remaining:
        cnt = comp[el]
        parts.append(f"{el}{cnt if cnt > 1 else ''}")

    return "".join(parts)


def formula_monoisotopic_mass(comp: dict[str, int] | str) -> float | None:
    """Calculate exact monoisotopic neutral mass from composition or string."""
    if isinstance(comp, str):
        comp = parse_formula(comp)
    if not comp:
        return None
    total_mass = 0.0
    for el, count in comp.items():
        if el not in ATOMIC_MASSES:
            return None
        total_mass += ATOMIC_MASSES[el] * count
    return total_mass


def calculate_rdbe(comp: dict[str, int]) -> float:
    """Calculate Ring Double Bond Equivalents (degree of unsaturation).

    Formula:
        RDBE = C + 1 - (H / 2) + (N / 2) + (P / 2) - (Halogens / 2)
    """
    c = comp.get("C", 0)
    h = comp.get("H", 0)
    n = comp.get("N", 0)
    p = comp.get("P", 0)
    halogens = comp.get("F", 0) + comp.get("Cl", 0) + comp.get("Br", 0) + comp.get("I", 0)
    return float(c + 1 - (h / 2.0) + (n / 2.0) + (p / 2.0) - (halogens / 2.0))


def validate_seven_golden_rules(comp: dict[str, int]) -> tuple[bool, str]:
    """Validate elemental composition against Seven Golden Rules & Senior's rules.

    Returns:
        (is_valid, failure_reason)
    """
    c = comp.get("C", 0)
    if c <= 0:
        return False, "Zero or negative carbon count"

    h = comp.get("H", 0)
    n = comp.get("N", 0)
    o = comp.get("O", 0)
    p = comp.get("P", 0)
    s = comp.get("S", 0)
    hal = comp.get("F", 0) + comp.get("Cl", 0) + comp.get("Br", 0) + comp.get("I", 0)

    # 1. Senior's Rule 2: Sum of odd valences must be even
    # Odd valence elements: H (1), N (3 or 5 -> odd), P (3 or 5 -> odd), halogens (1 -> odd)
    odd_count = h + n + p + hal
    if odd_count % 2 != 0:
        return False, f"Senior's 2nd rule violation: sum of odd valences is odd ({odd_count})"

    # 2. Senior's Rule 1: Sum of maximum valences >= 2 * max(valence)
    max_val_sum = (
        c * 4 + h * 1 + n * 5 + o * 2 + p * 5 + s * 6 + hal * 1
    )
    max_single_val = 4 if c > 0 else 1
    if s > 0:
        max_single_val = max(max_single_val, 6)
    elif p > 0 or n > 0:
        max_single_val = max(max_single_val, 5)

    if max_val_sum < 2 * max_single_val:
        return False, "Senior's 1st rule violation: valence sum too low"

    # 3. RDBE (Degree of Unsaturation)
    rdbe = calculate_rdbe(comp)
    if rdbe < -0.5:
        return False, f"Negative RDBE ({rdbe:.1f})"
    if rdbe > 40.0:
        return False, f"Excessive RDBE ({rdbe:.1f} > 40.0)"

    # 4. Hydrogen-to-Carbon ratio (0.1 to 3.2 for typical metabolites)
    hc = h / c
    if hc < 0.1 or hc > 3.2:
        return False, f"H/C ratio out of bounds ({hc:.2f} not in [0.1, 3.2])"

    # 5. Heteroatom-to-Carbon ratio bounds (Kind & Fiehn, 2007)
    if (n / c) > 1.5:
        return False, f"N/C ratio too high ({n/c:.2f} > 1.5)"
    if (o / c) > 1.5:
        return False, f"O/C ratio too high ({o/c:.2f} > 1.5)"
    if (p / c) > 0.4:
        return False, f"P/C ratio too high ({p/c:.2f} > 0.4)"
    if (s / c) > 0.8:
        return False, f"S/C ratio too high ({s/c:.2f} > 0.8)"
    if (hal / c) > 1.5:
        return False, f"Halogen/C ratio too high ({hal/c:.2f} > 1.5)"

    return True, "Valid"


def generate_candidate_formulas(
    target_mass: float,
    ppm_tol: float = 20.0,
    allowed_elements: list[str] | None = None,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """Generate candidate molecular formulas matching target neutral mass.

    Uses bounded integer search pruned by Kind & Fiehn's Seven Golden Rules.

    Args:
        target_mass: Neutral monoisotopic mass in Daltons
        ppm_tol: Mass tolerance in parts-per-million
        allowed_elements: Elements to consider (default: C, H, N, O, P, S)
        max_results: Maximum candidate formulas to return

    Returns:
        List of dicts: {"formula": str, "mass": float, "ppm_error": float, "rdbe": float}
    """
    if allowed_elements is None:
        allowed_elements = ["C", "H", "N", "O", "P", "S"]

    delta_mass = target_mass * ppm_tol / 1e6
    min_mass = target_mass - delta_mass
    max_mass = target_mass + delta_mass

    mass_c = ATOMIC_MASSES["C"]
    mass_h = ATOMIC_MASSES["H"]
    mass_n = ATOMIC_MASSES["N"]
    mass_o = ATOMIC_MASSES["O"]
    mass_p = ATOMIC_MASSES["P"]
    mass_s = ATOMIC_MASSES["S"]

    max_c = int(max_mass / mass_c) + 1
    max_p = int(max_mass / mass_p) + 1 if "P" in allowed_elements else 1
    max_s = int(max_mass / mass_s) + 1 if "S" in allowed_elements else 1
    max_n = int(max_mass / mass_n) + 1 if "N" in allowed_elements else 1
    max_o = int(max_mass / mass_o) + 1 if "O" in allowed_elements else 1

    # Cap heteroatoms to realistic metabolite bounds for target mass
    max_p = min(max_p, 3 if target_mass < 500 else 6)
    max_s = min(max_s, 3 if target_mass < 500 else 5)
    max_n = min(max_n, 10 if target_mass < 500 else 18)
    max_o = min(max_o, 12 if target_mass < 500 else 24)

    results: list[dict[str, Any]] = []

    for p in range(max_p if "P" in allowed_elements else 1):
        m_p = p * mass_p
        if m_p > max_mass:
            break

        for s in range(max_s if "S" in allowed_elements else 1):
            m_ps = m_p + s * mass_s
            if m_ps > max_mass:
                break

            for n in range(max_n if "N" in allowed_elements else 1):
                m_psn = m_ps + n * mass_n
                if m_psn > max_mass:
                    break

                for o in range(max_o if "O" in allowed_elements else 1):
                    m_psno = m_psn + o * mass_o
                    if m_psno > max_mass:
                        break

                    rem_after_hetero = target_mass - m_psno
                    if rem_after_hetero <= 0:
                        continue

                    # Bounded C range
                    min_c_val = max(1, int((min_mass - m_psno) / (mass_c + 3.2 * mass_h)))
                    max_c_val = min(max_c, int((max_mass - m_psno) / (mass_c + 0.1 * mass_h)) + 1)

                    for c in range(min_c_val, max_c_val + 1):
                        m_core = m_psno + c * mass_c
                        h_mass_needed = target_mass - m_core
                        if h_mass_needed < 0:
                            break

                        # Solve exact H count
                        approx_h = h_mass_needed / mass_h
                        h_low = int(approx_h)
                        h_candidates = [h_low, h_low + 1]

                        for h in h_candidates:
                            if h < 0:
                                continue
                            total_calc_mass = m_core + h * mass_h
                            err_ppm = abs(total_calc_mass - target_mass) / target_mass * 1e6
                            if err_ppm <= ppm_tol:
                                comp = {"C": c, "H": h, "N": n, "O": o, "P": p, "S": s}
                                valid, _ = validate_seven_golden_rules(comp)
                                if valid:
                                    f_str = format_formula(comp)
                                    results.append({
                                        "formula": f_str,
                                        "composition": {k: v for k, v in comp.items() if v > 0},
                                        "mass": float(total_calc_mass),
                                        "ppm_error": float(err_ppm),
                                        "rdbe": float(calculate_rdbe(comp)),
                                    })
                                    if len(results) >= max_results:
                                        results.sort(key=lambda r: r["ppm_error"])
                                        return results

    results.sort(key=lambda r: r["ppm_error"])
    return results
