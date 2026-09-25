"""Diagnose: (1) Protocol C identical ranks across variants; (2) pair-filter noise."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def diagnose_protocol_c_logic() -> None:
    """Unit-level: does protocol_c_ranks actually use accumulator scores?"""
    from src.search.spectral_search import Accumulator, protocol_c_ranks

    # Minimal synthetic meta: 3 molecules, known masses via [M+H]+
    # mol 0: nm=100, mol 1: nm=100 (isobar competitor), mol 2: nm=300
    rows = []
    # Use adduct [M+H]+ => precursor = nm + 1.00728 approx; easier: use [M]+ radical?
    # Simplest: adduct that gives nm = precursor - 0 => use empty? parse needs [..]+
    # [M+H]+: ion = mz + e; neutral = ion - 1.00728 => mz = nm + 1.00728 - e
    # Just set precursor so neutral_mass ≈ target by solving via neutral_mass inverse:
    # neutral = (charge*mz + e - adds)/1 with [M+H]: adds=1.00728
    # mz = (nm + adds - e) for pos
    from src.core.adducts import ELECTRON_MASS, ATOMIC_MASSES

    H = ATOMIC_MASSES["H"] - 0.0  # proton mass approx uses H atom - e in formula
    # neutral = (mz + e - H_mass)/1 => mz = nm + H_mass - e
    def mz_for(nm: float) -> float:
        return nm + 1.00728 - ELECTRON_MASS

    for mid, nm, pol in [(0, 100.0, "positive"), (1, 100.0, "positive"), (2, 300.0, "positive")]:
        for _ in range(2):
            rows.append(
                {
                    "mol_id": mid,
                    "inchikey": f"IK{mid}",
                    "precursor_mz": mz_for(nm),
                    "adduct": "[M+H]+",
                    "ionization_mode": pol,
                }
            )
    meta = pd.DataFrame(rows)

    # Query: true molecule = mol 0, mass 100
    queries = pd.DataFrame(
        [
            {
                "precursor_mz": mz_for(100.0),
                "adduct": "[M+H]+",
                "ionization_mode": "positive",
                "inchikey": "IK0",
                "row_id": 0,
            }
        ]
    )

    true_mol = np.array([0], dtype=np.int32)
    exclude = {"IK0", "IK1", "IK2"}  # all held out => pure structure ranking

    # Acc A: mol 1 (isobar competitor) has high spectral score 0.9
    # Acc B: same layout but score 0.1 — ranks must differ if scores are used
    def make_acc(score_1: float) -> Accumulator:
        acc = Accumulator(n_queries=1, cap=8)
        acc.update(0, np.array([1, 2], dtype=np.int32), np.array([score_1, 0.5], dtype=np.float32))
        return acc

    acc_hi = make_acc(0.9)
    acc_lo = make_acc(0.1)

    ranks_hi, spec_hi = protocol_c_ranks(meta, queries, acc_hi, true_mol, exclude, ppm=20.0)
    ranks_lo, spec_lo = protocol_c_ranks(meta, queries, acc_lo, true_mol, exclude, ppm=20.0)

    print("[diag] Protocol C unit test (scores MUST change rank of true mol 0 vs competitor 1):")
    print(f"  acc score(competitor)=0.9 -> true rank={ranks_hi[0]}, spectral_rank={spec_hi[0]}")
    print(f"  acc score(competitor)=0.1 -> true rank={ranks_lo[0]}, spectral_rank={spec_lo[0]}")
    if ranks_hi[0] == ranks_lo[0]:
        print("  BUG CONFIRMED: protocol_c_ranks IGNORES accumulator scores for ranking")
    else:
        print("  OK: scores affect ranking")

    # Also: does polarity filter drop candidates? Query pos, molecule majority neg
    meta2 = meta.copy()
    # mol 0 majority negative but query positive
    meta2.loc[meta2["mol_id"] == 0, "ionization_mode"] = "negative"
    meta2.loc[meta2["mol_id"] == 0, "precursor_mz"] = meta2.loc[meta2["mol_id"] == 0, "precursor_mz"]
    # For neg mode neutral mass differs — keep [M+H]+ labels but pol column neg; our table uses ionization_mode only
    q2 = queries.copy()
    r2, _ = protocol_c_ranks(meta2, q2, acc_hi, true_mol, exclude, ppm=20.0)
    print(f"[diag] polarity mismatch (true mol majority=neg, query=pos): true rank={r2[0]} (0 = dropped)")


def diagnose_pair_filter_regression() -> None:
    """Compare primary-only vs always-fallback candidate volumes on synthetic pairs."""
    from src.search.candidate_filter import ISOTOPE_DELTA, PPM_FALLBACK

    rng = np.random.default_rng(0)
    n = 10000
    nm_q = rng.uniform(100, 500, n)
    # Library: 70% primary match, 10% only in 50ppm, 10% only isotope, 10% none
    nm_lib = nm_q.copy()
    kind = rng.choice(["prim", "wide", "iso", "none"], size=n, p=[0.7, 0.1, 0.1, 0.1])
    tol_p = nm_q * 20 / 1e6
    tol_w = nm_q * 50 / 1e6
    nm_lib = np.where(kind == "wide", nm_q + 35e-6 * nm_q, nm_lib)
    nm_lib = np.where(kind == "iso", nm_q + ISOTOPE_DELTA, nm_lib)
    nm_lib = np.where(kind == "none", nm_q + 1000.0, nm_lib)

    primary = np.abs(nm_q - nm_lib) <= tol_p
    primary |= np.abs((nm_q - ISOTOPE_DELTA) - nm_lib) <= tol_p
    primary |= np.abs((nm_q + ISOTOPE_DELTA) - nm_lib) <= tol_p
    fallback = np.abs(nm_q - nm_lib) <= nm_q * PPM_FALLBACK / 1e6
    fallback |= np.abs((nm_q - ISOTOPE_DELTA) - nm_lib) <= nm_q * PPM_FALLBACK / 1e6
    fallback |= np.abs((nm_q + ISOTOPE_DELTA) - nm_lib) <= nm_q * PPM_FALLBACK / 1e6

    always = primary | fallback
    # per-query: fallback only if no primary for that query (here 1 pair per query)
    per_q = np.where(primary, primary, fallback)

    print("[diag] pair-filter candidate retention (synthetic):")
    print(f"  primary-only:     {primary.mean():.3f}")
    print(f"  always|fallback:  {always.mean():.3f}  (+{(always & ~primary).mean():.3f} extra noise pairs)")
    print(f"  per-query fb:     {per_q.mean():.3f}  (+{(per_q & ~primary).mean():.3f} recovery pairs)")


if __name__ == "__main__":
    diagnose_protocol_c_logic()
    diagnose_pair_filter_regression()
