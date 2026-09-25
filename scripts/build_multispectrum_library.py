"""Build Multi-Spectrum Reference Library for CASMI 2026 Stage 1.5.

Instead of keeping one representative spectrum per molecule (Stage 1),
this builder preserves ALL spectra per molecule with collision energy
metadata. This allows the submission kernel to:
  1. Match query spectra against same-CE reference spectra (preferred).
  2. Aggregate scores across multiple CE matches per molecule.
  3. Use consensus across collision energies for robust ranking.

Output schema (sorted by neutral_mass for binary search):
  - normalized_smiles: str
  - neutral_mass: float64
  - precursor_mz: float32
  - adduct: str
  - collision_energy: float32 (normalized to eV, NaN if unknown)
  - polarity: int8 (+1 or -1)
  - ms2_mzs: list[float32]
  - ms2_intensities: list[float32]  (sqrt-scaled, L2-normalized)
  - num_peaks: int16
  - ingest_lib: str
"""
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.core.adducts import neutral_mass
from src.core.preprocessing import deisotope_peaks

print("=== Building Multi-Spectrum Reference Library (Stage 1.5) ===", flush=True)
t0 = time.time()

train_path = Path("dataset/train.parquet")
pf = pq.ParquetFile(train_path)
n_rgs = pf.num_row_groups
print(f"Total row groups in train.parquet: {n_rgs}", flush=True)

# ── Collision energy normalization ──────────────────────────────────────
_CE_NUM_RE = re.compile(r"[-+]?\d+\.?\d*")


def normalize_ce(ce_orig, ce_units) -> float:
    """Parse collision energy string to a single float in eV."""
    if pd.isna(ce_orig) or ce_orig == "":
        return float("nan")

    ce_str = str(ce_orig).strip()

    # Handle composite CE strings like "20,40,60" or "[20.0, 30.0]"
    # These represent merged spectra - use the middle value as representative
    nums = _CE_NUM_RE.findall(ce_str)
    if not nums:
        return float("nan")

    vals = [abs(float(x)) for x in nums]
    if len(vals) == 0:
        return float("nan")

    # Use median of the listed values
    ce_val = float(np.median(vals))

    units = str(ce_units).strip().lower() if pd.notna(ce_units) else ""
    # NCE is normalized collision energy (instrument-specific),
    # treat it as approximate eV equivalent
    if units in ("ev", "nce", "v", "unknown", ""):
        return ce_val

    return ce_val


# ── Library priority (higher = preferred when deduplicating) ────────────
LIB_PRIORITY = {
    "enveda-np-examples": 10,
    "riken": 8,
    "gnps": 7,
    "massbank": 6,
    "mona": 5,
    "spectraverse": 4,
    "msdial": 3,
    "pluskal_ms2": 2,
    "drug_plus": 2,
    "enveda-180": 1,
    "masaryk": 1,
}

# ── Deduplicate within a molecule at the same CE level ──────────────────
# Key: (smiles, ce_bin) -> best (priority, n_peaks, record)
# ce_bin is rounded CE to nearest 5 eV to group similar energies
CE_BIN_WIDTH = 5.0

# Track unique spectra per molecule+CE bin
best_per_mol_ce: dict[tuple[str, int], tuple[int, int, dict]] = {}

cols = [
    "normalized_smiles",
    "ingest_lib",
    "adduct",
    "ionization_mode",
    "precursor_mz",
    "num_peaks",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "collision_energy_orig",
    "collision_energy_orig_units",
]

total_spectra_seen = 0
total_spectra_kept = 0

for rg_idx in range(n_rgs):
    t_rg = time.time()
    tbl = pf.read_row_group(rg_idx, columns=cols)
    df = tbl.to_pandas()

    for row in df.itertuples(index=False):
        total_spectra_seen += 1
        smi = row.normalized_smiles
        if not smi or pd.isna(smi):
            continue

        n_p = int(row.num_peaks) if pd.notna(row.num_peaks) else len(row.ms2_mzs)
        if n_p < 3:
            continue

        lib = str(row.ingest_lib) if pd.notna(row.ingest_lib) else ""
        prio = LIB_PRIORITY.get(lib, 0)

        ce_val = normalize_ce(row.collision_energy_orig, row.collision_energy_orig_units)
        polarity = -1 if (pd.notna(row.ionization_mode) and "neg" in str(row.ionization_mode).lower()) else 1

        # Bin CE to nearest CE_BIN_WIDTH
        if np.isfinite(ce_val):
            ce_bin = int(round(ce_val / CE_BIN_WIDTH))
        else:
            ce_bin = -9999  # unknown CE bin

        key = (smi, ce_bin)
        curr = best_per_mol_ce.get(key)

        record = {
            "normalized_smiles": smi,
            "precursor_mz": float(row.precursor_mz),
            "adduct": str(row.adduct),
            "collision_energy": ce_val,
            "polarity": polarity,
            "ingest_lib": lib,
            "ms2_mzs": row.ms2_mzs,
            "ms2_intensities": row.ms2_normalized_intensities,
        }

        if curr is None:
            best_per_mol_ce[key] = (prio, n_p, record)
        else:
            curr_prio, curr_np, _ = curr
            # Prefer higher-priority library, then richer spectra (closer to 40 peaks)
            if prio > curr_prio or (prio == curr_prio and abs(n_p - 40) < abs(curr_np - 40)):
                best_per_mol_ce[key] = (prio, n_p, record)

    elapsed = time.time() - t_rg
    print(
        f"Row group {rg_idx + 1}/{n_rgs} processed in {elapsed:.1f}s. "
        f"Unique (mol, CE) pairs so far: {len(best_per_mol_ce):,}",
        flush=True,
    )

print(f"\nExtraction complete in {time.time() - t0:.1f}s.", flush=True)
print(f"Total spectra seen: {total_spectra_seen:,}", flush=True)
print(f"Unique (molecule, CE-bin) pairs: {len(best_per_mol_ce):,}", flush=True)

# Count unique molecules
unique_mols = set(k[0] for k in best_per_mol_ce.keys())
print(f"Unique molecules: {len(unique_mols):,}", flush=True)
avg_spectra = len(best_per_mol_ce) / max(len(unique_mols), 1)
print(f"Average spectra per molecule: {avg_spectra:.2f}", flush=True)

# ── Preprocess all spectra ──────────────────────────────────────────────
print("\nPreprocessing spectra (deisotoping, sqrt-scaling, L2-norm, top-60 peaks)...", flush=True)

MAX_PEAKS = 60
records = []

for (smi, ce_bin), (prio, n_p, rec) in best_per_mol_ce.items():
    m_neut = neutral_mass(rec["precursor_mz"], rec["adduct"])
    if m_neut is None or m_neut <= 0:
        continue

    mzs = np.asarray(rec["ms2_mzs"], dtype=np.float32)
    intens = np.asarray(rec["ms2_intensities"], dtype=np.float32)

    # Deisotope
    if mzs.size > 1:
        mzs, intens = deisotope_peaks(mzs, intens)

    # Sqrt intensity scaling
    intens = np.sqrt(np.maximum(intens, 0.0))

    # Unit L2 normalization
    norm = np.linalg.norm(intens)
    if norm > 0:
        intens = intens / norm
    else:
        continue

    # Top-60 peaks by intensity, then sort by m/z
    if mzs.size > MAX_PEAKS:
        top_idx = np.argsort(-intens)[:MAX_PEAKS]
        top_idx = top_idx[np.argsort(mzs[top_idx])]
        mzs = mzs[top_idx]
        intens = intens[top_idx]

    records.append({
        "normalized_smiles": smi,
        "neutral_mass": float(m_neut),
        "precursor_mz": float(rec["precursor_mz"]),
        "adduct": rec["adduct"],
        "collision_energy": float(rec["collision_energy"]) if np.isfinite(rec["collision_energy"]) else None,
        "polarity": int(rec["polarity"]),
        "ingest_lib": rec["ingest_lib"],
        "ms2_mzs": mzs.tolist(),
        "ms2_intensities": intens.tolist(),
        "num_peaks": int(mzs.size),
    })

df_ref = pd.DataFrame(records)
df_ref = df_ref.sort_values("neutral_mass").reset_index(drop=True)

# ── Save ────────────────────────────────────────────────────────────────
out_path = Path("kaggle_dataset/reference_library_multice.parquet")
df_ref.to_parquet(out_path, compression="zstd")

# Stats
n_mols_final = df_ref["normalized_smiles"].nunique()
specs_per_mol = df_ref.groupby("normalized_smiles").size()

print(f"\n{'='*60}")
print(f"Multi-Spectrum Reference Library Built Successfully")
print(f"{'='*60}")
print(f"Total reference spectra: {len(df_ref):,}")
print(f"Unique molecules:        {n_mols_final:,}")
print(f"Avg spectra/molecule:    {specs_per_mol.mean():.2f}")
print(f"Median spectra/molecule: {specs_per_mol.median():.1f}")
print(f"Max spectra/molecule:    {specs_per_mol.max()}")
print(f"File size:               {out_path.stat().st_size / 1e6:.2f} MB")
print(f"Output:                  {out_path}")
print(f"Total time:              {time.time() - t0:.1f}s")

# Compare with old single-spectrum library
old_path = Path("kaggle_dataset/reference_library.parquet")
if old_path.exists():
    old_df = pd.read_parquet(old_path, columns=["normalized_smiles"])
    print(f"\n--- Comparison with old single-spectrum library ---")
    print(f"Old library: {len(old_df):,} spectra ({old_df['normalized_smiles'].nunique():,} molecules)")
    print(f"New library: {len(df_ref):,} spectra ({n_mols_final:,} molecules)")
    print(f"Increase:    {len(df_ref) - len(old_df):,} additional spectra ({len(df_ref)/len(old_df):.2f}x)")
