"""Build Curated Reference Spectral Library for CASMI 2026.

Extracts the single best representative MS/MS reference spectrum for every unique molecule
in train.parquet (277k structures). Preprocesses fragment peaks (deisotoping, sqrt-scaling,
top-60 peaks) and sorts by exact neutral mass for instant binary-search retrieval.
"""
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

print("=== Building Curated Reference Spectral Library ===", flush=True)
t0 = time.time()

train_path = ROOT / "dataset" / "train.parquet"
pf = pq.ParquetFile(train_path)
n_rgs = pf.num_row_groups
print(f"Total row groups in train.parquet: {n_rgs}", flush=True)

# Load validation molecules to prevent evidence leakage (C1)
val_mols_path = ROOT / "artifacts" / "canonical_benchmark" / "benchmark_contracts.json"
excluded_smiles: set[str] = set()
if val_mols_path.exists():
    try:
        import json
        with open(val_mols_path, "r", encoding="utf-8") as f:
            contracts = json.load(f)
            for q in contracts.get("benchmark_queries", []):
                if "true_smiles" in q:
                    excluded_smiles.add(q["true_smiles"])
            for q in contracts.get("tuning_queries", []):
                if "true_smiles" in q:
                    excluded_smiles.add(q["true_smiles"])
        print(f"Excluding {len(excluded_smiles)} validation/benchmark molecules from reference library to prevent evidence leakage.", flush=True)
    except Exception as e:
        print(f"Warning: Could not load exclusion set: {e}", flush=True)

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

# Best spectrum tracker per unique SMILES:
# smi -> (priority, num_peaks, adduct, precursor_mz, ms2_mzs, ms2_intensities)
best_spectra: dict[str, tuple[int, int, str, float, list[float], list[float]]] = {}

cols = [
    "normalized_smiles",
    "ingest_lib",
    "adduct",
    "precursor_mz",
    "num_peaks",
    "ms2_mzs",
    "ms2_normalized_intensities"
]

for rg_idx in range(n_rgs):
    t_rg = time.time()
    tbl = pf.read_row_group(rg_idx, columns=cols)
    df = tbl.to_pandas()
    
    for row in df.itertuples(index=False):
        smi = row.normalized_smiles
        if not smi or pd.isna(smi) or smi in excluded_smiles:
            continue
            
        n_p = int(row.num_peaks) if pd.notna(row.num_peaks) else len(row.ms2_mzs)
        if n_p < 3:  # Skip degenerate spectra
            continue
            
        lib = str(row.ingest_lib) if pd.notna(row.ingest_lib) else ""
        prio = LIB_PRIORITY.get(lib, 0)
        
        # We prefer higher priority library, and then closer to ideal peak count (~30-60)
        curr = best_spectra.get(smi)
        if curr is None:
            best_spectra[smi] = (prio, n_p, str(row.adduct), float(row.precursor_mz), row.ms2_mzs, row.ms2_normalized_intensities)
        else:
            curr_prio, curr_np, _, _, _, _ = curr
            if prio > curr_prio or (prio == curr_prio and abs(n_p - 40) < abs(curr_np - 40)):
                best_spectra[smi] = (prio, n_p, str(row.adduct), float(row.precursor_mz), row.ms2_mzs, row.ms2_normalized_intensities)

    print(f"Row group {rg_idx+1}/{n_rgs} processed in {time.time()-t_rg:.1f}s. Unique mols so far: {len(best_spectra):,}", flush=True)

print(f"\nExtraction complete in {time.time()-t0:.1f}s. Total unique molecules: {len(best_spectra):,}")
print("Preprocessing spectra (deisotoping, sqrt-scaling, top-60 peaks)...", flush=True)

records = []
for smi, (prio, n_p, adduct, prec_mz, raw_mzs, raw_intens) in best_spectra.items():
    m_neut = neutral_mass(prec_mz, adduct)
    if m_neut is None or m_neut <= 0:
        continue
        
    mzs = np.asarray(raw_mzs, dtype=np.float32)
    intens = np.asarray(raw_intens, dtype=np.float32)
    
    # Preprocess
    if mzs.size > 1:
        mzs, intens = deisotope_peaks(mzs, intens)
    
    # Sqrt intensity scaling + unit L2
    intens = np.sqrt(intens)
    norm = np.linalg.norm(intens)
    if norm > 0:
        intens = intens / norm
        
    # Top 60 peaks
    if mzs.size > 60:
        top_idx = np.argsort(-intens)[:60]
        # Keep sorted by m/z for binary search
        top_idx = top_idx[np.argsort(mzs[top_idx])]
        mzs = mzs[top_idx]
        intens = intens[top_idx]
        
    records.append({
        "normalized_smiles": smi,
        "neutral_mass": float(m_neut),
        "precursor_mz": float(prec_mz),
        "adduct": adduct,
        "ms2_mzs": mzs.tolist(),
        "ms2_intensities": intens.tolist(),
    })

df_ref = pd.DataFrame(records)
df_ref = df_ref.sort_values("neutral_mass").reset_index(drop=True)

out_path = Path("kaggle_dataset/reference_library.parquet")
df_ref.to_parquet(out_path, compression="zstd")
print(f"Saved {len(df_ref):,} reference spectra to {out_path}")
print(f"File size: {out_path.stat().st_size / 1e6:.2f} MB")
print(f"Total time: {time.time()-t0:.1f}s")
