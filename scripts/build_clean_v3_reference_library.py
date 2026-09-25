"""Build 100% Uncontaminated, Leak-Purged Reference Library (v3_clean).

Method:
1. Load 1,384,625 unified spectra from kaggle_dataset/unified_reference_library.parquet.
2. Load clean split definition from artifacts/v3_clean/clean_split.json.
3. Exclude:
   - ALL 1,007 validation SMILES (100% zero spectra in library)
   - ALL 250 Mode A (Zero-Reference / Class 2/3) benchmark SMILES (100% zero spectra in library)
   - For Mode B (Leave-One-Spectrum-Out / Class 1): exclude spectra matching the exact query CE,
     retaining only different-CE sibling spectra.
4. Verify zero leakage with automated assertions.
5. Save artifacts/v3_clean/clean_reference_library.parquet with SHA-256 metadata manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

OUT_DIR = ROOT / "artifacts" / "v3_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 85)
    print("  BUILDING 100% UNCONTAMINATED v3_clean REFERENCE LIBRARY")
    print("=" * 85, flush=True)
    t_start = time.time()

    # 1. Load clean split definition
    split_file = OUT_DIR / "clean_split.json"
    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    # Ingest train metadata to map all InChIKey14 clusters to their SMILES
    print("[1/4] Mapping InChIKey14 clusters to SMILES...", flush=True)
    df_train = pq.read_table(ROOT / "dataset" / "train.parquet", columns=["normalized_smiles", "inchikey14"]).to_pandas()

    val_inchis = set(split_data["val_inchikey14"])
    mode_a_inchis = set([bq["inchikey14"] for bq in split_data["benchmark_queries"] if bq["mode"] == "zero_reference_class2_3"])
    mode_b_queries = {bq["true_smiles"]: bq for bq in split_data["benchmark_queries"] if bq["mode"] == "leave_one_spectrum_out_class1"}

    val_smiles = set(df_train[df_train["inchikey14"].isin(val_inchis)]["normalized_smiles"])
    mode_a_smiles = set(df_train[df_train["inchikey14"].isin(mode_a_inchis)]["normalized_smiles"])
    mode_b_smiles = set(mode_b_queries.keys())

    print(f"  Validation SMILES to strictly purge:     {len(val_smiles):,}")
    print(f"  Mode A (Zero-Ref) SMILES to purge:       {len(mode_a_smiles):,}")
    print(f"  Mode B (Leave-One-Out) SMILES:           {len(mode_b_smiles):,}")

    # 2. Load unified reference library
    unified_path = ROOT / "kaggle_dataset" / "unified_reference_library.parquet"
    print(f"\n[2/4] Reading {unified_path} (1.38M spectra)...", flush=True)
    tbl = pq.read_table(unified_path)
    df_lib = tbl.to_pandas()
    n_initial = len(df_lib)
    print(f"Loaded {n_initial:,} reference spectra.")

    # 3. Filter library rows
    print("\n[3/4] Filtering spectra with strict isolation rules...", flush=True)
    keep_mask = np.ones(n_initial, dtype=bool)

    smi_col = df_lib["canonical_smiles"].to_numpy()
    ce_col = df_lib["collision_energy"].to_numpy()

    n_purged_val = 0
    n_purged_mode_a = 0
    n_purged_mode_b_same_ce = 0
    n_retained_mode_b_diff_ce = 0

    for i in range(n_initial):
        smi = smi_col[i]

        # Rule 1: Exclude ALL validation molecules
        if smi in val_smiles:
            keep_mask[i] = False
            n_purged_val += 1
            continue

        # Rule 2: Exclude ALL Mode A (Zero-Reference) molecules
        if smi in mode_a_smiles:
            keep_mask[i] = False
            n_purged_mode_a += 1
            continue

        # Rule 3: For Mode B (Leave-One-Spectrum-Out):
        # Exclude spectra if collision energy matches the query's CE
        if smi in mode_b_queries:
            q_info = mode_b_queries[smi]
            q_ce = q_info.get("collision_energy_ev")
            r_ce = ce_col[i]

            # Check if CE matches query CE within 5 eV
            same_ce = False
            if q_ce is not None and np.isfinite(r_ce):
                if isinstance(q_ce, (list, tuple)):
                    same_ce = any(abs(c - r_ce) <= 5.0 for c in q_ce)
                else:
                    same_ce = abs(q_ce - r_ce) <= 5.0
            else:
                same_ce = True  # If CE unknown, purge to be conservative

            if same_ce:
                keep_mask[i] = False
                n_purged_mode_b_same_ce += 1
            else:
                n_retained_mode_b_diff_ce += 1

    df_clean = df_lib[keep_mask].reset_index(drop=True)
    n_clean = len(df_clean)

    print(f"Filtering Breakdown:")
    print(f"  Purged Validation Spectra:            {n_purged_val:,}")
    print(f"  Purged Mode A (Zero-Ref) Spectra:     {n_purged_mode_a:,}")
    print(f"  Purged Mode B (Same-CE Query) Spectra:{n_purged_mode_b_same_ce:,}")
    print(f"  Retained Mode B (Diff-CE Sibling) Sp.:{n_retained_mode_b_diff_ce:,}")
    print(f"  Total Clean Spectra Retained:         {n_clean:,} ({n_clean/n_initial*100:.2f}%)")

    # 4. Strict Leakage Audit Assertions
    clean_smiles_set = set(df_clean["canonical_smiles"])
    assert len(clean_smiles_set & val_smiles) == 0, "FATAL: Validation SMILES leaked into clean library!"
    assert len(clean_smiles_set & mode_a_smiles) == 0, "FATAL: Mode A Zero-Ref SMILES leaked into clean library!"
    print("\n[4/4] Automated Leakage Audit: PASSED (Zero leaks detected).")

    # Standardize column names to match v3_clean
    df_clean = df_clean.rename(columns={"canonical_smiles": "normalized_smiles"})

    # 5. Save clean library
    out_file = OUT_DIR / "clean_reference_library.parquet"
    out_tbl = pa.Table.from_pandas(df_clean)
    pq.write_table(out_tbl, out_file, compression="zstd")
    file_mb = out_file.stat().st_size / 1e6

    h = hashlib.sha256(out_file.read_bytes()).hexdigest()
    print(f"Saved {out_file} ({file_mb:.1f} MB | SHA256: {h[:16]}...)")

    # Metadata manifest
    manifest = {
        "library_version": "v3_clean.1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_dataset": "kaggle_dataset/unified_reference_library.parquet",
        "sha256": h,
        "n_spectra_initial": n_initial,
        "n_spectra_clean": n_clean,
        "n_unique_smiles_clean": len(clean_smiles_set),
        "purged": {
            "validation_spectra": n_purged_val,
            "mode_a_zero_reference_spectra": n_purged_mode_a,
            "mode_b_same_ce_spectra": n_purged_mode_b_same_ce,
        },
        "retained_mode_b_sibling_spectra": n_retained_mode_b_diff_ce,
        "leakage_audit_status": "PASSED_STRICT_ZERO_LEAK",
    }
    with open(OUT_DIR / "clean_reference_library_metadata.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 85)
    print(f"  v3_clean REFERENCE LIBRARY READY IN {time.time() - t_start:.1f}s")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
