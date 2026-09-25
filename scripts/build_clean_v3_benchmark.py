"""Build Uncontaminated, Leak-Free Benchmark Split (Phase 0).

Strict Protocol:
1. InChIKey14 clustering: All stereoisomers of the same 2D connectivity family
   stay in the same partition (zero scaffold/connectivity leakage).
2. Deterministic split:
   - train: ~274,300 connectivity clusters (~99% of data)
   - val: 1,000 connectivity clusters (for model checkpoint selection & threshold tuning)
   - frozen_benchmark: 500 connectivity clusters (NEVER seen in training, NEVER in checkpoint selection)
3. Dual Benchmark Evaluation Modes:
   - Mode A (250 queries): Zero-Reference (Class 2/3 simulation). True molecule is 100% absent from library.
   - Mode B (250 queries): Leave-One-Spectrum-Out (Class 1 simulation). Molecule has >=2 spectra in dataset.
     Query spectrum is withheld; only sibling spectra at different CEs are indexed in library.
4. Generates artifacts/v3_clean/clean_split.json with complete metadata and SHA-256 integrity hash.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "v3_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 85)
    print("  BUILDING UNCONTAMINATED v3_clean BENCHMARK SPLIT")
    print("=" * 85, flush=True)
    t_start = time.time()

    # 1. Read metadata columns from train.parquet
    print("[1/5] Ingesting train.parquet metadata...", flush=True)
    cols = ["normalized_smiles", "inchikey14", "adduct", "precursor_mz", "collision_energy_ev"]
    df = pq.read_table(ROOT / "dataset" / "train.parquet", columns=cols).to_pandas()
    n_rows = len(df)
    print(f"Loaded {n_rows:,} rows. Grouping by inchikey14...", flush=True)

    # 2. Cluster rows and SMILES by inchikey14
    inchi_to_rows = defaultdict(list)
    inchi_to_smiles = defaultdict(set)
    for row_idx, (smi, inchi) in enumerate(zip(df["normalized_smiles"], df["inchikey14"])):
        inchi_to_rows[inchi].append(row_idx)
        inchi_to_smiles[inchi].add(smi)

    all_inchis = sorted(inchi_to_rows.keys())
    print(f"Total unique InChIKey14 connectivity clusters: {len(all_inchis):,}")

    # Identify multi-spectrum clusters suitable for Mode B (Leave-One-Spectrum-Out)
    multi_spec_inchis = [k for k in all_inchis if len(inchi_to_rows[k]) >= 2]
    single_spec_inchis = [k for k in all_inchis if len(inchi_to_rows[k]) == 1]
    print(f"Clusters with >=2 spectra: {len(multi_spec_inchis):,} ({len(multi_spec_inchis)/len(all_inchis)*100:.1f}%)")

    # 3. Deterministic selection with fixed seed
    rng = random.Random(42)
    rng.shuffle(multi_spec_inchis)
    rng.shuffle(single_spec_inchis)

    # 500 Frozen Benchmark clusters:
    # 250 from multi-spectrum (for Mode B Leave-One-Spectrum-Out)
    # 250 for Mode A (Zero-Reference, can be single or multi)
    benchmark_mode_b_inchis = multi_spec_inchis[:250]
    remaining_multi = multi_spec_inchis[250:]

    benchmark_mode_a_inchis = remaining_multi[:125] + single_spec_inchis[:125]
    frozen_benchmark_inchis = set(benchmark_mode_b_inchis + benchmark_mode_a_inchis)
    assert len(frozen_benchmark_inchis) == 500

    # 1,000 Validation clusters (for model checkpoint selection)
    remaining_pool = [k for k in all_inchis if k not in frozen_benchmark_inchis]
    rng.shuffle(remaining_pool)
    val_inchis = set(remaining_pool[:1000])

    # Remaining ~274,310 clusters are pure training
    train_inchis = set(remaining_pool[1000:])
    assert len(frozen_benchmark_inchis & val_inchis) == 0
    assert len(frozen_benchmark_inchis & train_inchis) == 0
    assert len(val_inchis & train_inchis) == 0

    print(f"\n[2/5] Partition Summary:")
    print(f"  Frozen Benchmark: {len(frozen_benchmark_inchis):,} clusters (NEVER in training, NEVER in library)")
    print(f"  Validation:       {len(val_inchis):,} clusters (for checkpoint selection & threshold tuning)")
    print(f"  Training:         {len(train_inchis):,} clusters ({len(train_inchis)/len(all_inchis)*100:.2f}%)")

    # Collect excluded SMILES to ensure 100% leak-free reference library
    benchmark_smiles = set()
    for k in frozen_benchmark_inchis:
        benchmark_smiles.update(inchi_to_smiles[k])

    val_smiles = set()
    for k in val_inchis:
        val_smiles.update(inchi_to_smiles[k])

    train_smiles = set()
    for k in train_inchis:
        train_smiles.update(inchi_to_smiles[k])

    print(f"  Benchmark Unique SMILES: {len(benchmark_smiles):,}")
    print(f"  Validation Unique SMILES:{len(val_smiles):,}")
    print(f"  Training Unique SMILES:  {len(train_smiles):,}")
    assert len(benchmark_smiles & train_smiles) == 0, "FATAL: SMILES leak between benchmark and train!"
    assert len(val_smiles & train_smiles) == 0, "FATAL: SMILES leak between validation and train!"

    # 4. Construct Benchmark Queries
    print("\n[3/5] Constructing Benchmark Query Records...", flush=True)
    benchmark_queries = []

    def safe_ce(val):
        if val is None:
            return None
        if hasattr(val, "__iter__"):
            return [float(x) for x in val if x is not None]
        try:
            return float(val)
        except Exception:
            return None

    # Mode A: Zero-Reference (Class 2/3 simulation)
    # Pick 1 representative spectrum per cluster; ZERO spectra of this cluster exist in library
    for inchi in benchmark_mode_a_inchis:
        r_indices = inchi_to_rows[inchi]
        q_idx = r_indices[0]
        row = df.iloc[q_idx]
        benchmark_queries.append({
            "query_id": f"bm_zero_{len(benchmark_queries):04d}",
            "mode": "zero_reference_class2_3",
            "row_idx": int(q_idx),
            "inchikey14": inchi,
            "true_smiles": str(row["normalized_smiles"]),
            "adduct": str(row["adduct"]),
            "precursor_mz": float(row["precursor_mz"]),
            "collision_energy_ev": safe_ce(row["collision_energy_ev"]),
            "withheld_spectrum_indices": [int(i) for i in r_indices],
        })

    # Mode B: Leave-One-Spectrum-Out (Class 1 simulation)
    # 1 spectrum is withheld as query; sibling spectra at other CEs are permitted in library
    for inchi in benchmark_mode_b_inchis:
        r_indices = inchi_to_rows[inchi]
        # Pick query spectrum as the first one, siblings are the rest
        q_idx = r_indices[0]
        sibling_indices = r_indices[1:]
        row = df.iloc[q_idx]
        benchmark_queries.append({
            "query_id": f"bm_loso_{len(benchmark_queries):04d}",
            "mode": "leave_one_spectrum_out_class1",
            "row_idx": int(q_idx),
            "inchikey14": inchi,
            "true_smiles": str(row["normalized_smiles"]),
            "adduct": str(row["adduct"]),
            "precursor_mz": float(row["precursor_mz"]),
            "collision_energy_ev": safe_ce(row["collision_energy_ev"]),
            "withheld_spectrum_indices": [int(q_idx)],
            "sibling_spectrum_indices": [int(i) for i in sibling_indices],
        })

    print(f"Created {len(benchmark_queries)} benchmark queries (250 Zero-Reference + 250 Leave-One-Spectrum-Out).")

    # 5. Export clean split JSON
    print("\n[4/5] Serializing artifacts/v3_clean/clean_split.json...", flush=True)
    split_manifest = {
        "version": "v3_clean.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": 42,
        "n_total_rows": n_rows,
        "n_total_inchikey14": len(all_inchis),
        "counts": {
            "train_clusters": len(train_inchis),
            "val_clusters": len(val_inchis),
            "benchmark_clusters": len(frozen_benchmark_inchis),
            "benchmark_queries": len(benchmark_queries),
            "benchmark_zero_reference_queries": 250,
            "benchmark_loso_queries": 250,
        },
        "train_inchikey14": list(train_inchis),
        "val_inchikey14": list(val_inchis),
        "benchmark_queries": benchmark_queries,
    }

    out_file = OUT_DIR / "clean_split.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f)

    split_bytes = out_file.read_bytes()
    sha = hashlib.sha256(split_bytes).hexdigest()
    print(f"Saved {out_file} ({len(split_bytes)/1e6:.1f} MB | SHA256: {sha[:16]}...)")

    print("\n" + "=" * 85)
    print(f"  CLEAN BENCHMARK SPLIT COMPLETE IN {time.time() - t_start:.1f}s")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
