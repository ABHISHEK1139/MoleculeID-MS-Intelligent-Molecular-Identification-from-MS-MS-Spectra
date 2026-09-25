"""Phase 3 Evaluation Gate: Comprehensive Candidate Recall & Pool Sizing on 500 Clean Benchmark Queries.

Compares:
- Baseline Legacy Catalog (276,940 structures from train.parquet only)
- Expanded Clean v3 Catalog (776,699 structures: Train + COCONUT + ChEBI + LIPID MAPS)

Metrics measured:
- Candidate Recall Overall (Is the true molecule in the retrieved pool?)
- Candidate Recall Mode A (Zero-Reference, Class 2/3 simulation)
- Candidate Recall Mode B (Leave-One-Spectrum-Out, Class 1 simulation)
- Median candidates/query
- P95 candidates/query
- Mean candidates/query
- New structures added outside training
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.preprocessing_v3 import neutral_mass, retrieve_candidates_progressive


def evaluate_catalog_recall(queries: list[dict], cand_df: pd.DataFrame, catalog_name: str) -> dict:
    cand_masses = cand_df["exact_mass"].to_numpy(dtype=np.float64)
    cand_smiles = cand_df["canonical_smiles"].to_numpy() if "canonical_smiles" in cand_df.columns else cand_df["normalized_smiles"].to_numpy()
    
    recalls = []
    mode_a_recalls = []
    mode_b_recalls = []
    pool_sizes = []
    mode_a_pool_sizes = []
    mode_b_pool_sizes = []
    
    tier_counts = {"tier1_20ppm": 0, "tier2_50ppm": 0, "tier3_100ppm": 0, "tier4_c13": 0, "tier5_nearest": 0}

    t0 = time.time()
    for q in queries:
        obs_nm = neutral_mass(q["precursor_mz"], q["adduct"])
        if obs_nm is None or not np.isfinite(obs_nm) or obs_nm <= 0:
            obs_nm = q["precursor_mz"] - 1.007825
            
        c_idx, tier_w = retrieve_candidates_progressive(obs_nm, cand_masses, min_cands=25)
        pool_sizes.append(len(c_idx))
        
        tw_val = float(tier_w[0]) if len(tier_w) > 0 else 0.0
        if tw_val >= 1.0:
            tier_counts["tier1_20ppm"] += 1
        elif tw_val >= 0.85:
            tier_counts["tier2_50ppm"] += 1
        elif tw_val >= 0.70:
            tier_counts["tier3_100ppm"] += 1
        elif tw_val >= 0.50:
            tier_counts["tier4_c13"] += 1
        else:
            tier_counts["tier5_nearest"] += 1

        retrieved_set = set(cand_smiles[c_idx])
        is_hit = 1.0 if q["true_smiles"] in retrieved_set else 0.0
        recalls.append(is_hit)
        
        if "zero_reference" in q["mode"]:
            mode_a_recalls.append(is_hit)
            mode_a_pool_sizes.append(len(c_idx))
        else:
            mode_b_recalls.append(is_hit)
            mode_b_pool_sizes.append(len(c_idx))

    elapsed = time.time() - t0
    
    return {
        "catalog_name": catalog_name,
        "total_catalog_size": len(cand_df),
        "elapsed_sec": elapsed,
        "overall_recall": float(np.mean(recalls)),
        "overall_hits": int(sum(recalls)),
        "total_queries": len(queries),
        "mode_a_recall": float(np.mean(mode_a_recalls)),
        "mode_a_hits": int(sum(mode_a_recalls)),
        "mode_a_queries": len(mode_a_recalls),
        "mode_b_recall": float(np.mean(mode_b_recalls)),
        "mode_b_hits": int(sum(mode_b_recalls)),
        "mode_b_queries": len(mode_b_recalls),
        "pool_size_median": float(np.median(pool_sizes)),
        "pool_size_mean": float(np.mean(pool_sizes)),
        "pool_size_p95": float(np.percentile(pool_sizes, 95)),
        "pool_size_min": int(np.min(pool_sizes)),
        "pool_size_max": int(np.max(pool_sizes)),
        "tier_distribution": tier_counts,
    }


def main():
    print("=" * 85)
    print("  PHASE 3 EVALUATION GATE: CANDIDATE RECALL ON 500 CLEAN BENCHMARK QUERIES")
    print("=" * 85, flush=True)

    split_file = ROOT / "artifacts" / "v3_clean" / "clean_split.json"
    with open(split_file, "r", encoding="utf-8") as f:
        split = json.load(f)
    queries = split["benchmark_queries"]
    print(f"Loaded {len(queries)} clean benchmark queries (250 Mode A + 250 Mode B).\n")

    # 1. Evaluate Legacy Catalog
    legacy_path = ROOT / "kaggle_dataset" / "candidate_library.parquet"
    print(f"[1/2] Evaluating Baseline Legacy Catalog ({legacy_path})...", flush=True)
    df_legacy = pd.read_parquet(legacy_path)
    res_legacy = evaluate_catalog_recall(queries, df_legacy, "Baseline Legacy (276,940 cands)")

    # 2. Evaluate Expanded Catalog
    expanded_path = ROOT / "artifacts" / "v3_clean" / "candidate_union.parquet"
    print(f"\n[2/2] Evaluating Expanded Clean v3 Catalog ({expanded_path})...", flush=True)
    df_expanded = pd.read_parquet(expanded_path)
    res_expanded = evaluate_catalog_recall(queries, df_expanded, "Expanded Clean v3 (776,699 cands)")

    # Output Comparative Table
    print("\n" + "=" * 85)
    print("  PHASE 3 CANDIDATE RECALL COMPARATIVE RESULTS")
    print("=" * 85)
    print(f"{'Metric':<32} | {'Legacy Catalog':<22} | {'Expanded v3 Catalog':<22}")
    print("-" * 85)
    print(f"{'Catalog Size':<32} | {res_legacy['total_catalog_size']:>22,} | {res_expanded['total_catalog_size']:>22,}")
    print(f"{'New Structures Added':<32} | {'0 (train-only)':>22} | {res_expanded['total_catalog_size'] - res_legacy['total_catalog_size']:>22,}")
    print(f"{'Candidate Recall Overall':<32} | {res_legacy['overall_recall']*100:>21.2f}% | {res_expanded['overall_recall']*100:>21.2f}%")
    print(f"{'  -> Mode A (Zero-Reference)':<32} | {res_legacy['mode_a_recall']*100:>21.2f}% | {res_expanded['mode_a_recall']*100:>21.2f}%")
    print(f"{'  -> Mode B (Leave-One-Out)':<32} | {res_legacy['mode_b_recall']*100:>21.2f}% | {res_expanded['mode_b_recall']*100:>21.2f}%")
    print("-" * 85)
    print(f"{'Median Candidates / Query':<32} | {res_legacy['pool_size_median']:>22.0f} | {res_expanded['pool_size_median']:>22.0f}")
    print(f"{'Mean Candidates / Query':<32} | {res_legacy['pool_size_mean']:>22.1f} | {res_expanded['pool_size_mean']:>22.1f}")
    print(f"{'P95 Candidates / Query':<32} | {res_legacy['pool_size_p95']:>22.0f} | {res_expanded['pool_size_p95']:>22.0f}")
    min_max_leg = f"{res_legacy['pool_size_min']} / {res_legacy['pool_size_max']}"
    min_max_exp = f"{res_expanded['pool_size_min']} / {res_expanded['pool_size_max']}"
    print(f"{'Min / Max Candidates':<32} | {min_max_leg:>22} | {min_max_exp:>22}")
    print(f"{'Evaluation Latency (500 q)':<32} | {res_legacy['elapsed_sec']:>21.2f}s | {res_expanded['elapsed_sec']:>21.2f}s")
    print("=" * 85)

    # Save Results
    results_payload = {
        "benchmark_version": "v3_clean.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "legacy_catalog": res_legacy,
        "expanded_catalog": res_expanded,
    }

    out_json = ROOT / "artifacts" / "v3_clean" / "candidate_recall_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved benchmark recall results to: {out_json}")


if __name__ == "__main__":
    main()
