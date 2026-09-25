"""Clean Evaluation v1 — Uncontaminated, Leak-Free Benchmark Evaluation.

Strict Protocol:
1. C1 Fixed: Validation/benchmark molecules are strictly EXCLUDED from reference library lookups.
   No self-matches or validation sibling matches.
2. C2 Fixed: Exact production binning (MZ_BIN_MIN = 20.0, MZ_BIN_MAX = 1500.0, 1480 bins).
3. Label Leakage Removed: NO true_mol formula or oracle features in candidate vectors.
4. Single Source of Truth: Uses CanonicalInferencePipeline (src.inference.pipeline).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.core.adducts import neutral_mass
from src.core.canonical_benchmark import CanonicalBenchmark
from src.core.spectral_search import CompactSpectralLibrary
from src.models.reranker_v2 import CrossModalRerankerV2
from src.models.spectrum_encoder import SpectrumEncoder
from src.inference.pipeline import CanonicalInferencePipeline, clean_spectrum


def main():
    parser = argparse.ArgumentParser(description="Clean Evaluation v1 for CASMI 2026")
    parser.add_argument("--gamma", type=float, default=0.15, help="Fusion weight for Stage 5 v2")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    print("=" * 85)
    print("  CASMI 2026 CLEAN EVALUATION v1 (UNCONTAMINATED BENCHMARK)")
    print(f"  Target Device: {args.device} | Fusion Gamma: {args.gamma}")
    print("=" * 85)
    t_start = time.time()

    # 1. Load Candidate Universe
    print("\n[1/4] Loading Candidate Universe (276,940 candidates)...", flush=True)
    cand_path = ROOT / "kaggle_dataset" / "candidate_library.parquet"
    cand_df = pd.read_parquet(cand_path)
    cand_embs = np.load(ROOT / "kaggle_dataset" / "candidate_embeddings.npy")
    cand_fps = np.load(ROOT / "kaggle_dataset" / "candidate_fps.npy")
    print(f"Loaded {len(cand_df):,} candidates and precomputed embeddings.")

    # 2. Load Models
    print("\n[2/4] Loading SpectrumEncoder & CrossModalRerankerV2 Models...", flush=True)
    spec_encoder = SpectrumEncoder(embed_dim=256)
    spec_ckpt = torch.load(ROOT / "kaggle_dataset" / "spec_encoder.pt", map_location=args.device, weights_only=False)
    spec_encoder.load_state_dict(spec_ckpt.get("model_state_dict", spec_ckpt))
    spec_encoder.eval()

    reranker = CrossModalRerankerV2(embed_dim=256, evidence_dim=10, hidden_dim=256, dropout=0.10)
    reranker_ckpt = torch.load(ROOT / "kaggle_dataset" / "reranker_v2.pt", map_location=args.device, weights_only=False)
    reranker.load_state_dict(reranker_ckpt.get("model_state_dict", reranker_ckpt))
    reranker.eval()

    # 3. Load Unified Reference Library
    print("\n[3/4] Ingesting Unified Reference Library (1.38M Spectra)...", flush=True)
    ref_paths = [
        ROOT / "kaggle_dataset" / "unified_reference_library.parquet",
        ROOT / "artifacts" / "baseline" / "reference_library_multice.parquet",
    ]
    ref_file = next((p for p in ref_paths if p.exists()), None)
    if ref_file is None:
        raise FileNotFoundError("Could not find unified_reference_library.parquet or reference_library_multice.parquet")

    df_lib = pd.read_parquet(ref_file)
    smi_col = "canonical_smiles" if "canonical_smiles" in df_lib.columns else "normalized_smiles"
    mzs_col = "peaks_mz" if "peaks_mz" in df_lib.columns else "ms2_mzs"
    ints_col = "peaks_intensity" if "peaks_intensity" in df_lib.columns else "ms2_intensities"

    ref_library = CompactSpectralLibrary(
        smiles=df_lib[smi_col].to_numpy(),
        neutral_masses=df_lib["neutral_mass"].to_numpy(dtype=np.float64),
        precursor_mzs=df_lib["precursor_mz"].to_numpy(dtype=np.float64),
        collision_energies=df_lib["collision_energy"].to_numpy(dtype=np.float32),
        mzs_list=[np.asarray(x, dtype=np.float32) for x in df_lib[mzs_col]],
        intens_list=[np.asarray(x, dtype=np.float32) for x in df_lib[ints_col]],
        n_supporting=df_lib["n_supporting_spectra"].to_numpy(dtype=np.int16) if "n_supporting_spectra" in df_lib.columns else None,
        source_counts=df_lib["source_count"].to_numpy(dtype=np.int8) if "source_count" in df_lib.columns else None,
    )
    print(f"Loaded {len(df_lib):,} reference spectra.")

    # Instantiate Canonical Pipeline
    pipeline = CanonicalInferencePipeline(
        candidate_df=cand_df,
        candidate_embs=cand_embs,
        candidate_fps=cand_fps,
        ref_library=ref_library,
        spec_encoder=spec_encoder,
        reranker=reranker,
        device=args.device,
        gamma=args.gamma,
    )

    # 4. Load Canonical Benchmark
    print("\n[4/4] Ingesting Canonical Benchmark (200 Frozen Benchmark Queries)...", flush=True)
    bm = CanonicalBenchmark(subset_size=10000, n_benchmark_queries=200, split_seed=42, benchmark_seed=123)
    bench_q = bm.benchmark_queries

    # Strictly collect ALL validation molecules to completely isolate them from library evidence (C1)
    val_smiles: set[str] = set(bm.val_ds.mol_smiles.values())
    print(f"  Strict Isolation: {len(val_smiles)} validation molecules excluded from reference library lookups.")

    recips = []
    h1s = []
    h5s = []
    h25s = []
    novel_recips = []
    isomer_recips = []

    t_eval_start = time.time()
    for qi, q in enumerate(bench_q):
        true_smi = bm.cand_db.mol_smiles.get(q.true_mol, "")
        prec_mz = float(q.precursor_mz)
        adduct = str(q.adduct) if q.adduct is not None else "[M+H]+"
        m_neut = neutral_mass(prec_mz, adduct)
        if m_neut is None or m_neut <= 0:
            m_neut = prec_mz - 1.007825

        spec_info = bm.val_ds.samples[q.query_id][0]
        raw_mzs = np.asarray(spec_info["mz"], dtype=np.float32)
        raw_intens = np.asarray(spec_info["intensity"], dtype=np.float32)

        # Preprocessing parity: exact same cleaning as production
        clean_mzs, clean_intens = clean_spectrum(raw_mzs, raw_intens, max_peaks=60)

        q_entry = {
            "adduct": adduct,
            "precursor_mz": prec_mz,
            "m_neutral": m_neut,
            "mzs": clean_mzs,
            "intens": clean_intens,
            "ce": float(spec_info.get("ce", float("nan"))),
            "ion_mode": 1.0 if not adduct.endswith("-") else 0.0,
        }

        # Predict using Canonical Pipeline with strict exclusion of validation molecules
        ranked_smis = pipeline.rank_query(
            [q_entry],
            top_k=25,
            excluded_smiles=val_smiles,
        )

        r = ranked_smis.index(true_smi) + 1 if true_smi in ranked_smis else 0
        rr = 1.0 / r if 1 <= r <= 25 else 0.0

        recips.append(rr)
        h1s.append(1.0 if r == 1 else 0.0)
        h5s.append(1.0 if 1 <= r <= 5 else 0.0)
        h25s.append(1.0 if 1 <= r <= 25 else 0.0)

        if qi < 100:
            novel_recips.append(rr)
        if q.is_isomer_query:
            isomer_recips.append(rr)

        if (qi + 1) % 50 == 0 or qi == len(bench_q) - 1:
            print(f"  Evaluated {qi+1}/{len(bench_q)} queries in {time.time()-t_eval_start:.1f}s (Current clean MRR: {np.mean(recips):.4f})...", flush=True)

    # 5. Output Clean Report
    clean_mrr = float(np.mean(recips))
    clean_h1 = float(np.mean(h1s)) * 100.0
    clean_h5 = float(np.mean(h5s)) * 100.0
    clean_h25 = float(np.mean(h25s)) * 100.0
    clean_novel_mrr = float(np.mean(novel_recips)) if novel_recips else clean_mrr
    clean_iso_mrr = float(np.mean(isomer_recips)) if isomer_recips else clean_mrr

    print("\n" + "=" * 85)
    print("  CASMI 2026 CLEAN EVALUATION v1 — RESULTS REPORT")
    print("=" * 85)
    print(f"  Clean Overall MRR@25:   {clean_mrr:.4f}  (Leaked was: 0.9688)")
    print(f"  Clean Rank-1 (Hit@1):   {clean_h1:.2f}% (Leaked was: 94.75%)")
    print(f"  Clean Top-5 (Hit@5):    {clean_h5:.2f}% (Leaked was: 99.25%)")
    print(f"  Clean Top-25 (Hit@25):  {clean_h25:.2f}% (Leaked was: 100.00%)")
    print(f"  Clean Novel MRR:        {clean_novel_mrr:.4f}")
    print(f"  Clean Isomer MRR:       {clean_iso_mrr:.4f}")
    print("=" * 85)
    print(f"Total clean evaluation elapsed time: {time.time() - t_start:.1f}s")

    # Save results artifact
    report_dict = {
        "timestamp": time.time(),
        "clean_overall_mrr": clean_mrr,
        "clean_hit1": clean_h1,
        "clean_hit5": clean_h5,
        "clean_hit25": clean_h25,
        "clean_novel_mrr": clean_novel_mrr,
        "clean_isomer_mrr": clean_iso_mrr,
        "leaked_overall_mrr": 0.9688,
        "gamma": args.gamma,
    }
    out_path = ROOT / "artifacts" / "clean_evaluation_v1_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved clean report artifact to: {out_path}")


if __name__ == "__main__":
    main()
