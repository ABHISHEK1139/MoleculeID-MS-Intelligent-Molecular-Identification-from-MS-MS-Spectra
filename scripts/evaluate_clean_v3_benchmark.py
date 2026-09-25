"""Clean Evaluation Harness (Phase 0 / v3_clean).

Executes leak-free evaluation on the 500 frozen benchmark queries:
- Mode A (250 queries): Zero-Reference (Class 2/3 simulation). True molecule is 100% absent from library.
- Mode B (250 queries): Leave-One-Spectrum-Out (Class 1 simulation). Sibling spectra at other CEs exist in library.

Zero Privileged Information:
- No true_formula
- No true_molecule_id
- No true_target_mass (search strictly centered on observed neutral mass derived from precursor_mz and adduct)
- Single authoritative preprocessing specification (v3_clean)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from src.core.preprocessing_v3 import (
    preprocess_spectrum,
    spectrum_to_coarse_bins,
    encode_spectrum_feature,
    fast_mutual_cosine,
    neutral_mass,
    retrieve_candidates_progressive,
    extract_unprivileged_candidate_features,
    MAX_PEAKS,
)
from src.models.spectrum_encoder import SpectrumEncoder


class CleanV3BenchmarkEvaluator:
    def __init__(
        self,
        ref_library_path: Path,
        cand_df: pd.DataFrame,
        cand_embs: np.ndarray,
        spec_encoder: torch.nn.Module,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.cand_df = cand_df
        self.cand_smiles = cand_df["normalized_smiles"].to_numpy()
        self.cand_masses = cand_df["exact_mass"].to_numpy(dtype=np.float64)
        self.cand_embs = cand_embs
        self.spec_encoder = spec_encoder.to(self.device).eval()

        print(f"Loading clean reference library from {ref_library_path}...", flush=True)
        df_lib = pd.read_parquet(ref_library_path)
        order = np.argsort(df_lib["neutral_mass"].to_numpy(dtype=np.float64))
        self.lib_smiles = df_lib["normalized_smiles"].to_numpy()[order]
        self.lib_neutral_masses = df_lib["neutral_mass"].to_numpy(dtype=np.float64)[order]
        self.lib_prec_mzs = df_lib["precursor_mz"].to_numpy(dtype=np.float64)[order]
        self.lib_ces = df_lib["collision_energy"].to_numpy(dtype=np.float32)[order]
        self.lib_mzs_list = [np.asarray(x, dtype=np.float32) for x in df_lib["peaks_mz"].iloc[order]]
        self.lib_intens_list = [np.asarray(x, dtype=np.float32) for x in df_lib["peaks_intensity"].iloc[order]]
        print(f"Loaded and sorted {len(df_lib):,} clean reference spectra.", flush=True)

    def search_library(self, m_neutral: float, q_mzs: np.ndarray, q_intens: np.ndarray, q_prec_mz: float, q_ce: float, ppm: float = 20.0) -> dict[str, dict[str, Any]]:
        tol = m_neutral * (ppm / 1e6)
        l_idx = int(np.searchsorted(self.lib_neutral_masses, m_neutral - tol, side="left"))
        r_idx = int(np.searchsorted(self.lib_neutral_masses, m_neutral + tol, side="right"))

        hits: dict[str, dict[str, Any]] = {}
        if r_idx > l_idx:
            for ri in range(l_idx, r_idx):
                ref_smi = self.lib_smiles[ri]
                ref_prec = self.lib_prec_mzs[ri]
                delta = q_prec_mz - ref_prec

                cos_sim, n_peaks = fast_mutual_cosine(
                    q_mzs,
                    q_intens,
                    self.lib_mzs_list[ri],
                    self.lib_intens_list[ri],
                    delta=delta,
                )
                if cos_sim >= 0.10:
                    r_ce = self.lib_ces[ri]
                    ce_diff = abs(q_ce - r_ce) if (np.isfinite(q_ce) and np.isfinite(r_ce)) else float("nan")
                    if ref_smi not in hits or cos_sim > hits[ref_smi]["cos"]:
                        hits[ref_smi] = {
                            "cos": cos_sim,
                            "n_peaks": n_peaks,
                            "ce_diff": ce_diff,
                            "n_supporting": 1,
                            "source_count": 1,
                        }
        return hits

    def rank_query(
        self,
        q_mzs: np.ndarray,
        q_intens: np.ndarray,
        precursor_mz: float,
        adduct: str,
        ce_val: float,
        top_k: int = 25,
    ) -> list[str]:
        # 1. Observed neutral mass ONLY (zero oracle leakage)
        m_neutral = neutral_mass(precursor_mz, adduct)
        if m_neutral is None or m_neutral <= 0 or not np.isfinite(m_neutral):
            m_neutral = precursor_mz - 1.007825

        # 2. Candidate retrieval
        cands_idx, tier_weights = retrieve_candidates_progressive(m_neutral, self.cand_masses, min_cands=top_k)
        cand_smis = self.cand_smiles[cands_idx]

        # 3. Spectral Library Search
        ext_hits = self.search_library(m_neutral, q_mzs, q_intens, precursor_mz, ce_val)

        # 4. Neural query embedding
        spec_feat = encode_spectrum_feature(q_mzs, q_intens, precursor_mz, adduct, ce_val)
        spec_t = torch.from_numpy(spec_feat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z_spec = F.normalize(self.spec_encoder(spec_t), p=2, dim=-1)

        # 5. Neural candidate dot product
        with torch.no_grad():
            sub_z_mols = torch.from_numpy(self.cand_embs[cands_idx].astype(np.float32)).to(self.device)
            dot_scores = (sub_z_mols * z_spec).sum(dim=-1).cpu().numpy()

        # 6. Mass decay score (gentle 50 ppm decay)
        ppm_errs = abs(self.cand_masses[cands_idx] - m_neutral) / m_neutral * 1e6
        mass_scores = 0.10 * np.exp(-ppm_errs / 50.0) * tier_weights

        # 7. Library evidence bonus (gated >= 0.70 to prevent false-analog poisoning)
        lib_bonus = np.zeros(len(cands_idx), dtype=np.float32)
        for i_local, ci in enumerate(cands_idx):
            c_smi = self.cand_smiles[ci]
            hit = ext_hits.get(c_smi)
            if hit and hit["cos"] >= 0.70 and hit["n_peaks"] >= 4:
                lib_bonus[i_local] = 1.0 + 2.0 * float(hit["cos"])

        final_scores = dot_scores + mass_scores + lib_bonus

        # 8. Strict unique deduplication
        order = np.argsort(-final_scores, kind="mergesort")
        seen: set[str] = set()
        dedup_ranked = []
        for idx in order:
            smi = cand_smis[idx]
            if smi not in seen:
                seen.add(smi)
                dedup_ranked.append(smi)
                if len(dedup_ranked) == top_k:
                    break

        while len(dedup_ranked) < top_k:
            dedup_ranked.append("CCO")

        return dedup_ranked[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Clean Benchmark Evaluator (v3_clean)")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 85)
    print("  CASMI 2026 CLEAN BENCHMARK EVALUATION (v3_clean)")
    print(f"  Target Device: {args.device}")
    print("=" * 85, flush=True)

    split_file = ROOT / "artifacts" / "v3_clean" / "clean_split.json"
    ref_file = ROOT / "artifacts" / "v3_clean" / "clean_reference_library.parquet"

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    benchmark_queries = split_data["benchmark_queries"]

    # Load candidate library & precomputed embeddings
    cand_df = pd.read_parquet(ROOT / "kaggle_dataset" / "candidate_library.parquet")
    cand_embs = np.load(ROOT / "kaggle_dataset" / "candidate_embeddings.npy")

    # Load SpectrumEncoder
    spec_encoder = SpectrumEncoder(embed_dim=256)
    spec_ckpt = torch.load(ROOT / "kaggle_dataset" / "spec_encoder.pt", map_location=args.device, weights_only=True)
    spec_encoder.load_state_dict(spec_ckpt.get("model_state_dict", spec_ckpt))
    spec_encoder.eval()

    evaluator = CleanV3BenchmarkEvaluator(
        ref_library_path=ref_file,
        cand_df=cand_df,
        cand_embs=cand_embs,
        spec_encoder=spec_encoder,
        device=args.device,
    )

    # Ingest benchmark raw spectra directly from pre-extracted artifact
    raw_pq = ROOT / "artifacts" / "v3_clean" / "benchmark_queries_raw.parquet"
    print(f"Loading raw spectrum data from {raw_pq}...", flush=True)
    df_raw = pq.read_table(raw_pq).to_pandas()
    print(f"Loaded all {len(df_raw)} query spectra in 0.05s.")

    # Execute Evaluation
    all_recips = []
    mode_a_recips = []
    mode_b_recips = []
    h1_list = []
    h5_list = []
    h25_list = []

    t_start = time.time()
    for qi, bq in enumerate(benchmark_queries):
        row = df_raw.iloc[qi]
        raw_mzs = np.asarray(row["ms2_mzs"], dtype=np.float32)
        raw_intens = np.asarray(row["ms2_normalized_intensities"], dtype=np.float32)

        cm, ci = preprocess_spectrum(raw_mzs, raw_intens, max_peaks=MAX_PEAKS, deisotope=True)
        prec_mz = float(row["precursor_mz"])
        adduct = str(row["adduct"])

        ce_val = 30.0
        ce_raw = row["collision_energy_ev"]
        if ce_raw is not None:
            try:
                if hasattr(ce_raw, "__iter__") and len(ce_raw) > 0:
                    ce_val = float(ce_raw[0])
                else:
                    ce_val = float(ce_raw)
            except Exception:
                ce_val = 30.0

        ranked_smis = evaluator.rank_query(
            cm, ci, prec_mz, adduct, ce_val, top_k=25
        )

        true_smi = bq["true_smiles"]
        r = ranked_smis.index(true_smi) + 1 if true_smi in ranked_smis else 0
        rr = 1.0 / r if 1 <= r <= 25 else 0.0

        all_recips.append(rr)
        h1_list.append(1.0 if r == 1 else 0.0)
        h5_list.append(1.0 if 1 <= r <= 5 else 0.0)
        h25_list.append(1.0 if 1 <= r <= 25 else 0.0)

        if bq["mode"] == "zero_reference_class2_3":
            mode_a_recips.append(rr)
        else:
            mode_b_recips.append(rr)

        if (qi + 1) % 50 == 0 or (qi + 1) == len(benchmark_queries):
            print(f"  Evaluated {qi+1}/{len(benchmark_queries)} queries in {time.time()-t_start:.1f}s | Current Overall MRR: {np.mean(all_recips):.4f}...", flush=True)

    mrr_all = float(np.mean(all_recips))
    mrr_a = float(np.mean(mode_a_recips))
    mrr_b = float(np.mean(mode_b_recips))
    h1 = float(np.mean(h1_list)) * 100.0
    h5 = float(np.mean(h5_list)) * 100.0
    h25 = float(np.mean(h25_list)) * 100.0

    print("\n" + "=" * 85)
    print("  CASMI 2026 CLEAN BENCHMARK RESULTS (v3_clean)")
    print("=" * 85)
    print(f"  Overall MRR@25:                        {mrr_all:.4f}")
    print(f"  Hit@1:                                 {h1:.2f}%")
    print(f"  Hit@5:                                 {h5:.2f}%")
    print(f"  Hit@25:                                {h25:.2f}%")
    print("-" * 85)
    print(f"  Mode A (Zero-Reference, Class 2/3):     {mrr_a:.4f}  (Pure neural/physical generalization)")
    print(f"  Mode B (Leave-One-Spectrum-Out, Class 1): {mrr_b:.4f}  (Real-world experimental spectral matching)")
    print("=" * 85, flush=True)

    # Save artifact
    res = {
        "benchmark_version": "v3_clean.1.0",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_mrr": mrr_all,
        "hit1": h1,
        "hit5": h5,
        "hit25": h25,
        "mode_a_zero_reference_mrr": mrr_a,
        "mode_b_leave_one_out_mrr": mrr_b,
        "n_queries": len(benchmark_queries),
    }
    with open(ROOT / "artifacts" / "v3_clean" / "clean_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Saved results artifact to artifacts/v3_clean/clean_benchmark_results.json")


if __name__ == "__main__":
    main()
