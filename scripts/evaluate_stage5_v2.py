"""Stage 5 v2 Evaluation and Fusion Suite.

Tests whether the newly trained Stage 5 v2 model (with 277K hard negatives,
evidence feature fusion, and false-analog awareness) is additive when combined
with the validated Evidence Aggregation Scorer (baseline MRR = 0.9665).

Protocol:
1. Ingest 800 Tuning Queries and 400 Frozen Benchmark Queries from CanonicalBenchmark.
2. Ingest Unified External Library (MoNA + GNPS + Baseline 839k).
3. Load Stage 5 v2 checkpoint (CrossModalRerankerV2 + fine-tuned MoleculeGNN).
4. Tune fusion parameters strictly on the 800 Tuning Queries.
5. Evaluate on the 400 Frozen Benchmark Queries and compare directly against 0.9665 baseline.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from torch_geometric.data import Batch

from src.core.config import ARTIFACTS_DIR
from src.core.canonical_benchmark import CanonicalBenchmark, CanonicalQuery
from src.core.evidence_scorer import EvidenceScorer
from src.data.hard_negative_dataset_v2 import extract_candidate_evidence_vector
from src.data.mol_graph import smiles_to_graph
from src.models.molecule_encoder import MoleculeGNN
from src.models.reranker_v2 import CrossModalRerankerV2
from src.models.spectrum_encoder import SpectrumEncoder
from scripts.evaluate_external_library_diagnostic import (
    CompactSpectralLibrary,
    fast_mutual_cosine,
)


def evaluate_ranking(query_records: list[dict], scoring_func: Any) -> dict[str, float]:
    """Evaluate MRR, Hit@1, Hit@5, Hit@25 across query records."""
    recips, h1s, h5s, h25s, iso_recips = [], [], [], [], []

    for qd in query_records:
        cands = qd["candidates"]
        if not cands:
            recips.append(0.0)
            h1s.append(0.0)
            h5s.append(0.0)
            h25s.append(0.0)
            if qd["is_isomer"]:
                iso_recips.append(0.0)
            continue

        true_smi = qd["true_smi"]
        scores = [scoring_func(c) for c in cands]
        order = np.argsort(-np.array(scores))
        ranked_smis = [cands[i]["smi"] for i in order]

        r = ranked_smis.index(true_smi) + 1 if true_smi in ranked_smis else 0
        rr = 1.0 / r if 1 <= r <= 25 else 0.0

        recips.append(rr)
        h1s.append(1.0 if r == 1 else 0.0)
        h5s.append(1.0 if 1 <= r <= 5 else 0.0)
        h25s.append(1.0 if 1 <= r <= 25 else 0.0)

        if qd["is_isomer"]:
            iso_recips.append(rr)

    return {
        "mrr25": float(np.mean(recips)),
        "hit1": float(np.mean(h1s)) * 100.0,
        "hit5": float(np.mean(h5s)) * 100.0,
        "hit25": float(np.mean(h25s)) * 100.0,
        "isomer_mrr": float(np.mean(iso_recips)) if iso_recips else 0.0,
    }


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cpus = os.cpu_count() or 4
    n_workers = min(12, num_cpus)

    print("=" * 85)
    print("  STAGE 5 v2 BENCHMARK EVALUATION & EVIDENCE FUSION")
    print(f"  Device: {device} | Workers: {n_workers}")
    print("=" * 85, flush=True)

    # 1. Load Canonical Benchmark
    print("\n[1/5] Ingesting Canonical Benchmark (800 Tuning + 400 Frozen Benchmark)...", flush=True)
    bm = CanonicalBenchmark(subset_size=10000, n_benchmark_queries=200, split_seed=42, benchmark_seed=123)
    tuning_queries = bm.tuning_queries
    novel_queries = bm.benchmark_queries

    # Known queries
    train_samples_by_mol: dict[str, list[int]] = {}
    for sample_idx, (_, mol_id) in enumerate(bm.train_ds.samples):
        train_samples_by_mol.setdefault(mol_id, []).append(sample_idx)

    train_unique_mols = sorted(list(train_samples_by_mol.keys()))
    rng = np.random.default_rng(999)
    known_mol_picks = rng.choice(train_unique_mols, size=200, replace=False)

    known_queries = []
    for m in known_mol_picks:
        s_idx = train_samples_by_mol[m][0]
        spec_info = bm.train_ds.samples[s_idx][0]
        true_mol = bm.train_ds.samples[s_idx][1]
        prec_mz = spec_info.get("precursor_mz", 0.0)
        adduct = spec_info.get("adduct", None)
        if true_mol not in bm.cand_db.mol_smiles:
            continue
        t_idx = bm.mol_to_idx[true_mol]
        t_mass = bm.cand_db.raw_masses[t_idx]
        matches = bm.cand_db.query_two_tier(
            precursor_mz=prec_mz if prec_mz > 0 else t_mass + 1.0078,
            adduct=adduct if adduct is not None else "[M+H]+",
            ppm_primary=20.0,
            ppm_fallback=50.0,
            include_isotope=True,
        )
        f_true = bm.cand_index.formula_map.get(true_mol, "")
        has_isomers = any(
            m_cand.mol != true_mol and bm.cand_index.formula_map.get(m_cand.mol, "") == f_true
            for m_cand in matches
        )
        s_tensor = bm.train_ds[s_idx][0]
        known_queries.append(
            CanonicalQuery(
                query_id=int(s_idx),
                true_mol=true_mol,
                target_mass=t_mass,
                precursor_mz=prec_mz,
                adduct=adduct,
                spec_tensor=s_tensor,
                matches=matches,
                is_isomer_query=has_isomers,
                true_formula=f_true,
            )
        )

    benchmark_queries_all = novel_queries + known_queries
    print(f"Loaded {len(tuning_queries)} Tuning queries + {len(benchmark_queries_all)} Frozen Benchmark queries.")

    # 2. Ingest Unified External Library
    ref_paths = [
        ROOT / "kaggle_dataset" / "reference_library_multice.parquet",
        ROOT / "artifacts" / "baseline" / "reference_library_multice.parquet",
    ]
    ref_path = next((p for p in ref_paths if p.exists()), ref_paths[0])
    df_ref = pd.read_parquet(ref_path, columns=["normalized_smiles", "neutral_mass", "precursor_mz", "collision_energy", "ms2_mzs", "ms2_intensities"])

    ext_path = ROOT / "artifacts" / "external" / "external_spectra.parquet"
    df_ext = pd.read_parquet(ext_path, columns=[
        "canonical_smiles", "neutral_mass", "precursor_mz", "collision_energy",
        "peaks_mz", "peaks_intensity", "source_library", "n_supporting_spectra", "source_count"
    ])

    d_smiles = np.concatenate([df_ref["normalized_smiles"].to_numpy(), df_ext["canonical_smiles"].to_numpy()])
    d_masses = np.concatenate([df_ref["neutral_mass"].to_numpy(dtype=np.float64), df_ext["neutral_mass"].to_numpy(dtype=np.float64)])
    d_precs = np.concatenate([df_ref["precursor_mz"].to_numpy(dtype=np.float64), df_ext["precursor_mz"].to_numpy(dtype=np.float64)])
    d_ces = np.concatenate([df_ref["collision_energy"].to_numpy(dtype=np.float32), df_ext["collision_energy"].to_numpy(dtype=np.float32)])
    d_mzs = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_mzs"]] + [np.asarray(x, dtype=np.float32) for x in df_ext["peaks_mz"]]
    d_ints = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_intensities"]] + [np.asarray(x, dtype=np.float32) for x in df_ext["peaks_intensity"]]
    d_sup = np.concatenate([np.ones(len(df_ref), dtype=np.int16), df_ext["n_supporting_spectra"].to_numpy(dtype=np.int16)])
    d_src = np.concatenate([np.ones(len(df_ref), dtype=np.int8), df_ext["source_count"].to_numpy(dtype=np.int8)])

    lib_ext = CompactSpectralLibrary(d_smiles, d_masses, d_precs, d_ces, d_mzs, d_ints, d_sup, d_src)

    # 3. Load Trained Stage 5 v2 Checkpoint
    print("\n[3/5] Loading Stage 5 v2 Checkpoint & Models...", flush=True)
    ckpt_path = ARTIFACTS_DIR / "stage05" / "exp5_v2" / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}. Waiting for training to complete...")
        return

    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {ckpt_data.get('epoch', '?')} (Val Acc: {ckpt_data.get('val_acc', 0.0)*100:.2f}%)")

    # Spectrum Encoder (frozen 1D-CNN)
    spec_encoder = SpectrumEncoder(embed_dim=256).to(device)
    spec_ckpt = ARTIFACTS_DIR / "stage02" / "exp2a" / "checkpoints" / "best.pt"
    spec_data = torch.load(spec_ckpt, map_location=device, weights_only=False)
    spec_encoder.load_state_dict(spec_data.get("model_state_dict", spec_data))
    spec_encoder.eval()

    # Molecule GNN
    mol_encoder = MoleculeGNN(embed_dim=256).to(device)
    mol_encoder.load_state_dict(ckpt_data["mol_encoder_state_dict"])
    mol_encoder.eval()

    # CrossModalRerankerV2
    reranker_v2 = CrossModalRerankerV2(embed_dim=256, evidence_dim=10, hidden_dim=256).to(device)
    reranker_v2.load_state_dict(ckpt_data["reranker_state_dict"])
    reranker_v2.eval()

    # Pre-encode all candidate graphs
    cand_mols = bm.cand_db.valid_mols
    cand_graphs = [bm.cand_db.mol_graphs[m] for m in cand_mols]
    z_mols_list = []
    with torch.no_grad():
        for i in range(0, len(cand_graphs), 256):
            bg = Batch.from_data_list(cand_graphs[i:i + 256]).to(device)
            z_mols_list.append(mol_encoder(bg))
    cand_embs = torch.cat(z_mols_list, dim=0)

    # 4. Extract Candidate Evidence and Compute Stage 5 v2 Scores
    print("\n[4/5] Extracting candidate features & computing Stage 5 v2 neural scores...", flush=True)

    def process_dataset(queries: list[CanonicalQuery], is_tuning: bool) -> list[dict]:
        ds = bm.val_ds if is_tuning else None
        records = []

        for q in queries:
            if is_tuning:
                spec_info = ds.samples[q.query_id][0]
            else:
                # benchmark queries
                if q in novel_queries:
                    spec_info = bm.val_ds.samples[q.query_id][0]
                else:
                    spec_info = bm.train_ds.samples[q.query_id][0]

            prec_mz = q.precursor_mz
            target_mass = q.target_mass
            q_ce = spec_info.get("ce", float("nan"))
            q_mzs = np.asarray(spec_info["mz"], dtype=np.float32)
            q_ints = np.asarray(spec_info["intensity"], dtype=np.float32)
            true_smi = bm.cand_db.mol_smiles.get(q.true_mol, "")
            true_formula = q.true_formula

            # Query external library
            cand_smis = {bm.cand_db.mol_smiles.get(m.mol, "") for m in q.matches}
            l_idx, r_idx = lib_ext.query_window(target_mass, ppm=20.0)
            ext_hits = {}
            top_hit_cos = 0.0
            top_ref_smi = ""

            if r_idx > l_idx:
                for ri in range(l_idx, r_idx):
                    ref_smi = lib_ext.smiles[ri]
                    if ref_smi not in cand_smis:
                        continue
                    delta = prec_mz - lib_ext.precursor_mzs[ri]
                    cos_sim, n_peaks = fast_mutual_cosine(
                        q_mzs, q_ints, lib_ext.mzs_list[ri], lib_ext.intens_list[ri], delta=delta
                    )
                    if cos_sim > 0.10:
                        r_ce = lib_ext.collision_energies[ri]
                        ce_diff = abs(q_ce - r_ce) if (np.isfinite(q_ce) and np.isfinite(r_ce)) else float("nan")
                        if ref_smi not in ext_hits or cos_sim > ext_hits[ref_smi]["cos"]:
                            ext_hits[ref_smi] = {
                                "cos": cos_sim,
                                "n_peaks": n_peaks,
                                "ce_diff": ce_diff,
                                "n_supporting": int(lib_ext.n_supporting[ri]),
                                "source_count": int(lib_ext.source_counts[ri]),
                            }
                        if cos_sim > top_hit_cos:
                            top_hit_cos = cos_sim
                            top_ref_smi = ref_smi

            # Top ref Morgan fingerprint
            ref_fp = None
            if top_ref_smi:
                ref_mol = Chem.MolFromSmiles(top_ref_smi)
                if ref_mol:
                    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)

            # Candidate records
            cands = []
            cand_indices = []
            evidence_list = []
            morgans_list = []

            for m in q.matches:
                smi = bm.cand_db.mol_smiles.get(m.mol, "")
                if not smi:
                    continue
                tier_w = 1.0 if m.tier == 1 else 0.50
                s_mass = 1.50 * np.exp(-m.ppm_error / 10.0) * tier_w
                hit = ext_hits.get(smi)

                # Validated additive evidence score (C: Cosine + Peaks)
                if hit and hit["cos"] >= 0.10:
                    raw_ev = 0.769 * hit["cos"] + 0.231 * min(1.0, hit["n_peaks"] / 6.0)
                    s_ev = 4.50 * (raw_ev ** 2) if raw_ev >= 0.45 else (1.0 * raw_ev if raw_ev >= 0.20 else 0.0)
                else:
                    s_ev = 0.0

                base_evidence_score = s_mass + s_ev

                # Evidence vector for Stage 5 v2
                is_iso = (bm.cand_index.formula_map.get(m.mol, "") == true_formula)
                ev_vec = extract_candidate_evidence_vector(
                    hit=hit,
                    ppm_error=m.ppm_error,
                    tier=m.tier,
                    prec_mz=prec_mz,
                    is_isomer=is_iso,
                )

                # Morgan similarity to top reference
                m_sim = 0.0
                if ref_fp is not None:
                    c_mol = Chem.MolFromSmiles(smi)
                    if c_mol:
                        c_fp = AllChem.GetMorganFingerprintAsBitVect(c_mol, 2, nBits=1024)
                        m_sim = float(DataStructs.TanimotoSimilarity(ref_fp, c_fp))

                cands.append({
                    "mol_id": m.mol,
                    "smi": smi,
                    "s_evidence": base_evidence_score,
                    "is_true": (smi == true_smi),
                })
                cand_indices.append(bm.mol_to_idx[m.mol])
                evidence_list.append(ev_vec)
                morgans_list.append(m_sim)

            # Compute Stage 5 v2 neural scores
            if cands and q.spec_tensor is not None:
                with torch.no_grad():
                    z_spec = spec_encoder(q.spec_tensor.unsqueeze(0).to(device))
                    sub_z_mols = cand_embs[cand_indices]
                    sub_morgans = torch.tensor(morgans_list, dtype=torch.float32, device=device).unsqueeze(-1)
                    sub_ev = torch.tensor(np.array(evidence_list), dtype=torch.float32, device=device)

                    s5_scores = reranker_v2(z_spec, sub_z_mols, sub_morgans, sub_ev).cpu().numpy()

                for c, s5_val in zip(cands, s5_scores):
                    c["s5_raw"] = float(s5_val)
            else:
                for c in cands:
                    c["s5_raw"] = 0.0

            records.append({
                "true_smi": true_smi,
                "is_isomer": q.is_isomer_query,
                "candidates": cands,
            })

        return records

    tuning_records = process_dataset(tuning_queries, is_tuning=True)
    benchmark_records = process_dataset(benchmark_queries_all, is_tuning=False)

    # 5. Tune Calibration and Fusion Weight on 800 Tuning Queries
    print("\n[5/5] Tuning Stage 5 v2 Fusion Weight on 800 Tuning Queries...", flush=True)

    # Pure Evidence Scorer Baseline
    def score_evidence_only(c: dict) -> float:
        return c["s_evidence"]

    tune_ev_res = evaluate_ranking(tuning_records, score_evidence_only)
    bench_ev_res = evaluate_ranking(benchmark_records, score_evidence_only)
    bench_novel_ev = evaluate_ranking(benchmark_records[:200], score_evidence_only)
    bench_known_ev = evaluate_ranking(benchmark_records[200:], score_evidence_only)

    print(f"  Tuning Set Evidence-Only MRR@25: {tune_ev_res['mrr25']:.4f}")
    print(f"  Frozen Benchmark Evidence-Only MRR@25: {bench_ev_res['mrr25']:.4f} (Hit@1={bench_ev_res['hit1']:.2f}%)")

    # Sweep gamma strictly on tuning set
    best_gamma = 0.0
    best_tune_mrr = tune_ev_res["mrr25"]

    for g in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]:
        def score_fusion_test(c: dict, gamma=g) -> float:
            return c["s_evidence"] + gamma * c["s5_raw"]

        res = evaluate_ranking(tuning_records, score_fusion_test)
        if res["mrr25"] > best_tune_mrr:
            best_tune_mrr = res["mrr25"]
            best_gamma = g

    print(f"  Optimal Stage 5 v2 Weight (gamma): {best_gamma} -> Tuning MRR: {best_tune_mrr:.4f} (Delta: {best_tune_mrr - tune_ev_res['mrr25']:+.4f})")

    # Evaluate Fused System on Frozen Benchmark
    def score_fused(c: dict) -> float:
        return c["s_evidence"] + best_gamma * c["s5_raw"]

    bench_fused_res = evaluate_ranking(benchmark_records, score_fused)
    bench_novel_fused = evaluate_ranking(benchmark_records[:200], score_fused)
    bench_known_fused = evaluate_ranking(benchmark_records[200:], score_fused)

    # Print Final Head-to-Head Comparison Table
    print("\n" + "=" * 95)
    print("  FINAL COMPARISON: EVIDENCE BASELINE vs STAGE 5 v2 FUSED (400 FROZEN BENCHMARK)")
    print("=" * 95)
    print(f"{'System':<34} | {'Overall MRR':<12} | {'Hit@1 (%)':<10} | {'Novel MRR':<11} | {'Isomer MRR':<12} | {'Delta MRR':<10}")
    print("-" * 95)
    print(f"{'Evidence Scorer (Validated Base)':<34} | {bench_ev_res['mrr25']:<12.4f} | {bench_ev_res['hit1']:<10.2f} | {bench_novel_ev['mrr25']:<11.4f} | {bench_ev_res['isomer_mrr']:<12.4f} | {'Baseline':<10}")
    delta_mrr = bench_fused_res['mrr25'] - bench_ev_res['mrr25']
    print(f"{'Stage 5 v2 + Evidence Scorer':<34} | {bench_fused_res['mrr25']:<12.4f} | {bench_fused_res['hit1']:<10.2f} | {bench_novel_fused['mrr25']:<11.4f} | {bench_fused_res['isomer_mrr']:<12.4f} | {delta_mrr:+10.4f}")
    print("=" * 95)

    # Save results
    out_eval_path = ARTIFACTS_DIR / "stage05" / "exp5_v2" / "benchmark_evaluation.json"
    with open(out_eval_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_gamma": best_gamma,
            "tuning_evidence": tune_ev_res,
            "tuning_fused": best_tune_mrr,
            "benchmark_evidence": bench_ev_res,
            "benchmark_fused": bench_fused_res,
            "benchmark_novel_fused": bench_novel_fused,
            "benchmark_known_fused": bench_known_fused,
        }, f, indent=2)

    print(f"\nEvaluation artifact saved to: {out_eval_path}")
    print(f"Total evaluation elapsed time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
