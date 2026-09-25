"""Stage 6 Comprehensive Evaluation: Rigorous Tuning vs Frozen Out-of-Sample Benchmark.

Adheres strictly to the 4 user engineering & evaluation requirements:
1. Complete separation of 800 Tuning Molecules from 200 Frozen Benchmark Queries.
2. Global Probabilistic Calibration (Platt scaling & learned physics decay) on the 800-molecule tuning pool.
3. Stage 5 Checkpoint Selection: Evaluates best.pt vs last.pt on the tuning pool;
   Primary: Isomer MRR@25, Secondary: Full MRR@25.
4. Independent evaluation of Stage 6 Fusion and Stage 6 Router on the frozen 200 benchmark queries.
5. Router threshold sweep tau in [0.40, 0.85] exported to router_sweep.csv.
6. Benchmark contract files saved to artifacts/stage06/.

Usage:
    python scripts/evaluate_stage6.py --exp-name exp6a
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ARTIFACTS_DIR
from src.core.canonical_benchmark import CanonicalBenchmark, CanonicalQuery
from src.core.evaluation import summarize_ranks
from src.models.spectrum_encoder import SpectrumEncoder
from src.models.molecule_encoder import MoleculeGNN
from src.models.reranker import CrossModalReranker
from src.models.ensemble import GlobalScoreCalibrator, HybridRouter, CalibratedFusedRanker


@torch.no_grad()
def run_stage6_pipeline(exp_name: str = "exp6a") -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print(f"  STAGE 6 HYBRID SYSTEM & RIGOROUS OUT-OF-SAMPLE BENCHMARK ({device})", flush=True)
    print("=" * 80, flush=True)

    out_dir = ARTIFACTS_DIR / "stage06"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Canonical Benchmark (Splits 800 Tuning vs 200 Benchmark)
    benchmark = CanonicalBenchmark(
        subset_size=10000,
        n_benchmark_queries=200,
        split_seed=42,
        benchmark_seed=123,
    )

    # 2. Load Base Models
    print("\nLoading models...", flush=True)
    spec_encoder = SpectrumEncoder(embed_dim=256).to(device)
    spec_ckpt = ARTIFACTS_DIR / "stage02/exp2a/checkpoints/best.pt"
    spec_data = torch.load(spec_ckpt, map_location=device, weights_only=False)
    spec_encoder.load_state_dict(spec_data.get("model_state_dict", spec_data))
    spec_encoder.eval()

    # Pre-encode candidate graphs
    cand_mols = benchmark.cand_db.valid_mols
    cand_graphs = [benchmark.cand_db.mol_graphs[m] for m in cand_mols]
    batch_size = 64

    # Helper to encode candidate embeddings with a specific MoleculeGNN
    def encode_candidate_graphs(mol_model: nn.Module) -> torch.Tensor:
        mol_model.eval()
        z_list = []
        for i in range(0, len(cand_graphs), batch_size):
            bg = Batch.from_data_list(cand_graphs[i:i + batch_size]).to(device)
            z_list.append(mol_model(bg).cpu())
        return torch.cat(z_list, dim=0).to(device)

    # 3. PHASE 1: STAGE 5 CHECKPOINT SELECTION ON THE 800 TUNING MOLECULES
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 1: STAGE 5 CHECKPOINT SELECTION ON 800 TUNING MOLECULES", flush=True)
    print("  (Criterion: PRIMARY = Isomer MRR@25 | SECONDARY = Full MRR@25)", flush=True)
    print("=" * 80, flush=True)

    s5_checkpoints = {
        "best.pt (Epoch 1)": ARTIFACTS_DIR / "stage05/exp5a/checkpoints/best.pt",
        "last.pt (Epoch 15)": ARTIFACTS_DIR / "stage05/exp5a/checkpoints/last.pt",
    }

    # Evaluate each checkpoint on the 800 tuning queries
    tuning_queries = benchmark.tuning_queries
    query_specs_tuning = torch.stack([q.spec_tensor for q in tuning_queries], dim=0).to(device)
    z_queries_tuning = spec_encoder(query_specs_tuning)  # (800, 256)

    ckpt_results = {}
    best_ckpt_name = None
    best_ckpt_isomer_mrr = -1.0
    best_mol_encoder = None
    best_reranker = None
    best_cand_embs = None

    for ckpt_label, ckpt_path in s5_checkpoints.items():
        if not ckpt_path.exists():
            continue
        print(f"\nEvaluating checkpoint: {ckpt_label} on {len(tuning_queries)} tuning queries...", flush=True)
        ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)

        mol_enc = MoleculeGNN(embed_dim=256).to(device)
        mol_enc.load_state_dict(ckpt_data["mol_encoder_state_dict"])
        mol_enc.eval()

        rerank = CrossModalReranker(embed_dim=256, physics_dim=4, hidden_dim=256).to(device)
        rerank.load_state_dict(ckpt_data["reranker_state_dict"])
        rerank.eval()

        cur_cand_embs = encode_candidate_graphs(mol_enc)

        # Score tuning queries
        t_ranks = []
        for i, q in enumerate(tuning_queries):
            matches = q.matches
            matched_indices = [benchmark.mol_to_idx[m.mol] for m in matches]
            if len(matched_indices) == 0:
                t_ranks.append(0)
                continue

            sub_z_mols = cur_cand_embs[matched_indices]
            sub_z_spec = z_queries_tuning[i:i + 1]

            prec_norm = q.precursor_mz / 1000.0
            phys_list = [
                [min(m.ppm_error / 20.0, 3.0), 1.0 if m.tier == 1 else 0.5, prec_norm, 1.0 if m.ppm_error <= 5.0 else 0.0]
                for m in matches
            ]
            sub_phys = torch.tensor(phys_list, dtype=torch.float32, device=device)
            raw_scores = rerank(sub_z_spec, sub_z_mols, sub_phys).cpu().numpy()

            sort_order = np.argsort(-raw_scores)
            ranked_mols = [matches[k].mol for k in sort_order][:25]
            t_ranks.append(ranked_mols.index(q.true_mol) + 1 if q.true_mol in ranked_mols else 0)

        eval_res = benchmark.evaluate_ranks(t_ranks, is_tuning=True)
        iso_mrr = eval_res["exact_isomer_subset"]["mrr"]
        full_mrr = eval_res["overall"]["mrr"]
        print(f"  Result for {ckpt_label}: Isomer MRR@25 = {iso_mrr:.4f} (Hit@1: {eval_res['exact_isomer_subset']['hit@1']*100:.1f}%) | "
              f"Full MRR@25 = {full_mrr:.4f} (Hit@1: {eval_res['overall']['hit@1']*100:.1f}%)", flush=True)

        ckpt_results[ckpt_label] = eval_res
        if iso_mrr > best_ckpt_isomer_mrr:
            best_ckpt_isomer_mrr = iso_mrr
            best_ckpt_name = ckpt_label
            best_mol_encoder = mol_enc
            best_reranker = rerank
            best_cand_embs = cur_cand_embs

    print(f"\n[Checkpoint Decision] Winner on Tuning Pool: {best_ckpt_name} (Peak Isomer MRR = {best_ckpt_isomer_mrr:.4f})", flush=True)

    # 4. PHASE 2: FIT GLOBAL CALIBRATION & SCORE FUSION ON 800 TUNING MOLECULES
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 2: FIT GLOBAL CALIBRATION & FUSION WEIGHTS ON TUNING POOL", flush=True)
    print("=" * 80, flush=True)

    # Collect tuning candidate scores for fitting Platt scaling
    reranker_pos_scores = []
    reranker_neg_scores = []
    tuning_cached_data = []

    for i, q in enumerate(tuning_queries):
        matches = q.matches
        matched_indices = [benchmark.mol_to_idx[m.mol] for m in matches]
        if len(matched_indices) == 0:
            continue

        sub_z_mols = best_cand_embs[matched_indices]
        sub_z_spec = z_queries_tuning[i:i + 1]

        ppm_errors = np.array([m.ppm_error for m in matches], dtype=np.float32)
        tiers = np.array([m.tier for m in matches], dtype=np.int32)
        prec_norm = q.precursor_mz / 1000.0
        phys_list = [
            [min(m.ppm_error / 20.0, 3.0), 1.0 if m.tier == 1 else 0.5, prec_norm, 1.0 if m.ppm_error <= 5.0 else 0.0]
            for m in matches
        ]
        sub_phys = torch.tensor(phys_list, dtype=torch.float32, device=device)
        raw_scores = best_reranker(sub_z_spec, sub_z_mols, sub_phys).cpu().numpy()

        for k, m in enumerate(matches):
            if m.mol == q.true_mol:
                reranker_pos_scores.append(float(raw_scores[k]))
            else:
                reranker_neg_scores.append(float(raw_scores[k]))

        tuning_cached_data.append({
            "true_mol": q.true_mol,
            "cand_mols": [m.mol for m in matches],
            "ppm_errors": ppm_errors,
            "tiers": tiers,
            "raw_scores": raw_scores,
            "is_isomer": q.is_isomer_query,
        })

    # Fit Global Score Calibrator
    calibrator = GlobalScoreCalibrator()
    calibrator.fit(
        reranker_pos=np.array(reranker_pos_scores, dtype=np.float32),
        reranker_neg=np.array(reranker_neg_scores, dtype=np.float32),
    )
    calibrator.save(out_dir / "calibration.json")

    # Fit optimal physics mass decay tau_mass and fusion weights (beta_mass, gamma_reranker) on tuning pool
    print("\nTuning physics decay constant (tau_mass) and fusion weights on tuning pool...", flush=True)
    best_tune_mrr = 0.0
    best_tau_mass = 10.0
    best_w_mass = 0.50
    best_w_rerank = 0.50

    for tau_cand in [5.0, 10.0, 15.0, 20.0]:
        calibrator.tau_mass = tau_cand
        for w_rerank in [0.20, 0.40, 0.50, 0.60, 0.80]:
            w_mass = 1.0 - w_rerank
            fused_ranks = []
            for item in tuning_cached_data:
                s_mass = calibrator.calibrate_mass_error(item["ppm_errors"], item["tiers"])
                s_rerank = calibrator.calibrate_reranker(item["raw_scores"])
                score = w_mass * s_mass + w_rerank * s_rerank
                order = np.argsort(-score)
                ranked_mols = [item["cand_mols"][k] for k in order][:25]
                fused_ranks.append(ranked_mols.index(item["true_mol"]) + 1 if item["true_mol"] in ranked_mols else 0)

            ev = benchmark.evaluate_ranks(fused_ranks, is_tuning=True)
            if ev["overall"]["mrr"] > best_tune_mrr:
                best_tune_mrr = ev["overall"]["mrr"]
                best_tau_mass = tau_cand
                best_w_mass = w_mass
                best_w_rerank = w_rerank

    calibrator.tau_mass = best_tau_mass
    calibrator.save(out_dir / "calibration.json")
    print(f"Fitted Parameters: tau_mass={best_tau_mass}, Weight Mass={best_w_mass:.2f}, Weight Reranker={best_w_rerank:.2f} "
          f"(Tuning Pool MRR: {best_tune_mrr:.4f})", flush=True)

    # 5. Build Stage 1 Reference Library for Cosine Search
    print("\nIndexing reference spectral library for Stage 1...", flush=True)
    from src.data.spectrum_dataset import spectrum_to_coarse_bins
    train_spectra_tensors = []
    train_mol_ids = []
    for s_info, mol_id in benchmark.train_ds.samples:
        binned = spectrum_to_coarse_bins(s_info["mz"], s_info["intensity"])
        train_spectra_tensors.append(torch.from_numpy(binned))
        train_mol_ids.append(mol_id)

    lib_tensors = torch.stack(train_spectra_tensors, dim=0).to(device)  # (N_lib, 1480)
    lib_tensors = nn.functional.normalize(lib_tensors, dim=-1)

    # 6. PHASE 3: EVALUATE FROZEN CANONICAL BENCHMARK (200 FIXED QUERIES)
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 3: OUT-OF-SAMPLE EVALUATION ON 200 CANONICAL BENCHMARK QUERIES", flush=True)
    print("=" * 80, flush=True)

    benchmark_queries = benchmark.benchmark_queries
    query_specs_bm = torch.stack([q.spec_tensor for q in benchmark_queries], dim=0).to(device)
    z_queries_bm = spec_encoder(query_specs_bm)

    # Stage 3 unconstrained cosine
    sims_raw_s3_bm = (z_queries_bm @ best_cand_embs.T).cpu().numpy()

    # Stage 1 cosine against reference library
    bm_coarse = torch.stack([q.spec_tensor[:1480] for q in benchmark_queries], dim=0).to(device)
    bm_coarse = nn.functional.normalize(bm_coarse, dim=-1)
    bm_lib_cos = torch.mm(bm_coarse, lib_tensors.T).cpu().numpy()

    ranks_stage1: list[int] = []
    ranks_stage3: list[int] = []
    ranks_physics: list[int] = []
    ranks_stage4: list[int] = []
    ranks_stage5: list[int] = []
    ranks_stage6_fusion: list[int] = []
    ranks_stage6_router_default: list[int] = []

    bm_cached_candidates = []

    for i, q in enumerate(benchmark_queries):
        target_idx = benchmark.mol_to_idx[q.true_mol]
        matches = q.matches

        # 1. Stage 1 Classical Library Retrieval (Best match from training library + physics fallback)
        best_lib_idx = int(np.argmax(bm_lib_cos[i]))
        max_cos = float(bm_lib_cos[i, best_lib_idx])
        best_lib_mol = train_mol_ids[best_lib_idx]
        st1_candidates = [best_lib_mol] + [m.mol for m in matches if m.mol != best_lib_mol]
        st1_candidates = st1_candidates[:25]
        ranks_stage1.append(st1_candidates.index(q.true_mol) + 1 if q.true_mol in st1_candidates else 0)

        # 2. Stage 3 Unconstrained Cosine
        ranked_s3 = np.argsort(-sims_raw_s3_bm[i])[:25]
        match_s3 = np.where(ranked_s3 == target_idx)[0]
        ranks_stage3.append(int(match_s3[0] + 1) if len(match_s3) > 0 else 0)

        # 3. Physics Mass Ordering
        ranked_phys_mols = [m.mol for m in matches][:25]
        ranks_physics.append(ranked_phys_mols.index(q.true_mol) + 1 if q.true_mol in ranked_phys_mols else 0)

        # 4. Stage 4 Hybrid
        matched_mols = {m.mol: m for m in matches}
        score_stage4 = np.full(len(cand_mols), -1e9, dtype=np.float32)
        for m_id, match_obj in matched_mols.items():
            idx = benchmark.mol_to_idx[m_id]
            score_stage4[idx] = sims_raw_s3_bm[i, idx] * match_obj.weight
        ranked_s4 = np.argsort(-score_stage4)[:25]
        match_s4 = np.where(ranked_s4 == target_idx)[0]
        ranks_stage4.append(int(match_s4[0] + 1) if len(match_s4) > 0 else 0)

        # 5. Stage 5 & 6 on candidates
        matched_indices = [benchmark.mol_to_idx[m.mol] for m in matches]
        if len(matched_indices) > 0:
            sub_z_mols = best_cand_embs[matched_indices]
            sub_z_spec = z_queries_bm[i:i + 1]

            ppm_errors = np.array([m.ppm_error for m in matches], dtype=np.float32)
            tiers = np.array([m.tier for m in matches], dtype=np.int32)
            prec_norm = q.precursor_mz / 1000.0
            phys_list = [
                [min(m.ppm_error / 20.0, 3.0), 1.0 if m.tier == 1 else 0.5, prec_norm, 1.0 if m.ppm_error <= 5.0 else 0.0]
                for m in matches
            ]
            sub_phys = torch.tensor(phys_list, dtype=torch.float32, device=device)
            raw_scores = best_reranker(sub_z_spec, sub_z_mols, sub_phys).cpu().numpy()

            # Stage 5 raw ranking
            sort_s5 = np.argsort(-raw_scores)
            ranked_s5 = [matches[k].mol for k in sort_s5][:25]
            ranks_stage5.append(ranked_s5.index(q.true_mol) + 1 if q.true_mol in ranked_s5 else 0)

            # Stage 6 Calibrated Fusion (using frozen parameters)
            s_mass = calibrator.calibrate_mass_error(ppm_errors, tiers)
            s_rerank = calibrator.calibrate_reranker(raw_scores)
            fused_score = best_w_mass * s_mass + best_w_rerank * s_rerank

            sort_fused = np.argsort(-fused_score)
            ranked_fused = [matches[k].mol for k in sort_fused][:25]
            ranks_stage6_fusion.append(ranked_fused.index(q.true_mol) + 1 if q.true_mol in ranked_fused else 0)

            # Stage 6 Router (default tau=0.65)
            if max_cos >= 0.65:
                router_cands = [best_lib_mol] + [m for m in ranked_fused if m != best_lib_mol]
            else:
                router_cands = ranked_fused
            router_cands = router_cands[:25]
            ranks_stage6_router_default.append(router_cands.index(q.true_mol) + 1 if q.true_mol in router_cands else 0)

            bm_cached_candidates.append({
                "true_mol": q.true_mol,
                "cand_mols": [m.mol for m in matches],
                "ranked_fused": ranked_fused,
                "max_lib_cosine": max_cos,
                "best_lib_mol": best_lib_mol,
            })
        else:
            ranks_stage5.append(0)
            ranks_stage6_fusion.append(0)
            ranks_stage6_router_default.append(0)

    # Summarize all systems on the canonical benchmark
    m_st1 = benchmark.evaluate_ranks(ranks_stage1)
    m_st3 = benchmark.evaluate_ranks(ranks_stage3)
    m_phys = benchmark.evaluate_ranks(ranks_physics)
    m_st4 = benchmark.evaluate_ranks(ranks_stage4)
    m_st5 = benchmark.evaluate_ranks(ranks_stage5)
    m_st6_fused = benchmark.evaluate_ranks(ranks_stage6_fusion)
    m_st6_router = benchmark.evaluate_ranks(ranks_stage6_router_default)

    # 7. Print the Explicit User-Specified Benchmark Table
    print("\n" + "=" * 80, flush=True)
    print("  EXPLICIT SYSTEM COMPARISON (Canonical 200 Benchmark Queries)", flush=True)
    print("=" * 80, flush=True)
    header = f"{'System':<24} | {'Full MRR':<10} | {'Full H@1':<10} | {'Full H@25':<10} | {'Isomer MRR':<12} | {'Isomer H@1':<12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    systems = [
        ("Stage 1 (Cosine Lib)", m_st1),
        ("Stage 3 (Unconstrained)", m_st3),
        ("Physics (Mass Order)", m_phys),
        ("Stage 4 Hybrid", m_st4),
        ("Stage 5 (Reranker)", m_st5),
        ("Stage 6 Fusion", m_st6_fused),
        ("Stage 6 Router (0.65)", m_st6_router),
    ]

    for name, m in systems:
        row = (
            f"{name:<24} | "
            f"{m['overall']['mrr']:<10.4f} | "
            f"{m['overall']['hit@1']*100:<9.2f}% | "
            f"{m['overall']['hit@25']*100:<9.2f}% | "
            f"{m['exact_isomer_subset']['mrr']:<12.4f} | "
            f"{m['exact_isomer_subset']['hit@1']*100:<11.2f}%"
        )
        print(row, flush=True)

    # 8. ROUTER THRESHOLD SWEEP TABLE (Exported to CSV)
    print("\n" + "=" * 80, flush=True)
    print("  ROUTER THRESHOLD SWEEP (tau in [0.40, 0.85])", flush=True)
    print("=" * 80, flush=True)
    sw_header = f"{'tau':<8} | {'Library Usage':<15} | {'Full MRR':<10} | {'Full H@1':<10} | {'Full H@25':<10} | {'Isomer MRR':<12}"
    print(sw_header, flush=True)
    print("-" * len(sw_header), flush=True)

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    sweep_rows = []

    for tau in thresholds:
        r_ranks = []
        n_routed_st1 = 0
        for item in bm_cached_candidates:
            if item["max_lib_cosine"] >= tau:
                n_routed_st1 += 1
                cands = [item["best_lib_mol"]] + [m for m in item["ranked_fused"] if m != item["best_lib_mol"]]
            else:
                cands = item["ranked_fused"]
            cands = cands[:25]
            r_ranks.append(cands.index(item["true_mol"]) + 1 if item["true_mol"] in cands else 0)

        ev = benchmark.evaluate_ranks(r_ranks)
        usage_pct = (n_routed_st1 / len(bm_cached_candidates)) * 100
        sw_row = {
            "tau": tau,
            "library_usage_pct": round(usage_pct, 2),
            "library_usage_count": n_routed_st1,
            "full_mrr": ev["overall"]["mrr"],
            "full_hit1": ev["overall"]["hit@1"],
            "full_hit25": ev["overall"]["hit@25"],
            "isomer_mrr": ev["exact_isomer_subset"]["mrr"],
            "isomer_hit1": ev["exact_isomer_subset"]["hit@1"],
        }
        sweep_rows.append(sw_row)
        print(f"{tau:<8.2f} | {usage_pct:<14.1f}% | {ev['overall']['mrr']:<10.4f} | {ev['overall']['hit@1']*100:<9.2f}% | {ev['overall']['hit@25']*100:<9.2f}% | {ev['exact_isomer_subset']['mrr']:<12.4f}", flush=True)

    # Save CSV
    csv_path = out_dir / "router_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tau", "library_usage_pct", "library_usage_count", "full_mrr", "full_hit1", "full_hit25", "isomer_mrr", "isomer_hit1"])
        writer.writeheader()
        writer.writerows(sweep_rows)

    # Save benchmark metrics JSON
    benchmark_metrics = {
        "checkpoint_selection_tuning_pool": ckpt_results,
        "selected_stage5_checkpoint": best_ckpt_name,
        "calibration_parameters": calibrator.to_dict(),
        "optimal_weights": {"weight_mass": best_w_mass, "weight_reranker": best_w_rerank},
        "systems": {name: m for name, m in systems},
        "router_sweep": sweep_rows,
    }
    metrics_path = out_dir / "benchmark_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_metrics, f, indent=2)

    print(f"\nAll benchmark contracts and artifacts saved to: {out_dir}", flush=True)
    print("=" * 80, flush=True)
    return benchmark_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="exp6a")
    args = parser.parse_args()
    run_stage6_pipeline(args.exp_name)
