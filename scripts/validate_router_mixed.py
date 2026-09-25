"""Stage 6.1: Mixed Router Validation Benchmark (Known vs Zero-Reference).

Addresses Issue 1 raised by the user:
In a pure Protocol C benchmark, true molecules are never in the library, so any library match
is technically a false positive. In a real MS/MS identification problem, some compounds
ARE present in reference libraries (Known / Class 1) while others are completely novel (Zero-Reference / Class 2-3).

This script constructs a rigorous balanced mixed benchmark using the 800 Tuning Molecules:
1. 400 Known-Reference Queries (Class 1 simulation):
   - Query spectra matched against the library with leave-one-spectrum-out (sibling spectra present).
2. 400 Zero-Reference Queries (Class 2/3 simulation):
   - Target molecule strictly absent from the library (drawn from the 800 tuning pool).

Sweeps tau in [0.40, 0.99] to find the genuine optimal confidence operating point tau*
that maximizes overall system MRR@25.

Outputs:
- artifacts/stage06/mixed_router_sweep.csv
- artifacts/stage06/router_calibration_analysis.json
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

from src.core.config import ARTIFACTS_DIR, TRAIN_PATH
from src.core.canonical_benchmark import CanonicalBenchmark
from src.core.evaluation import summarize_ranks
from src.data.spectrum_dataset import spectrum_to_coarse_bins
from src.models.spectrum_encoder import SpectrumEncoder
from src.models.molecule_encoder import MoleculeGNN
from src.models.reranker import CrossModalReranker
from src.models.ensemble import GlobalScoreCalibrator, CalibratedFusedRanker


@torch.no_grad()
def run_mixed_router_validation() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80, flush=True)
    print(f"  STAGE 6.1: MIXED ROUTER VALIDATION BENCHMARK ({device})", flush=True)
    print("  (Evaluating Known-Reference vs Zero-Reference Routing Decision)", flush=True)
    print("=" * 80, flush=True)

    out_dir = ARTIFACTS_DIR / "stage06"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Canonical Benchmark and Models
    benchmark = CanonicalBenchmark(
        subset_size=10000,
        n_benchmark_queries=200,
        split_seed=42,
        benchmark_seed=123,
    )

    calibrator = GlobalScoreCalibrator.load(out_dir / "calibration.json")
    print(f"Loaded calibrator: reranker_a={calibrator.reranker_a:.4f}, tau_mass={calibrator.tau_mass}", flush=True)

    spec_encoder = SpectrumEncoder(embed_dim=256).to(device)
    spec_ckpt = ARTIFACTS_DIR / "stage02/exp2a/checkpoints/best.pt"
    spec_data = torch.load(spec_ckpt, map_location=device, weights_only=False)
    spec_encoder.load_state_dict(spec_data.get("model_state_dict", spec_data))
    spec_encoder.eval()

    s5_ckpt = ARTIFACTS_DIR / "stage05/exp5a/checkpoints/best.pt"
    s5_data = torch.load(s5_ckpt, map_location=device, weights_only=False)

    mol_encoder = MoleculeGNN(embed_dim=256).to(device)
    mol_encoder.load_state_dict(s5_data["mol_encoder_state_dict"])
    mol_encoder.eval()

    reranker = CrossModalReranker(embed_dim=256, physics_dim=4, hidden_dim=256).to(device)
    reranker.load_state_dict(s5_data["reranker_state_dict"])
    reranker.eval()

    # Pre-encode all candidate graphs
    cand_mols = benchmark.cand_db.valid_mols
    cand_graphs = [benchmark.cand_db.mol_graphs[m] for m in cand_mols]
    batch_size = 64
    z_mols_list = []
    for i in range(0, len(cand_graphs), batch_size):
        bg = Batch.from_data_list(cand_graphs[i:i + batch_size]).to(device)
        z_mols_list.append(mol_encoder(bg).cpu())
    cand_embs = torch.cat(z_mols_list, dim=0).to(device)

    # Pre-index Reference Library (Training Spectra)
    print("\nPre-indexing reference spectral library...", flush=True)
    train_spectra_tensors = []
    train_mol_ids = []
    train_spectrum_ids = []
    for idx, (s_info, mol_id) in enumerate(benchmark.train_ds.samples):
        binned = spectrum_to_coarse_bins(s_info["mz"], s_info["intensity"])
        train_spectra_tensors.append(torch.from_numpy(binned))
        train_mol_ids.append(mol_id)
        train_spectrum_ids.append(idx)

    lib_tensors = torch.stack(train_spectra_tensors, dim=0).to(device)
    lib_tensors = nn.functional.normalize(lib_tensors, dim=-1)

    # 2. Construct 400 Known-Reference Queries (Class 1 Simulation)
    # Pick molecules with >= 2 spectra in train_ds
    print("\nConstructing 400 Known-Reference queries with leave-one-spectrum-out...", flush=True)
    mol_sample_indices: dict[str, list[int]] = {}
    for idx, (_, mol_id) in enumerate(benchmark.train_ds.samples):
        mol_sample_indices.setdefault(mol_id, []).append(idx)

    multi_spec_mols = [m for m, indices in mol_sample_indices.items() if len(indices) >= 2]
    rng = np.random.default_rng(42)
    selected_known_mols = rng.choice(multi_spec_mols, size=min(400, len(multi_spec_mols)), replace=False)

    known_queries_data = []
    for m in selected_known_mols:
        indices = mol_sample_indices[m]
        query_sample_idx = indices[0]
        sibling_indices = set(indices[1:])

        s_tensor, _, true_mol = benchmark.train_ds[query_sample_idx]
        q_coarse = s_tensor[:1480].unsqueeze(0).to(device)
        q_coarse = nn.functional.normalize(q_coarse, dim=-1)

        # Compute cosine against all library spectra
        cosines = torch.mm(q_coarse, lib_tensors.T).cpu().numpy()[0]
        # Zero out the exact query spectrum itself to simulate genuine retrieval of sibling spectra
        cosines[query_sample_idx] = -1.0

        best_lib_idx = int(np.argmax(cosines))
        max_cos = float(cosines[best_lib_idx])
        best_lib_mol = train_mol_ids[best_lib_idx]
        is_correct_lib_match = (best_lib_mol == true_mol)

        # Candidate pool for this molecule
        spec_info = benchmark.train_ds.samples[query_sample_idx][0]
        prec_mz = spec_info.get("precursor_mz", 0.0)
        adduct = spec_info.get("adduct", "[M+H]+")
        matches = benchmark.cand_db.query_two_tier(prec_mz, adduct, 20.0, 50.0, True)

        known_queries_data.append({
            "type": "known",
            "true_mol": true_mol,
            "max_lib_cosine": max_cos,
            "best_lib_mol": best_lib_mol,
            "is_correct_lib_match": is_correct_lib_match,
            "matches": matches,
            "spec_tensor": s_tensor,
            "prec_mz": prec_mz,
        })

    # 3. Construct 400 Zero-Reference Queries (Class 2/3 Simulation)
    print("Constructing 400 Zero-Reference queries from Tuning Pool...", flush=True)
    zero_ref_queries = benchmark.tuning_queries[:400]
    zero_queries_data = []

    for q in zero_ref_queries:
        q_coarse = q.spec_tensor[:1480].unsqueeze(0).to(device)
        q_coarse = nn.functional.normalize(q_coarse, dim=-1)
        cosines = torch.mm(q_coarse, lib_tensors.T).cpu().numpy()[0]
        best_lib_idx = int(np.argmax(cosines))
        max_cos = float(cosines[best_lib_idx])
        best_lib_mol = train_mol_ids[best_lib_idx]
        # In zero-reference, the true molecule is NEVER in the library!
        is_correct_lib_match = False

        zero_queries_data.append({
            "type": "zero_ref",
            "true_mol": q.true_mol,
            "max_lib_cosine": max_cos,
            "best_lib_mol": best_lib_mol,
            "is_correct_lib_match": is_correct_lib_match,
            "matches": q.matches,
            "spec_tensor": q.spec_tensor,
            "prec_mz": q.precursor_mz,
        })

    mixed_universe = known_queries_data + zero_queries_data
    print(f"Total Mixed Benchmark: {len(mixed_universe)} queries (400 Known + 400 Zero-Reference).", flush=True)

    # Precompute Stage 6 Calibrated Fused Ranks for all 800 queries
    print("Pre-computing calibrated physics + neural reranker scores on candidate pools...", flush=True)
    for item in mixed_universe:
        matches = item["matches"]
        true_mol = item["true_mol"]
        matched_indices = [benchmark.mol_to_idx[m.mol] for m in matches if m.mol in benchmark.mol_to_idx]

        if len(matched_indices) > 0:
            sub_z_mols = cand_embs[matched_indices]
            sub_z_spec = spec_encoder(item["spec_tensor"].unsqueeze(0).to(device))

            ppm_errors = np.array([m.ppm_error for m in matches if m.mol in benchmark.mol_to_idx], dtype=np.float32)
            tiers = np.array([m.tier for m in matches if m.mol in benchmark.mol_to_idx], dtype=np.int32)
            prec_norm = item["prec_mz"] / 1000.0
            phys_list = [
                [min(err / 20.0, 3.0), 1.0 if t == 1 else 0.5, prec_norm, 1.0 if err <= 5.0 else 0.0]
                for err, t in zip(ppm_errors, tiers)
            ]
            sub_phys = torch.tensor(phys_list, dtype=torch.float32, device=device)
            raw_scores = reranker(sub_z_spec, sub_z_mols, sub_phys).cpu().numpy()

            s_mass = calibrator.calibrate_mass_error(ppm_errors, tiers)
            s_rerank = calibrator.calibrate_reranker(raw_scores)
            fused_score = 0.60 * s_mass + 0.40 * s_rerank

            sort_order = np.argsort(-fused_score)
            valid_matches = [m for m in matches if m.mol in benchmark.mol_to_idx]
            ranked_mols = [valid_matches[k].mol for k in sort_order][:25]
            item["ranked_fused"] = ranked_mols
        else:
            item["ranked_fused"] = []

    # 4. Sweep Tau in [0.40, 0.99]
    print("\n" + "=" * 80, flush=True)
    print("  MIXED ROUTER THRESHOLD SWEEP RESULTS", flush=True)
    print("=" * 80, flush=True)
    header = f"{'tau':<6} | {'Usage Known':<12} | {'Usage Novel':<12} | {'Lib Precision':<14} | {'Known MRR':<10} | {'Novel MRR':<10} | {'Overall MRR':<12} | {'Overall H@1':<12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    thresholds = [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
    sweep_rows = []
    best_tau = None
    best_overall_mrr = -1.0

    for tau in thresholds:
        ranks_all = []
        ranks_known = []
        ranks_novel = []

        n_known_routed = 0
        n_novel_routed = 0
        n_true_positive_routes = 0

        for item in mixed_universe:
            max_cos = item["max_lib_cosine"]
            best_lib_mol = item["best_lib_mol"]
            true_mol = item["true_mol"]
            fused_cands = item["ranked_fused"]

            if max_cos >= tau:
                # Routed to Stage 1
                if item["type"] == "known":
                    n_known_routed += 1
                else:
                    n_novel_routed += 1

                if item["is_correct_lib_match"]:
                    n_true_positive_routes += 1

                cand_list = [best_lib_mol] + [m for m in fused_cands if m != best_lib_mol]
            else:
                # Routed to Stage 6 Calibrated Fusion
                cand_list = fused_cands

            cand_list = cand_list[:25]
            r = cand_list.index(true_mol) + 1 if true_mol in cand_list else 0

            ranks_all.append(r)
            if item["type"] == "known":
                ranks_known.append(r)
            else:
                ranks_novel.append(r)

        total_routed = n_known_routed + n_novel_routed
        lib_precision = (n_true_positive_routes / total_routed) * 100 if total_routed > 0 else 0.0

        pct_known_used = (n_known_routed / 400) * 100
        pct_novel_used = (n_novel_routed / 400) * 100

        m_all = summarize_ranks(ranks_all, k=25)
        m_known = summarize_ranks(ranks_known, k=25)
        m_novel = summarize_ranks(ranks_novel, k=25)

        row = {
            "tau": tau,
            "known_usage_pct": round(pct_known_used, 2),
            "novel_usage_pct": round(pct_novel_used, 2),
            "library_precision_pct": round(lib_precision, 2),
            "known_mrr": round(float(m_known["mrr"]), 4),
            "known_hit1": round(float(m_known["hit@1"]), 4),
            "novel_mrr": round(float(m_novel["mrr"]), 4),
            "novel_hit1": round(float(m_novel["hit@1"]), 4),
            "overall_mrr": round(float(m_all["mrr"]), 4),
            "overall_hit1": round(float(m_all["hit@1"]), 4),
            "overall_hit25": round(float(m_all["hit@25"]), 4),
        }
        sweep_rows.append(row)

        print(
            f"{tau:<6.2f} | "
            f"{pct_known_used:<11.1f}% | "
            f"{pct_novel_used:<11.1f}% | "
            f"{lib_precision:<13.1f}% | "
            f"{m_known['mrr']:<10.4f} | "
            f"{m_novel['mrr']:<10.4f} | "
            f"{m_all['mrr']:<12.4f} | "
            f"{m_all['hit@1']*100:<11.2f}%",
            flush=True,
        )

        if m_all["mrr"] > best_overall_mrr:
            best_overall_mrr = m_all["mrr"]
            best_tau = tau

    print(f"\n[Optimal Operating Point] Peak Overall MRR = {best_overall_mrr:.4f} achieved at tau = {best_tau:.2f}!", flush=True)

    # Save to CSV
    csv_path = out_dir / "mixed_router_sweep.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    # Save Analysis JSON
    analysis_path = out_dir / "router_calibration_analysis.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_tau": best_tau,
            "peak_overall_mrr": best_overall_mrr,
            "sweep": sweep_rows,
        }, f, indent=2)

    print(f"Results saved to: {csv_path} and {analysis_path}", flush=True)
    print("=" * 80, flush=True)
    return {"best_tau": best_tau, "peak_mrr": best_overall_mrr}


if __name__ == "__main__":
    run_mixed_router_validation()
