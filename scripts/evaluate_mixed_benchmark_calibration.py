"""Stage 6 Calibrated Router & Hybrid Score Fusion: Mixed Benchmark Evaluation.

Evaluates override threshold tau across a balanced 400-query benchmark:
- 200 Known In-Library Queries (True molecule IS in reference library)
- 200 Novel Out-of-Library Queries (True molecule is NOT in reference library)

Measures:
- Known In-Library MRR@25 & Hit@1 (preservation of genuine spectral hits)
- Novel Out-of-Library MRR@25 & Hit@1 (suppression of false analog overrides)
- Combined Overall MRR@25 (Pareto-optimal operating point tau*)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.core.canonical_benchmark import CanonicalBenchmark, CanonicalQuery
from src.models.spectrum_encoder import SpectrumEncoder
from src.models.reranker import CrossModalReranker
from src.models.molecule_encoder import MoleculeGNN
from src.data.spectrum_dataset import spectrum_to_coarse_bins


def main() -> None:
    print("=" * 80)
    print("  STAGE 6: MIXED BENCHMARK CALIBRATION SWEEP (KNOWN vs NOVEL REGIMES)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # 1. Load Canonical Benchmark
    bm = CanonicalBenchmark(
        subset_size=10000,
        n_benchmark_queries=200,
        split_seed=42,
        benchmark_seed=123,
    )
    novel_queries = bm.benchmark_queries  # 200 novel out-of-library queries
    cand_db = bm.cand_db
    cand_mols = cand_db.valid_mols
    mol_to_idx = bm.mol_to_idx

    # 2. Build 200 Known In-Library Queries from train_ds
    print("Constructing 200 Known In-Library queries from training catalog...", flush=True)
    train_samples_by_mol: dict[str, list[int]] = {}
    for sample_idx, (_, mol_id) in enumerate(bm.train_ds.samples):
        train_samples_by_mol.setdefault(mol_id, []).append(sample_idx)

    train_unique_mols = sorted(list(train_samples_by_mol.keys()))
    rng = np.random.default_rng(999)
    known_mol_picks = rng.choice(train_unique_mols, size=200, replace=False)

    known_queries: list[CanonicalQuery] = []
    for m in known_mol_picks:
        s_idx = train_samples_by_mol[m][0]
        spec_tensor, _, true_mol = bm.train_ds[s_idx]
        spec_info = bm.train_ds.samples[s_idx][0]
        prec_mz = spec_info.get("precursor_mz", 0.0)
        adduct = spec_info.get("adduct", None)

        if true_mol not in cand_db.mol_smiles:
            continue
        t_idx = mol_to_idx[true_mol]
        t_mass = cand_db.raw_masses[t_idx]

        matches = cand_db.query_two_tier(
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
        known_queries.append(
            CanonicalQuery(
                query_id=int(s_idx),
                true_mol=true_mol,
                target_mass=t_mass,
                precursor_mz=prec_mz,
                adduct=adduct,
                spec_tensor=spec_tensor,
                matches=matches,
                is_isomer_query=has_isomers,
                true_formula=f_true,
            )
        )

    print(f"Total Evaluation Sets: {len(known_queries)} Known Queries | {len(novel_queries)} Novel Queries.", flush=True)

    # 3. Load Models
    spec_encoder = SpectrumEncoder(embed_dim=256).to(device)
    spec_data = torch.load("kaggle_dataset/spec_encoder.pt", map_location=device, weights_only=False)
    spec_encoder.load_state_dict(spec_data.get("model_state_dict", spec_data))
    spec_encoder.eval()

    reranker = CrossModalReranker(embed_dim=256, physics_dim=4, hidden_dim=256).to(device)
    rerank_data = torch.load("kaggle_dataset/reranker.pt", map_location=device, weights_only=False)
    reranker.load_state_dict(rerank_data.get("reranker_state_dict", rerank_data))
    reranker.eval()

    s5_data = torch.load("artifacts/stage05/exp5a/checkpoints/best.pt", map_location=device, weights_only=False)
    mol_encoder = MoleculeGNN(embed_dim=256).to(device)
    mol_encoder.load_state_dict(s5_data["mol_encoder_state_dict"])
    mol_encoder.eval()

    # 4. Precompute Candidate Embeddings for 10,000 Catalog
    print("Precomputing 256-D molecular embeddings for candidate catalog...", flush=True)
    cand_graphs = [cand_db.mol_graphs[m] for m in cand_mols]
    z_mols_list = []
    with torch.no_grad():
        for i in range(0, len(cand_graphs), 256):
            bg = Batch.from_data_list(cand_graphs[i:i + 256]).to(device)
            z_mols_list.append(mol_encoder(bg))
    cand_embs = torch.cat(z_mols_list, dim=0)

    # 5. Build Reference Library Tensors (Stage 1.5)
    train_spectra_tensors = []
    train_mol_ids = []
    for s_info, mol_id in bm.train_ds.samples:
        binned = spectrum_to_coarse_bins(s_info["mz"], s_info["intensity"])
        train_spectra_tensors.append(torch.from_numpy(binned))
        train_mol_ids.append(mol_id)
    lib_tensors = F.normalize(torch.stack(train_spectra_tensors, dim=0).to(device), dim=-1)

    # 6. Encode All 400 Queries
    all_queries = known_queries + novel_queries
    query_specs = torch.stack([q.spec_tensor for q in all_queries], dim=0).to(device)
    with torch.no_grad():
        z_queries = spec_encoder(query_specs)

    query_coarse = F.normalize(torch.stack([q.spec_tensor[:1480] for q in all_queries], dim=0).to(device), dim=-1)
    cos_matrix = torch.mm(query_coarse, lib_tensors.T).cpu().numpy()  # (400, N_lib)

    # Platt Calibration Parameters
    RERANKER_A = 4.2645
    RERANKER_B = -1.0935
    W_RERANK = 2.00  # Frozen winning weight

    # 7. Sweep Override Threshold tau
    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]

    print("\n" + "=" * 90)
    print(f"{'Threshold':<11} | {'Known MRR':<11} | {'Known Hit@1':<13} | {'Novel MRR':<11} | {'Novel Hit@1':<13} | {'Combined MRR':<13}")
    print("-" * 90)

    results_table = []
    for tau in thresholds:
        known_recips = []
        known_hits1 = []
        novel_recips = []
        novel_hits1 = []

        for i, q in enumerate(all_queries):
            is_known = (i < len(known_queries))
            true_mol = q.true_mol
            matches = q.matches
            if len(matches) == 0:
                if is_known:
                    known_recips.append(0.0)
                    known_hits1.append(0.0)
                else:
                    novel_recips.append(0.0)
                    novel_hits1.append(0.0)
                continue

            matched_mols = [m.mol for m in matches]
            matched_indices = [mol_to_idx[m.mol] for m in matches]

            # Library hit
            best_lib_idx = int(np.argmax(cos_matrix[i]))
            max_cos = float(cos_matrix[i, best_lib_idx])
            best_lib_mol = train_mol_ids[best_lib_idx]

            # Candidate physics
            ppm_errors = np.array([m.ppm_error for m in matches], dtype=np.float32)
            tiers = np.array([m.tier for m in matches], dtype=np.int32)
            tier_weights = np.where(tiers == 1, 1.0, 0.50).astype(np.float32)
            s_mass = np.exp(-ppm_errors / 10.0) * tier_weights

            top_k = min(len(matches), 50)
            initial_order = np.argsort(-s_mass)[:top_k]
            sub_matches = [matches[k] for k in initial_order]
            sub_indices = [matched_indices[k] for k in initial_order]

            sub_z_mols = cand_embs[sub_indices]
            sub_z_spec = z_queries[i:i + 1]

            prec_norm = q.precursor_mz / 1000.0
            phys_list = [
                [min(m.ppm_error / 20.0, 3.0), 1.0 if m.tier == 1 else 0.5, prec_norm, 1.0 if m.ppm_error <= 5.0 else 0.0]
                for m in sub_matches
            ]
            sub_phys = torch.tensor(phys_list, dtype=torch.float32, device=device)

            with torch.no_grad():
                raw_rerank = reranker(sub_z_spec, sub_z_mols, sub_phys).cpu().numpy()

            calib_prob = 1.0 / (1.0 + np.exp(-np.clip(RERANKER_A * raw_rerank + RERANKER_B, -15.0, 15.0)))

            # Fusion: Base physics + Stage 5 Neural Reranking
            fused_scores = s_mass[initial_order] + W_RERANK * calib_prob
            sub_mols = [m.mol for m in sub_matches]

            # Stage 6 Calibrated Router Policy:
            # ONLY apply decisive library override if spectral match reaches high-fidelity threshold tau
            if max_cos >= tau and best_lib_mol in sub_mols:
                lib_pos = sub_mols.index(best_lib_mol)
                fused_scores[lib_pos] += 4.50 * (max_cos ** 2)

            reranked_order = np.argsort(-fused_scores)
            ranked = [sub_mols[k] for k in reranked_order][:25]

            # If high-confidence match wasn't in physics window, place at #1
            if max_cos >= tau and best_lib_mol not in ranked:
                ranked = [best_lib_mol] + ranked[:24]

            rank = ranked.index(true_mol) + 1 if true_mol in ranked else 0
            rr = 1.0 / rank if rank > 0 else 0.0
            h1 = 1.0 if rank == 1 else 0.0

            if is_known:
                known_recips.append(rr)
                known_hits1.append(h1)
            else:
                novel_recips.append(rr)
                novel_hits1.append(h1)

        m_k = float(np.mean(known_recips))
        h_k = float(np.mean(known_hits1)) * 100
        m_n = float(np.mean(novel_recips))
        h_n = float(np.mean(novel_hits1)) * 100
        m_c = (m_k + m_n) / 2.0

        results_table.append({
            "tau": tau,
            "known_mrr": m_k,
            "known_hit1": h_k,
            "novel_mrr": m_n,
            "novel_hit1": h_n,
            "combined_mrr": m_c,
        })

        tag = " (Current)" if tau == 0.45 else (" (No Boost)" if tau > 1.0 else "")
        print(f"tau={tau:<6.2f}{tag:<10} | {m_k:<11.4f} | {h_k:<11.1f}% | {m_n:<11.4f} | {h_n:<11.1f}% | {m_c:<13.4f}")

    best_entry = max(results_table, key=lambda x: x["combined_mrr"])
    print("=" * 90)
    print(f"\nOptimal Operating Point: tau* = {best_entry['tau']:.2f}")
    print(f"  Combined MRR@25: {best_entry['combined_mrr']:.4f}")
    print(f"  Known In-Library MRR@25: {best_entry['known_mrr']:.4f} (Hit@1: {best_entry['known_hit1']:.1f}%)")
    print(f"  Novel Out-of-Library MRR@25: {best_entry['novel_mrr']:.4f} (Hit@1: {best_entry['novel_hit1']:.1f}%)")


if __name__ == "__main__":
    main()
