"""Sweep Library Override Thresholds on the 400-query Mixed Benchmark (200 Known + 200 Novel).

Directly addresses User Step 2:
Test thresholds: 0.45, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.01 (no override).
Evaluates using the exact Stage 5 + Stage 6 scoring architecture:
- W_rerank = 2.00 (frozen)
- Platt calibrated reranker probabilities: 1 / (1 + exp(-(4.2645 * s - 1.0935)))
- Mass physics: s_mass = exp(-ppm / 10.0)
- Library override:
    if s_spec >= tau:
        score += 4.50 * (s_spec ** 2)
    elif s_spec >= 0.35:
        score += 0.75 * s_spec
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.core.canonical_benchmark import CanonicalBenchmark, CanonicalQuery
from src.data.spectrum_dataset import spectrum_to_coarse_bins
from src.models.molecule_encoder import MoleculeGNN
from src.models.reranker import CrossModalReranker
from src.models.spectrum_encoder import SpectrumEncoder


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # 1. Load 200 novel benchmark queries
    bm = CanonicalBenchmark(subset_size=10000, n_benchmark_queries=200, split_seed=42, benchmark_seed=123)
    novel_queries = bm.benchmark_queries
    cand_db = bm.cand_db
    cand_mols = cand_db.valid_mols
    mol_to_idx = bm.mol_to_idx

    # 2. Build 200 known in-library benchmark queries
    train_samples_by_mol: dict[str, list[int]] = {}
    for s_idx, (_, mol_id) in enumerate(bm.train_ds.samples):
        train_samples_by_mol.setdefault(mol_id, []).append(s_idx)
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
        t_idx = mol_to_idx[true_mol]
        t_mass = cand_db.raw_masses[t_idx]
        matches = cand_db.query_two_tier(
            prec_mz if prec_mz > 0 else t_mass + 1.0078,
            adduct if adduct else "[M+H]+",
            20.0,
            50.0,
            True,
        )
        f_true = bm.cand_index.formula_map.get(true_mol, "")
        has_isomers = any(
            m_c.mol != true_mol and bm.cand_index.formula_map.get(m_c.mol, "") == f_true
            for m_c in matches
        )
        known_queries.append(
            CanonicalQuery(
                int(s_idx),
                true_mol,
                t_mass,
                prec_mz,
                adduct,
                spec_tensor,
                matches,
                has_isomers,
                f_true,
            )
        )

    print(f"Loaded {len(novel_queries)} Novel Queries and {len(known_queries)} Known Queries.")

    # 3. Load Stage 2 Spectrum Encoder & Stage 5 Reranker
    spec_encoder = SpectrumEncoder(embed_dim=256).to(device)
    spec_data = torch.load("kaggle_dataset/spec_encoder.pt", map_location=device, weights_only=False)
    spec_encoder.load_state_dict(spec_data.get("model_state_dict", spec_data))
    spec_encoder.eval()

    reranker = CrossModalReranker(embed_dim=256, physics_dim=4, hidden_dim=256).to(device)
    rerank_data = torch.load("kaggle_dataset/reranker.pt", map_location=device, weights_only=False)
    reranker.load_state_dict(rerank_data.get("reranker_state_dict", rerank_data))
    reranker.eval()

    # 4. Load Candidate Embeddings
    s5_data = torch.load("artifacts/stage05/exp5a/checkpoints/best.pt", map_location=device, weights_only=False)
    mol_encoder = MoleculeGNN(embed_dim=256).to(device)
    mol_encoder.load_state_dict(s5_data["mol_encoder_state_dict"])
    mol_encoder.eval()

    cand_graphs = [cand_db.mol_graphs[m] for m in cand_mols]
    z_mols_list = []
    with torch.no_grad():
        for i in range(0, len(cand_graphs), 256):
            bg = Batch.from_data_list(cand_graphs[i:i + 256]).to(device)
            z_mols_list.append(mol_encoder(bg))
    cand_embs = torch.cat(z_mols_list, dim=0)

    # 5. Build Reference Library Tensors
    train_spectra_tensors = []
    train_mol_ids = []
    for s_info, mol_id in bm.train_ds.samples:
        binned = spectrum_to_coarse_bins(s_info["mz"], s_info["intensity"])
        train_spectra_tensors.append(torch.from_numpy(binned))
        train_mol_ids.append(mol_id)
    lib_tensors = F.normalize(torch.stack(train_spectra_tensors, dim=0).to(device), dim=-1)

    all_queries = known_queries + novel_queries
    query_specs = torch.stack([q.spec_tensor for q in all_queries], dim=0).to(device)
    with torch.no_grad():
        z_queries = spec_encoder(query_specs)

    query_coarse = F.normalize(
        torch.stack([q.spec_tensor[:1480] for q in all_queries], dim=0).to(device),
        dim=-1,
    )
    cos_matrix = torch.mm(query_coarse, lib_tensors.T).cpu().numpy()

    # Precompute neural rerank predictions for all queries
    print("Precomputing Stage 5 neural reranking for all 400 queries...")
    query_data = []
    for i, q in enumerate(all_queries):
        is_known = (i < len(known_queries))
        true_mol = q.true_mol
        matches = q.matches
        if len(matches) == 0:
            query_data.append(None)
            continue

        matched_mols = [m.mol for m in matches]
        matched_indices = [mol_to_idx[m.mol] for m in matches]
        best_lib_idx = int(np.argmax(cos_matrix[i]))
        max_cos = float(cos_matrix[i, best_lib_idx])
        best_lib_mol = train_mol_ids[best_lib_idx]

        ppm_errors = np.array([m.ppm_error for m in matches], dtype=np.float32)
        s_mass = np.exp(-ppm_errors / 10.0)

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
        calib_prob = 1.0 / (1.0 + np.exp(-np.clip(4.2645 * raw_rerank - 1.0935, -15.0, 15.0)))

        query_data.append({
            "is_known": is_known,
            "true_mol": true_mol,
            "sub_mols": [m.mol for m in sub_matches],
            "base_scores": s_mass[initial_order] + 2.00 * calib_prob,
            "max_cos": max_cos,
            "best_lib_mol": best_lib_mol,
            "is_isomer": q.is_isomer_query,
        })

    # Sweep thresholds
    thresholds = [0.45, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.01]

    print("\n" + "=" * 90)
    print(f"{'Threshold (tau)':<16} | {'Known MRR@25':<14} | {'Novel MRR@25':<14} | {'Isomer MRR@25':<14} | {'Combined MRR@25':<16}")
    print("=" * 90)

    for tau in thresholds:
        k_recips, n_recips, iso_recips = [], [], []

        for qd in query_data:
            if qd is None:
                continue

            sub_mols = list(qd["sub_mols"])
            fused = qd["base_scores"].copy()
            max_cos = qd["max_cos"]
            best_lib_mol = qd["best_lib_mol"]
            true_mol = qd["true_mol"]
            is_known = qd["is_known"]

            # Library boost policy
            if max_cos >= tau and best_lib_mol in sub_mols:
                fused[sub_mols.index(best_lib_mol)] += 4.50 * (max_cos ** 2)
            elif max_cos >= 0.35 and best_lib_mol in sub_mols:
                fused[sub_mols.index(best_lib_mol)] += 0.75 * max_cos

            ranked = [sub_mols[k] for k in np.argsort(-fused)][:25]
            if max_cos >= tau and best_lib_mol not in ranked:
                ranked = [best_lib_mol] + ranked[:24]

            r = ranked.index(true_mol) + 1 if true_mol in ranked else 0
            rr = 1.0 / r if r > 0 else 0.0

            if is_known:
                k_recips.append(rr)
            else:
                n_recips.append(rr)
                if qd["is_isomer"]:
                    iso_recips.append(rr)

        k_mrr = float(np.mean(k_recips))
        n_mrr = float(np.mean(n_recips))
        iso_mrr = float(np.mean(iso_recips))
        comb_mrr = 0.5 * (k_mrr + n_mrr)

        tau_label = f"tau = {tau:.2f}" if tau <= 1.0 else "tau = None (1.01)"
        print(f"{tau_label:<16} | {k_mrr:<14.4f} | {n_mrr:<14.4f} | {iso_mrr:<14.4f} | {comb_mrr:<16.4f}")

    print("=" * 90)


if __name__ == "__main__":
    main()
