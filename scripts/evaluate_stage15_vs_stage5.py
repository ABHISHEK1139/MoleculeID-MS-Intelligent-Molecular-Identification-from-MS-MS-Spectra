"""Head-to-head local benchmark comparison: Stage 1.5 Only vs Stage 1.5 -> Stage 5 Reranker.

Evaluates on the frozen 200 out-of-sample benchmark queries:
- Full 200 Queries
- Exact-Isomer Subset (queries with same-formula constitutional isomers in pool)

Metrics: MRR@25, Hit@1, Hit@5, Hit@25
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

from src.core.config import ARTIFACTS_DIR
from src.core.canonical_benchmark import CanonicalBenchmark
from src.core.evaluation import summarize_ranks
from src.models.spectrum_encoder import SpectrumEncoder
from src.models.reranker import CrossModalReranker


def evaluate_ranking(ranks: list[int]) -> dict[str, float]:
    valid = [r for r in ranks if r > 0]
    reciprocal_ranks = [1.0 / r if r > 0 else 0.0 for r in ranks]
    mrr = float(np.mean(reciprocal_ranks))
    hit1 = float(np.mean([1.0 if r == 1 else 0.0 for r in ranks]))
    hit5 = float(np.mean([1.0 if 1 <= r <= 5 else 0.0 for r in ranks]))
    hit25 = float(np.mean([1.0 if 1 <= r <= 25 else 0.0 for r in ranks]))
    return {
        "mrr": round(mrr, 4),
        "hit1": round(hit1 * 100, 2),
        "hit5": round(hit5 * 100, 2),
        "hit25": round(hit25 * 100, 2),
    }


def print_table(results_a: dict[str, Any], results_b: dict[str, Any], title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Metric':<12} | {'Stage 1.5 Only':<16} | {'Stage 1.5 -> Stage 5':<22} | {'Delta':<10}")
    print("-" * 68)
    for m in ["mrr", "hit1", "hit5", "hit25"]:
        v_a = results_a[m]
        v_b = results_b[m]
        diff = v_b - v_a
        diff_str = f"{diff:+.4f}" if m == "mrr" else f"{diff:+.2f}%"
        unit = "" if m == "mrr" else "%"
        print(f"{m.upper():<12} | {v_a:<16}{unit} | {v_b:<22}{unit} | {diff_str}")


def main() -> None:
    print("=" * 80)
    print("  RIGOROUS BENCHMARK: STAGE 1.5 ONLY vs STAGE 1.5 -> STAGE 5 RERANKER")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # 1. Load Canonical Benchmark (200 frozen out-of-sample queries)
    benchmark = CanonicalBenchmark(
        subset_size=10000,
        n_benchmark_queries=200,
        split_seed=42,
        benchmark_seed=123,
    )
    queries = benchmark.benchmark_queries
    cand_db = benchmark.cand_db
    cand_mols = cand_db.valid_mols
    mol_to_idx = benchmark.mol_to_idx

    print(f"Loaded {len(queries)} frozen benchmark queries.")
    iso_query_indices = [i for i, q in enumerate(queries) if q.is_isomer_query]
    print(f"Exact Same-Formula Multi-Isomer queries: {len(iso_query_indices)} / {len(queries)}")

    # 2. Load Models
    spec_encoder = SpectrumEncoder(embed_dim=256).to(device)
    spec_data = torch.load("kaggle_dataset/spec_encoder.pt", map_location=device, weights_only=False)
    spec_encoder.load_state_dict(spec_data.get("model_state_dict", spec_data))
    spec_encoder.eval()

    reranker = CrossModalReranker(embed_dim=256, physics_dim=4, hidden_dim=256).to(device)
    rerank_data = torch.load("kaggle_dataset/reranker.pt", map_location=device, weights_only=False)
    reranker.load_state_dict(rerank_data.get("reranker_state_dict", rerank_data))
    reranker.eval()

    # 3. Load Candidate Embeddings
    print("Loading candidate embeddings (or encoding on the fly for benchmark catalog)...")
    # For the benchmark's 10,000 catalog, encode using fine-tuned MoleculeGNN
    s5_data = torch.load("artifacts/stage05/exp5a/checkpoints/best.pt", map_location=device, weights_only=False)
    from src.models.molecule_encoder import MoleculeGNN
    mol_encoder = MoleculeGNN(embed_dim=256).to(device)
    mol_encoder.load_state_dict(s5_data["mol_encoder_state_dict"])
    mol_encoder.eval()

    from torch_geometric.data import Batch
    cand_graphs = [cand_db.mol_graphs[m] for m in cand_mols]
    batch_size = 256
    z_mols_list = []
    with torch.no_grad():
        for i in range(0, len(cand_graphs), batch_size):
            bg = Batch.from_data_list(cand_graphs[i:i + batch_size]).to(device)
            z_mols_list.append(mol_encoder(bg))
    cand_embs = torch.cat(z_mols_list, dim=0)  # (10000, 256)

    # 4. Build Reference Library for Stage 1.5
    from src.data.spectrum_dataset import spectrum_to_coarse_bins
    train_spectra_tensors = []
    train_mol_ids = []
    for s_info, mol_id in benchmark.train_ds.samples:
        binned = spectrum_to_coarse_bins(s_info["mz"], s_info["intensity"])
        train_spectra_tensors.append(torch.from_numpy(binned))
        train_mol_ids.append(mol_id)
    lib_tensors = torch.stack(train_spectra_tensors, dim=0).to(device)
    lib_tensors = F.normalize(lib_tensors, dim=-1)

    # 5. Encode Benchmark Queries
    query_specs = torch.stack([q.spec_tensor for q in queries], dim=0).to(device)
    with torch.no_grad():
        z_queries = spec_encoder(query_specs)  # (200, 256)

    query_coarse = torch.stack([q.spec_tensor[:1480] for q in queries], dim=0).to(device)
    query_coarse = F.normalize(query_coarse, dim=-1)
    cos_matrix = torch.mm(query_coarse, lib_tensors.T).cpu().numpy()  # (200, N_lib)

    # Calibration parameters
    RERANKER_A = 4.2645
    RERANKER_B = -1.0935
    TAU_MASS = 10.0

    ranks_stage15_only: list[int] = []
    ranks_stage5_reranked: list[int] = []

    for i, q in enumerate(queries):
        true_mol = q.true_mol
        matches = q.matches  # Top mass matches from physics gate
        if len(matches) == 0:
            ranks_stage15_only.append(0)
            ranks_stage5_reranked.append(0)
            continue

        matched_mols = [m.mol for m in matches]
        matched_indices = [mol_to_idx[m.mol] for m in matches]

        # Stage 1.5 Cosine against reference library
        best_lib_idx = int(np.argmax(cos_matrix[i]))
        max_cos = float(cos_matrix[i, best_lib_idx])
        best_lib_mol = train_mol_ids[best_lib_idx]

        # Candidate physics scores
        ppm_errors = np.array([m.ppm_error for m in matches], dtype=np.float32)
        tiers = np.array([m.tier for m in matches], dtype=np.int32)
        tier_weights = np.where(tiers == 1, 1.0, 0.50).astype(np.float32)
        s_mass = np.exp(-ppm_errors / TAU_MASS) * tier_weights

        # ── SYSTEM A: STAGE 1.5 ONLY ──
        # Candidate ranking: Top library hit + Physics mass ordering
        if max_cos >= 0.45 and best_lib_mol in matched_mols:
            # Boost library hit
            s_st15 = s_mass.copy()
            hit_pos = matched_mols.index(best_lib_mol)
            s_st15[hit_pos] += 4.50 * (max_cos ** 2)
            ranked_a = [matched_mols[k] for k in np.argsort(-s_st15)][:25]
        elif max_cos >= 0.45:
            ranked_a = [best_lib_mol] + [m for m in matched_mols if m != best_lib_mol][:24]
        else:
            ranked_a = [matched_mols[k] for k in np.argsort(-s_mass)][:25]

        ranks_stage15_only.append(ranked_a.index(true_mol) + 1 if true_mol in ranked_a else 0)

        # ── SYSTEM B: STAGE 1.5 -> STAGE 5 RERANKER ──
        # Take Top 25-50 candidates from Stage 1.5 / physics and rerank with Stage 5
        top_cand_k = min(len(matches), 50)
        initial_order = np.argsort(-s_mass)[:top_cand_k]
        sub_matches = [matches[k] for k in initial_order]
        sub_indices = [matched_indices[k] for k in initial_order]

        sub_z_mols = cand_embs[sub_indices]  # (K, 256)
        sub_z_spec = z_queries[i:i + 1]      # (1, 256)

        prec_norm = q.precursor_mz / 1000.0
        phys_list = [
            [min(m.ppm_error / 20.0, 3.0), 1.0 if m.tier == 1 else 0.5, prec_norm, 1.0 if m.ppm_error <= 5.0 else 0.0]
            for m in sub_matches
        ]
        sub_phys = torch.tensor(phys_list, dtype=torch.float32, device=device)

        with torch.no_grad():
            raw_rerank = reranker(sub_z_spec, sub_z_mols, sub_phys).cpu().numpy()

        calib_prob = 1.0 / (1.0 + np.exp(-np.clip(RERANKER_A * raw_rerank + RERANKER_B, -15.0, 15.0)))

        # Fuse: Mass physics + Stage 5 reranker probability
        sub_mass_scores = s_mass[initial_order]
        fused_scores = 0.50 * sub_mass_scores + 0.50 * calib_prob

        # If high-confidence library match is present, inject it at top
        sub_mols = [m.mol for m in sub_matches]
        if max_cos >= 0.70 and best_lib_mol in sub_mols:
            lib_idx = sub_mols.index(best_lib_mol)
            fused_scores[lib_idx] += 3.0

        reranked_order = np.argsort(-fused_scores)
        ranked_b = [sub_mols[k] for k in reranked_order][:25]

        # If high-confidence library hit wasn't in physics window, place at #1
        if max_cos >= 0.75 and best_lib_mol not in ranked_b:
            ranked_b = [best_lib_mol] + ranked_b[:24]

        ranks_stage5_reranked.append(ranked_b.index(true_mol) + 1 if true_mol in ranked_b else 0)

    # 6. Compute Metrics
    # (A) Full 200 Queries
    ev_a_full = evaluate_ranking(ranks_stage15_only)
    ev_b_full = evaluate_ranking(ranks_stage5_reranked)
    print_table(ev_a_full, ev_b_full, f"FULL PROTOCOL C BENCHMARK (All {len(queries)} Queries)")

    # (B) Exact-Isomer Subset
    ranks_a_iso = [ranks_stage15_only[idx] for idx in iso_query_indices]
    ranks_b_iso = [ranks_stage5_reranked[idx] for idx in iso_query_indices]
    ev_a_iso = evaluate_ranking(ranks_a_iso)
    ev_b_iso = evaluate_ranking(ranks_b_iso)
    print_table(ev_a_iso, ev_b_iso, f"EXACT SAME-FORMULA ISOMER SUBSET ({len(iso_query_indices)} Queries)")


if __name__ == "__main__":
    main()
