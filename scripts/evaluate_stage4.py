"""Stage 4 Evaluation: Physics & Formula Candidate Pruning with Cross-Modal GNN.

Evaluates how physics-informed two-tier mass gating (20/50 ppm + isotope fallback)
and chemical formula constraints prune candidate search spaces and dramatically boost
Protocol C zero-reference molecule identification.

Usage:
    python scripts/evaluate_stage4.py --config configs/stage04/exp4a.yaml --exp-name exp4a
"""
from __future__ import annotations

import argparse
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
from src.core.evaluation import summarize_ranks
from src.core.adducts import neutral_mass
from src.models.spectrum_encoder import SpectrumEncoder
from src.models.molecule_encoder import MoleculeGNN
from src.data.cross_modal_dataset import create_cross_modal_datasets
from src.search.candidate_generator import CandidateDatabase


@torch.no_grad()
def evaluate_benchmark(
    cand_db: CandidateDatabase,
    val_dataset: Any,
    spec_encoder: nn.Module,
    mol_encoder: nn.Module,
    device: torch.device,
    n_queries: int = 200,
    top_k: int = 25,
    ppm_primary: float = 20.0,
    ppm_fallback: float = 50.0,
    include_isotope: bool = True,
    seed: int = 123,
) -> dict[str, Any]:
    """Run comparative evaluation of Stage 3 (raw), Physics-only, and Stage 4 (Hybrid)."""
    spec_encoder.eval()
    mol_encoder.eval()

    # 1. Encode candidate graphs
    cand_mols = cand_db.valid_mols
    cand_graphs = [cand_db.mol_graphs[m] for m in cand_mols]
    mol_embs_list = []
    batch_size = 64
    for i in range(0, len(cand_graphs), batch_size):
        batch_g = Batch.from_data_list(cand_graphs[i:i + batch_size]).to(device)
        z_m = mol_encoder(batch_g)
        mol_embs_list.append(z_m.cpu())
    cand_embs = torch.cat(mol_embs_list, dim=0).numpy()  # (M_cand, 256)

    mol_to_idx = {m: i for i, m in enumerate(cand_mols)}

    # 2. Select query spectra from held-out validation set
    rng = np.random.default_rng(seed)
    n_q = min(n_queries, len(val_dataset))
    query_indices = rng.permutation(len(val_dataset))[:n_q]

    query_specs_list = []
    query_true_mols = []
    query_prec_mzs = []
    query_adducts = []
    query_target_masses = []

    for q_idx in query_indices:
        spec_tensor, _, true_mol = val_dataset[q_idx]
        spec_info = val_dataset.samples[q_idx][0]
        prec_mz = spec_info.get("precursor_mz", 0.0)
        adduct = spec_info.get("adduct", None)

        # True neutral mass of the ground-truth candidate
        if true_mol not in cand_db.mol_smiles:
            continue

        query_specs_list.append(spec_tensor)
        query_true_mols.append(true_mol)
        query_prec_mzs.append(prec_mz)
        query_adducts.append(adduct)
        query_target_masses.append(cand_db.raw_masses[mol_to_idx[true_mol]])

    query_specs = torch.stack(query_specs_list, dim=0).to(device)
    z_queries = spec_encoder(query_specs).cpu().numpy()  # (n_q, 256)

    # 3. Dense Cross-Modal Similarity Matrix: cos(z_spec, z_mol)
    sims_raw = z_queries @ cand_embs.T  # (n_q, M_cand)

    # Metrics accumulators
    ranks_stage3_raw: list[int] = []
    ranks_physics_only: list[int] = []
    ranks_stage4_hybrid: list[int] = []

    pool_sizes_primary: list[int] = []
    pool_sizes_total: list[int] = []
    target_in_primary_count = 0
    target_in_total_count = 0

    # 4. Evaluate each query
    for i, true_mol in enumerate(query_true_mols):
        target_idx = mol_to_idx[true_mol]
        target_mass = query_target_masses[i]
        prec_mz = query_prec_mzs[i]
        adduct = query_adducts[i]

        # ── Query Candidate Database with Two-Tier Mass Filter ──────────────
        matches = cand_db.query_two_tier(
            precursor_mz=prec_mz if prec_mz > 0 else target_mass + 1.0078,
            adduct=adduct if adduct is not None else "[M+H]+",
            ppm_primary=ppm_primary,
            ppm_fallback=ppm_fallback,
            include_isotope=include_isotope,
        )

        matched_mols = {m.mol: m for m in matches}
        tier1_mols = {m.mol for m in matches if m.tier == 1}

        # Track candidate pruning statistics
        pool_sizes_primary.append(len(tier1_mols))
        pool_sizes_total.append(len(matched_mols))

        if true_mol in tier1_mols:
            target_in_primary_count += 1
        if true_mol in matched_mols:
            target_in_total_count += 1

        # ── Mode 1: Stage 3 Unconstrained GNN ──────────────────────────────
        ranked_s3 = np.argsort(-sims_raw[i])[:top_k]
        match_s3 = np.where(ranked_s3 == target_idx)[0]
        ranks_stage3_raw.append(int(match_s3[0] + 1) if len(match_s3) > 0 else 0)

        # ── Mode 2: Physics Ordering Baseline (Sorted by tier ASC, ppm_error ASC) ──
        ranked_phys_mols = [m.mol for m in matches][:top_k]
        if true_mol in ranked_phys_mols:
            ranks_physics_only.append(ranked_phys_mols.index(true_mol) + 1)
        else:
            ranks_physics_only.append(0)

        # ── Mode 3: Stage 4 Physics + GNN Hybrid ───────────────────────────
        score_stage4 = np.full(len(cand_mols), -1e9, dtype=np.float32)
        for m_id, match_obj in matched_mols.items():
            idx = mol_to_idx[m_id]
            # GNN cosine similarity weighted by tier penalty multiplier
            score_stage4[idx] = sims_raw[i, idx] * match_obj.weight

        ranked_s4 = np.argsort(-score_stage4)[:top_k]
        match_s4 = np.where(ranked_s4 == target_idx)[0]
        ranks_stage4_hybrid.append(int(match_s4[0] + 1) if len(match_s4) > 0 else 0)

    # ── True Uniform Random Baseline (Averaged over 100 shuffle trials) ─────
    random_trial_mrrs = []
    random_trial_hit1s = []
    random_trial_hit5s = []
    random_trial_hit25s = []

    for trial in range(100):
        trial_ranks = []
        trial_rng = np.random.default_rng(1000 + trial)
        for i, true_mol in enumerate(query_true_mols):
            target_mass = query_target_masses[i]
            prec_mz = query_prec_mzs[i]
            adduct = query_adducts[i]

            matches = cand_db.query_two_tier(
                precursor_mz=prec_mz if prec_mz > 0 else target_mass + 1.0078,
                adduct=adduct if adduct is not None else "[M+H]+",
                ppm_primary=ppm_primary,
                ppm_fallback=ppm_fallback,
                include_isotope=include_isotope,
            )
            cand_ids = [m.mol for m in matches]
            trial_rng.shuffle(cand_ids)
            if true_mol in cand_ids:
                r = cand_ids.index(true_mol) + 1
                trial_ranks.append(r if r <= top_k else 0)
            else:
                trial_ranks.append(0)
        sm = summarize_ranks(trial_ranks, k=top_k)
        random_trial_mrrs.append(sm["mrr"])
        random_trial_hit1s.append(sm["hit@1"])
        random_trial_hit5s.append(sm.get("hit@5", 0.0))
        random_trial_hit25s.append(sm.get("hit@25", 0.0))

    m_random = {
        "n": len(query_true_mols),
        "mrr": float(np.mean(random_trial_mrrs)),
        "hit@1": float(np.mean(random_trial_hit1s)),
        "hit@5": float(np.mean(random_trial_hit5s)),
        "hit@25": float(np.mean(random_trial_hit25s)),
    }

    n_valid = len(query_true_mols)
    m_s3 = summarize_ranks(ranks_stage3_raw, k=top_k)
    m_phys = summarize_ranks(ranks_physics_only, k=top_k)
    m_s4 = summarize_ranks(ranks_stage4_hybrid, k=top_k)

    return {
        "candidate_pool_size": len(cand_mols),
        "n_queries": n_valid,
        "pruning": {
            "mean_primary_pool": float(np.mean(pool_sizes_primary)),
            "median_primary_pool": float(np.median(pool_sizes_primary)),
            "mean_total_pool": float(np.mean(pool_sizes_total)),
            "primary_recall": float(target_in_primary_count / max(n_valid, 1)),
            "total_recall": float(target_in_total_count / max(n_valid, 1)),
            "pruning_factor": float(len(cand_mols) / max(np.mean(pool_sizes_total), 1e-3)),
        },
        "stage3_unconstrained": m_s3,
        "uniform_random": m_random,
        "physics_ordering": m_phys,
        "stage4_hybrid": m_s4,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Formula + Physics Candidate Generation")
    parser.add_argument("--config", type=str, default="configs/stage04/exp4a.yaml", help="Path to config YAML")
    parser.add_argument("--exp-name", type=str, default="exp4a", help="Experiment name")
    parser.add_argument("--subset", type=int, default=10000, help="Number of molecules to load")
    parser.add_argument("--n-queries", type=int, default=200, help="Number of validation query spectra")
    parser.add_argument("--ppm-primary", type=float, default=20.0, help="Primary mass tolerance (ppm)")
    parser.add_argument("--ppm-fallback", type=float, default=50.0, help="Fallback mass tolerance (ppm)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Load YAML if present
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        if "physics" in cfg:
            args.ppm_primary = cfg["physics"].get("ppm_primary", args.ppm_primary)
            args.ppm_fallback = cfg["physics"].get("ppm_fallback", args.ppm_fallback)
        if "data" in cfg:
            args.subset = cfg["data"].get("subset_size", args.subset)
        if "evaluation" in cfg:
            args.n_queries = cfg["evaluation"].get("n_queries", args.n_queries)
        print(f"[stage4] loaded config from {args.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage4] device={device}, subset={args.subset}, queries={args.n_queries}")

    exp_dir = ARTIFACTS_DIR / "stage04" / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    print("[stage4] loading datasets with single parquet pass...")
    train_dataset, val_dataset = create_cross_modal_datasets(
        train_path=TRAIN_PATH,
        subset_size=args.subset,
        val_frac=0.10,
        seed=args.seed,
    )

    # 2. Load models
    print("[stage4] loading frozen Stage 2 SpectrumEncoder and Stage 3 MoleculeGNN...")
    spec_encoder = SpectrumEncoder(input_dim=1483, embed_dim=256, hidden_channels=128, n_blocks=4, dropout=0.1).to(device)
    spec_encoder.load_state_dict(torch.load("artifacts/stage02/exp2a/checkpoints/best.pt", map_location=device, weights_only=True))
    spec_encoder.eval()

    mol_encoder = MoleculeGNN(hidden_dim=128, embed_dim=256, n_layers=4, dropout=0.1).to(device)
    mol_encoder.load_state_dict(torch.load("artifacts/stage03/exp3a/checkpoints/best.pt", map_location=device, weights_only=True))
    mol_encoder.eval()
    print("  [*] models loaded and set to eval mode")

    all_results: dict[str, Any] = {}

    # ── Benchmark A: 1,000 Candidate Pool (Held-Out Zero-Reference Baseline) ──
    print("\n" + "=" * 70)
    print("BENCHMARK A: Held-Out Zero-Reference (1,000 Candidate Pool)")
    print("Direct comparison with Stage 3 baseline under identical candidates")
    print("=" * 70)

    val_cand_mols = [m for m in val_dataset.selected_mols if m in val_dataset.graph_cache]
    val_cand_db = CandidateDatabase(
        molecules=val_cand_mols,
        mol_smiles=val_dataset.mol_smiles,
        mol_graphs=val_dataset.graph_cache,
    )
    print(f"  Indexed {len(val_cand_db)} held-out candidate molecules")

    t0 = time.time()
    res_a = evaluate_benchmark(
        cand_db=val_cand_db,
        val_dataset=val_dataset,
        spec_encoder=spec_encoder,
        mol_encoder=mol_encoder,
        device=device,
        n_queries=args.n_queries,
        top_k=25,
        ppm_primary=args.ppm_primary,
        ppm_fallback=args.ppm_fallback,
    )
    res_a["elapsed_s"] = time.time() - t0
    all_results["benchmark_a_1000"] = res_a

    print(f"\n  [Benchmark A Results (1,000 Candidates)]")
    print(f"  Candidate Recall @ 20 ppm:      {res_a['pruning']['primary_recall'] * 100:.1f}%")
    print(f"  Search Space Pruning:           {res_a['pruning']['pruning_factor']:.1f}x reduction "
          f"(mean pool: {res_a['pruning']['mean_total_pool']:.1f} molecules, primary: {res_a['pruning']['mean_primary_pool']:.1f})")
    print(f"  A. True Uniform Random:         MRR@25 = {res_a['uniform_random']['mrr']:.4f} | "
          f"Hit@1 = {res_a['uniform_random']['hit@1'] * 100:.1f}% | "
          f"Hit@25 = {res_a['uniform_random'].get('hit@25', 0) * 100:.1f}%")
    print(f"  B. Physics Ordering (|ppm|):     MRR@25 = {res_a['physics_ordering']['mrr']:.4f} | "
          f"Hit@1 = {res_a['physics_ordering']['hit@1'] * 100:.1f}% | "
          f"Hit@25 = {res_a['physics_ordering'].get('hit@25', 0) * 100:.1f}%")
    print(f"  C. Stage 3 GNN (Unconstrained):  MRR@25 = {res_a['stage3_unconstrained']['mrr']:.4f} | "
          f"Hit@1 = {res_a['stage3_unconstrained']['hit@1'] * 100:.1f}% | "
          f"Hit@25 = {res_a['stage3_unconstrained'].get('hit@25', 0) * 100:.1f}%")
    print(f"  D. Stage 4 Hybrid (Physics+GNN): MRR@25 = {res_a['stage4_hybrid']['mrr']:.4f} | "
          f"Hit@1 = {res_a['stage4_hybrid']['hit@1'] * 100:.1f}% | "
          f"Hit@25 = {res_a['stage4_hybrid'].get('hit@25', 0) * 100:.1f}%")

    # ── Benchmark B: 10,000 Candidate Pool (Large Isomer-Dense Scale) ─────────
    print("\n" + "=" * 70)
    print("BENCHMARK B: Large Isomer-Dense Search (10,000 Candidate Pool)")
    print("Stress-testing candidate pruning and GNN ranking at 10x scale")
    print("=" * 70)

    # Combine train and val candidate molecules into 10,000 candidate pool
    all_mols = list(train_dataset.selected_mols) + list(val_dataset.selected_mols)
    all_smiles = {**train_dataset.mol_smiles, **val_dataset.mol_smiles}
    all_graphs = {**train_dataset.graph_cache, **val_dataset.graph_cache}
    valid_all_mols = [m for m in all_mols if m in all_graphs]

    large_cand_db = CandidateDatabase(
        molecules=valid_all_mols,
        mol_smiles=all_smiles,
        mol_graphs=all_graphs,
    )
    print(f"  Indexed {len(large_cand_db)} total candidate molecules")

    t1 = time.time()
    res_b = evaluate_benchmark(
        cand_db=large_cand_db,
        val_dataset=val_dataset,
        spec_encoder=spec_encoder,
        mol_encoder=mol_encoder,
        device=device,
        n_queries=args.n_queries,
        top_k=25,
        ppm_primary=args.ppm_primary,
        ppm_fallback=args.ppm_fallback,
    )
    res_b["elapsed_s"] = time.time() - t1
    all_results["benchmark_b_10000"] = res_b

    print(f"\n  [Benchmark B Results (10,000 Candidates)]")
    print(f"  Candidate Recall @ 20 ppm:      {res_b['pruning']['primary_recall'] * 100:.1f}%")
    print(f"  Search Space Pruning:           {res_b['pruning']['pruning_factor']:.1f}x reduction "
          f"(mean pool: {res_b['pruning']['mean_total_pool']:.1f} molecules, primary: {res_b['pruning']['mean_primary_pool']:.1f})")
    print(f"  A. True Uniform Random:         MRR@25 = {res_b['uniform_random']['mrr']:.4f} | "
          f"Hit@1 = {res_b['uniform_random']['hit@1'] * 100:.1f}% | "
          f"Hit@25 = {res_b['uniform_random'].get('hit@25', 0) * 100:.1f}%")
    print(f"  B. Physics Ordering (|ppm|):     MRR@25 = {res_b['physics_ordering']['mrr']:.4f} | "
          f"Hit@1 = {res_b['physics_ordering']['hit@1'] * 100:.1f}% | "
          f"Hit@25 = {res_b['physics_ordering'].get('hit@25', 0) * 100:.1f}%")
    print(f"  C. Stage 3 GNN (Unconstrained):  MRR@25 = {res_b['stage3_unconstrained']['mrr']:.4f} | "
          f"Hit@1 = {res_b['stage3_unconstrained']['hit@1'] * 100:.1f}% | "
          f"Hit@25 = {res_b['stage3_unconstrained'].get('hit@25', 0) * 100:.1f}%")
    print(f"  D. Stage 4 Hybrid (Physics+GNN): MRR@25 = {res_b['stage4_hybrid']['mrr']:.4f} | "
          f"Hit@1 = {res_b['stage4_hybrid']['hit@1'] * 100:.1f}% | "
          f"Hit@25 = {res_b['stage4_hybrid'].get('hit@25', 0) * 100:.1f}%")

    # Save metrics
    metrics_path = exp_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[stage4] all metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
