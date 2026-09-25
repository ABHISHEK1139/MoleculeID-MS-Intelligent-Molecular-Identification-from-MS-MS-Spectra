"""Stage 5 Evaluation: Hard-Negative Isomer Reranking Benchmarks.

Runs two rigorous scientific evaluations:
1. Benchmark 1: Exact Constitutional Isomer Discrimination Challenge.
   Tests directly whether the learned fragmentation signal distinguishes exact
   same-formula constitutional isomers (|Δppm| = 0.00) where physics is blind.
   Compares: Random (50.0%), Physics (50.0%), Stage 3 GNN, Stage 5 Reranker.

2. Benchmark 2: Full 10,000-Candidate Zero-Reference Protocol C Retrieval.
   Tests held-out queries against the full 10,000 candidate catalog comparing:
   Uniform Random, Physics Mass Ordering, Stage 3 GNN, Stage 4 Hybrid, and Stage 5 Reranker.

Usage:
    python scripts/evaluate_stage5.py --config configs/stage05/exp5a.yaml --exp-name exp5a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
import yaml

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
from src.models.reranker import CrossModalReranker
from src.data.cross_modal_dataset import create_cross_modal_datasets
from src.search.candidate_generator import CandidateDatabase
from src.data.hard_negative_dataset import HardNegativeIndex


@torch.no_grad()
def run_isomer_challenge(
    cand_db: CandidateDatabase,
    val_dataset: Any,
    spec_encoder: nn.Module,
    mol_encoder_s3: nn.Module,
    mol_encoder_s5: nn.Module,
    reranker: nn.Module,
    device: torch.device,
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark 1: Exact Constitutional Isomer Discrimination (|Δppm| = 0.00)."""
    print("\n--- Benchmark 1: Exact Isomer Discrimination Challenge ---", flush=True)
    spec_encoder.eval()
    mol_encoder_s3.eval()
    mol_encoder_s5.eval()
    reranker.eval()

    # Build formula index over candidate database
    cand_index = HardNegativeIndex(cand_db.valid_mols, cand_db.mol_smiles)

    # Encode all candidate molecules under Stage 3 and Stage 5
    cand_mols = cand_db.valid_mols
    cand_graphs = [cand_db.mol_graphs[m] for m in cand_mols]
    batch_size = 64

    z_mols_s3_list = []
    z_mols_s5_list = []
    for i in range(0, len(cand_graphs), batch_size):
        bg = Batch.from_data_list(cand_graphs[i:i + batch_size]).to(device)
        z_mols_s3_list.append(mol_encoder_s3(bg).cpu())
        z_mols_s5_list.append(mol_encoder_s5(bg).cpu())

    z_mols_s3 = torch.cat(z_mols_s3_list, dim=0)  # (M, 256)
    z_mols_s5 = torch.cat(z_mols_s5_list, dim=0)  # (M, 256)
    mol_to_idx = {m: i for i, m in enumerate(cand_mols)}

    rng = np.random.default_rng(seed)

    s3_correct = 0
    s5_correct = 0
    total_pairs = 0

    s3_margins = []
    s5_margins = []

    for q_idx in range(len(val_dataset)):
        spec_tensor, _, true_mol = val_dataset[q_idx]
        if true_mol not in cand_index.formula_map:
            continue

        true_formula = cand_index.formula_map[true_mol]
        if not true_formula:
            continue

        # Find all constitutional isomers of true_mol in candidate database
        isomers = [m for m in cand_index.formula_to_mols.get(true_formula, []) if m != true_mol]
        if len(isomers) == 0:
            continue

        # Ground truth spectrum embedding
        spec_t = spec_tensor.unsqueeze(0).to(device)
        z_spec = spec_encoder(spec_t)  # (1, 256)

        idx_pos = mol_to_idx[true_mol]
        z_pos_s3 = z_mols_s3[idx_pos:idx_pos + 1].to(device)
        z_pos_s5 = z_mols_s5[idx_pos:idx_pos + 1].to(device)

        # Physics features for exact isomer pair: both match formula and exact mass!
        phys_pos = torch.tensor([[0.0, 1.0, 0.3, 1.0]], device=device)
        phys_neg = torch.tensor([[0.0, 1.0, 0.3, 1.0]], device=device)

        s_pos_s5 = reranker(z_spec, z_pos_s5, phys_pos).item()
        sim_pos_s3 = torch.sum(z_spec * z_pos_s3).item()

        for iso_mol in isomers:
            idx_neg = mol_to_idx[iso_mol]
            z_neg_s3 = z_mols_s3[idx_neg:idx_neg + 1].to(device)
            z_neg_s5 = z_mols_s5[idx_neg:idx_neg + 1].to(device)

            sim_neg_s3 = torch.sum(z_spec * z_neg_s3).item()
            s_neg_s5 = reranker(z_spec, z_neg_s5, phys_neg).item()

            if sim_pos_s3 > sim_neg_s3:
                s3_correct += 1
            if s_pos_s5 > s_neg_s5:
                s5_correct += 1

            s3_margins.append(sim_pos_s3 - sim_neg_s3)
            s5_margins.append(s_pos_s5 - s_neg_s5)
            total_pairs += 1

    acc_random = 0.50
    acc_physics = 0.50  # Physics mass error |Δppm| = 0.00 for both, exactly tied
    acc_s3 = s3_correct / max(total_pairs, 1)
    acc_s5 = s5_correct / max(total_pairs, 1)

    print(f"Total Exact Isomer Pairs Evaluated: {total_pairs}", flush=True)
    print(f"Uniform Random Baseline:            {acc_random * 100:.2f}%", flush=True)
    print(f"Physics Mass Ordering (Tied):       {acc_physics * 100:.2f}%", flush=True)
    print(f"Stage 3 GNN (Unconstrained Cosine): {acc_s3 * 100:.2f}% (mean margin: {np.mean(s3_margins):.4f})", flush=True)
    print(f"Stage 5 Reranker (Fragmentation):   {acc_s5 * 100:.2f}% (mean margin: {np.mean(s5_margins):.4f})", flush=True)

    return {
        "total_isomer_pairs": total_pairs,
        "accuracy_random": round(acc_random, 4),
        "accuracy_physics": round(acc_physics, 4),
        "accuracy_stage3_gnn": round(acc_s3, 4),
        "accuracy_stage5_reranker": round(acc_s5, 4),
        "mean_margin_stage3": round(float(np.mean(s3_margins)), 4),
        "mean_margin_stage5": round(float(np.mean(s5_margins)), 4),
    }


@torch.no_grad()
def run_retrieval_benchmark(
    cand_db: CandidateDatabase,
    val_dataset: Any,
    spec_encoder: nn.Module,
    mol_encoder_s3: nn.Module,
    mol_encoder_s5: nn.Module,
    reranker: nn.Module,
    device: torch.device,
    n_queries: int = 200,
    top_k: int = 25,
    ppm_primary: float = 20.0,
    ppm_fallback: float = 50.0,
    include_isotope: bool = True,
    seed: int = 123,
) -> dict[str, Any]:
    """Benchmark 2: Full 10,000 Candidate Zero-Reference Protocol C Retrieval."""
    print(f"\n--- Benchmark 2: Full 10,000 Candidate Retrieval ({n_queries} queries) ---", flush=True)
    spec_encoder.eval()
    mol_encoder_s3.eval()
    mol_encoder_s5.eval()
    reranker.eval()

    cand_mols = cand_db.valid_mols
    cand_graphs = [cand_db.mol_graphs[m] for m in cand_mols]
    batch_size = 64

    z_mols_s3_list = []
    z_mols_s5_list = []
    for i in range(0, len(cand_graphs), batch_size):
        bg = Batch.from_data_list(cand_graphs[i:i + batch_size]).to(device)
        z_mols_s3_list.append(mol_encoder_s3(bg).cpu())
        z_mols_s5_list.append(mol_encoder_s5(bg).cpu())

    cand_embs_s3 = torch.cat(z_mols_s3_list, dim=0).to(device)  # (M, 256)
    cand_embs_s5 = torch.cat(z_mols_s5_list, dim=0).to(device)  # (M, 256)
    mol_to_idx = {m: i for i, m in enumerate(cand_mols)}

    # Select queries
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

        if true_mol not in cand_db.mol_smiles:
            continue

        query_specs_list.append(spec_tensor)
        query_true_mols.append(true_mol)
        query_prec_mzs.append(prec_mz)
        query_adducts.append(adduct)
        query_target_masses.append(cand_db.raw_masses[mol_to_idx[true_mol]])

    query_specs = torch.stack(query_specs_list, dim=0).to(device)
    z_queries = spec_encoder(query_specs)  # (n_q, 256)

    # Raw cosine similarities for Stage 3
    sims_raw_s3 = (z_queries @ cand_embs_s3.T).cpu().numpy()  # (n_q, M)

    ranks_stage3_raw: list[int] = []
    ranks_physics_only: list[int] = []
    ranks_stage4_hybrid: list[int] = []
    ranks_stage5_reranker: list[int] = []

    for i, true_mol in enumerate(query_true_mols):
        target_idx = mol_to_idx[true_mol]
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
        matched_mols = {m.mol: m for m in matches}

        # 1. Stage 3 Unconstrained
        ranked_s3 = np.argsort(-sims_raw_s3[i])[:top_k]
        match_s3 = np.where(ranked_s3 == target_idx)[0]
        ranks_stage3_raw.append(int(match_s3[0] + 1) if len(match_s3) > 0 else 0)

        # 2. Physics Only
        ranked_phys_mols = [m.mol for m in matches][:top_k]
        if true_mol in ranked_phys_mols:
            ranks_physics_only.append(ranked_phys_mols.index(true_mol) + 1)
        else:
            ranks_physics_only.append(0)

        # 3. Stage 4 Hybrid
        score_stage4 = np.full(len(cand_mols), -1e9, dtype=np.float32)
        for m_id, match_obj in matched_mols.items():
            idx = mol_to_idx[m_id]
            score_stage4[idx] = sims_raw_s3[i, idx] * match_obj.weight

        ranked_s4 = np.argsort(-score_stage4)[:top_k]
        match_s4 = np.where(ranked_s4 == target_idx)[0]
        ranks_stage4_hybrid.append(int(match_s4[0] + 1) if len(match_s4) > 0 else 0)

        # 4. Stage 5 Reranker (Scores only the physics-pruned candidates)
        matched_indices = [mol_to_idx[m.mol] for m in matches]
        if len(matched_indices) > 0:
            sub_z_mols = cand_embs_s5[matched_indices]  # (K, 256)
            sub_z_spec = z_queries[i:i + 1]  # (1, 256)

            # Build physics vectors
            phys_list = []
            for m in matches:
                ppm_norm = min(m.ppm_error / 20.0, 3.0)
                tier_w = 1.0 if m.tier == 1 else 0.5
                prec_norm = prec_mz / 1000.0
                formula_m = 1.0 if m.ppm_error <= 5.0 else 0.0
                phys_list.append([ppm_norm, tier_w, prec_norm, formula_m])

            sub_phys = torch.tensor(phys_list, dtype=torch.float32, device=device)

            # Reranker scoring
            sub_scores = reranker(sub_z_spec, sub_z_mols, sub_phys).cpu().numpy()

            # Rank matched candidates
            sort_order = np.argsort(-sub_scores)
            ranked_s5_mols = [matches[k].mol for k in sort_order][:top_k]

            if true_mol in ranked_s5_mols:
                ranks_stage5_reranker.append(ranked_s5_mols.index(true_mol) + 1)
            else:
                ranks_stage5_reranker.append(0)
        else:
            ranks_stage5_reranker.append(0)

    # Identify which queries have exact same-formula isomer distractors in their pool
    cand_index = HardNegativeIndex(cand_db.valid_mols, cand_db.mol_smiles)
    query_has_isomers = []
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
        f_true = cand_index.formula_map.get(true_mol, "")
        has_iso = any(m.mol != true_mol and cand_index.formula_map.get(m.mol, "") == f_true for m in matches)
        query_has_isomers.append(has_iso)

    iso_indices = [i for i, h in enumerate(query_has_isomers) if h]
    uniq_indices = [i for i, h in enumerate(query_has_isomers) if not h]

    # 5. True Uniform Random Baseline (Averaged over 100 shuffle trials)
    random_trial_mrrs = []
    random_trial_hit1s = []
    random_trial_iso_mrrs = []
    random_trial_iso_hit1s = []

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
            cands = [m.mol for m in matches]
            trial_rng.shuffle(cands)
            cands = cands[:top_k]
            if true_mol in cands:
                trial_ranks.append(cands.index(true_mol) + 1)
            else:
                trial_ranks.append(0)
        m = summarize_ranks(trial_ranks, k=top_k)
        random_trial_mrrs.append(m["mrr"])
        random_trial_hit1s.append(m["hit@1"])

        if len(iso_indices) > 0:
            m_iso = summarize_ranks([trial_ranks[k_idx] for k_idx in iso_indices], k=top_k)
            random_trial_iso_mrrs.append(m_iso["mrr"])
            random_trial_iso_hit1s.append(m_iso["hit@1"])

    metrics_random = {
        "mrr": round(float(np.mean(random_trial_mrrs)), 4),
        "hit@1": round(float(np.mean(random_trial_hit1s)), 4),
    }
    metrics_s3 = summarize_ranks(ranks_stage3_raw, k=top_k)
    metrics_phys = summarize_ranks(ranks_physics_only, k=top_k)
    metrics_s4 = summarize_ranks(ranks_stage4_hybrid, k=top_k)
    metrics_s5 = summarize_ranks(ranks_stage5_reranker, k=top_k)

    print("\nBenchmark 2 Results (All 200 Queries):", flush=True)
    print(f"Uniform Random:           MRR@25 = {metrics_random['mrr']:.4f} | Hit@1 = {metrics_random['hit@1']*100:.2f}%", flush=True)
    print(f"Stage 3 GNN (Unpruned):   MRR@25 = {metrics_s3['mrr']:.4f} | Hit@1 = {metrics_s3['hit@1']*100:.2f}%", flush=True)
    print(f"Stage 4 Mass + S3 GNN:    MRR@25 = {metrics_s4['mrr']:.4f} | Hit@1 = {metrics_s4['hit@1']*100:.2f}%", flush=True)
    print(f"Physics Mass Ordering:    MRR@25 = {metrics_phys['mrr']:.4f} | Hit@1 = {metrics_phys['hit@1']*100:.2f}%", flush=True)
    print(f"Stage 5 Reranker (NEW):   MRR@25 = {metrics_s5['mrr']:.4f} | Hit@1 = {metrics_s5['hit@1']*100:.2f}%", flush=True)

    # Isomer subset metrics
    subset_metrics = {}
    if len(iso_indices) > 0:
        iso_s3 = summarize_ranks([ranks_stage3_raw[k_idx] for k_idx in iso_indices], k=top_k)
        iso_phys = summarize_ranks([ranks_physics_only[k_idx] for k_idx in iso_indices], k=top_k)
        iso_s4 = summarize_ranks([ranks_stage4_hybrid[k_idx] for k_idx in iso_indices], k=top_k)
        iso_s5 = summarize_ranks([ranks_stage5_reranker[k_idx] for k_idx in iso_indices], k=top_k)
        iso_rand = {
            "mrr": round(float(np.mean(random_trial_iso_mrrs)), 4),
            "hit@1": round(float(np.mean(random_trial_iso_hit1s)), 4),
        }

        print(f"\n--- Benchmark 2 SUBSET: Exact Same-Formula Multi-Isomer Queries ({len(iso_indices)}/{len(query_true_mols)} queries) ---", flush=True)
        print(f"Uniform Random:           MRR@25 = {iso_rand['mrr']:.4f} | Hit@1 = {iso_rand['hit@1']*100:.2f}%", flush=True)
        print(f"Physics Mass Ordering:    MRR@25 = {iso_phys['mrr']:.4f} | Hit@1 = {iso_phys['hit@1']*100:.2f}%", flush=True)
        print(f"Stage 4 Mass + S3 GNN:    MRR@25 = {iso_s4['mrr']:.4f} | Hit@1 = {iso_s4['hit@1']*100:.2f}%", flush=True)
        print(f"Stage 5 Reranker (NEW):   MRR@25 = {iso_s5['mrr']:.4f} | Hit@1 = {iso_s5['hit@1']*100:.2f}%", flush=True)

        subset_metrics = {
            "n_isomer_queries": len(iso_indices),
            "uniform_random": iso_rand,
            "stage3_gnn_unpruned": iso_s3,
            "physics_mass_ordering": iso_phys,
            "stage4_hybrid": iso_s4,
            "stage5_reranker": iso_s5,
        }

    return {
        "overall_all_queries": {
            "uniform_random": metrics_random,
            "stage3_gnn_unpruned": metrics_s3,
            "physics_mass_ordering": metrics_phys,
            "stage4_hybrid": metrics_s4,
            "stage5_reranker": metrics_s5,
        },
        "exact_isomer_subset": subset_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 5 Dual Benchmark Evaluation")
    parser.add_argument("--config", type=str, default="configs/stage05/exp5a.yaml")
    parser.add_argument("--exp-name", type=str, default="exp5a")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_dir = ARTIFACTS_DIR / "stage05" / args.exp_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 5 Dual Benchmark [{args.exp_name}] on {device} ===", flush=True)

    # 1. Load Data
    data_cfg = cfg["data"]
    base_train_ds, base_val_ds = create_cross_modal_datasets(
        train_path=data_cfg.get("train_path", TRAIN_PATH),
        subset_size=data_cfg.get("subset_size", 10000),
        val_frac=data_cfg.get("val_frac", 0.10),
        seed=data_cfg.get("seed", 42),
    )

    # 2. Build Candidate Database across all 10,000 molecules
    print("Building 10,000 candidate database...", flush=True)
    all_mols = list(base_train_ds.selected_mols) + list(base_val_ds.selected_mols)
    all_smiles = {**base_train_ds.mol_smiles, **base_val_ds.mol_smiles}
    all_graphs = {**base_train_ds.graph_cache, **base_val_ds.graph_cache}
    valid_all_mols = [m for m in all_mols if m in all_graphs]
    cand_db = CandidateDatabase(
        molecules=valid_all_mols,
        mol_smiles=all_smiles,
        mol_graphs=all_graphs,
    )

    # 3. Load Models
    m_cfg = cfg["models"]

    # Frozen Spectrum Encoder
    spec_encoder = SpectrumEncoder(embed_dim=m_cfg.get("embed_dim", 256)).to(device)
    spec_ckpt = m_cfg.get("spectrum_checkpoint", "artifacts/stage02/exp2a/checkpoints/best.pt")
    spec_data = torch.load(spec_ckpt, map_location=device, weights_only=False)
    spec_encoder.load_state_dict(spec_data.get("model_state_dict", spec_data))
    spec_encoder.eval()

    # Pretrained Stage 3 GNN (Baseline)
    mol_encoder_s3 = MoleculeGNN(embed_dim=m_cfg.get("embed_dim", 256)).to(device)
    gnn_s3_ckpt = m_cfg.get("gnn_checkpoint", "artifacts/stage03/exp3a/checkpoints/best.pt")
    gnn_s3_data = torch.load(gnn_s3_ckpt, map_location=device, weights_only=False)
    mol_encoder_s3.load_state_dict(gnn_s3_data.get("model_state_dict", gnn_s3_data))
    mol_encoder_s3.eval()

    # Trained Stage 5 Reranker + Fine-tuned GNN
    s5_ckpt_path = exp_dir / "checkpoints" / "best.pt"
    print(f"Loading Stage 5 checkpoint from: {s5_ckpt_path}", flush=True)
    s5_data = torch.load(s5_ckpt_path, map_location=device, weights_only=False)

    mol_encoder_s5 = MoleculeGNN(embed_dim=m_cfg.get("embed_dim", 256)).to(device)
    mol_encoder_s5.load_state_dict(s5_data["mol_encoder_state_dict"])
    mol_encoder_s5.eval()

    reranker = CrossModalReranker(
        embed_dim=m_cfg.get("embed_dim", 256),
        physics_dim=m_cfg.get("physics_dim", 4),
        hidden_dim=m_cfg.get("reranker_hidden_dim", 256),
    ).to(device)
    reranker.load_state_dict(s5_data["reranker_state_dict"])
    reranker.eval()

    # Run Benchmark 1: Exact Isomer Discrimination Challenge
    res_b1 = run_isomer_challenge(
        cand_db=cand_db,
        val_dataset=base_val_ds,
        spec_encoder=spec_encoder,
        mol_encoder_s3=mol_encoder_s3,
        mol_encoder_s5=mol_encoder_s5,
        reranker=reranker,
        device=device,
    )

    # Run Benchmark 2: Full 10,000 Candidate Zero-Reference Retrieval
    res_b2 = run_retrieval_benchmark(
        cand_db=cand_db,
        val_dataset=base_val_ds,
        spec_encoder=spec_encoder,
        mol_encoder_s3=mol_encoder_s3,
        mol_encoder_s5=mol_encoder_s5,
        reranker=reranker,
        device=device,
        n_queries=cfg["evaluation"].get("n_protocol_c_queries", 200),
        top_k=cfg["evaluation"].get("top_k", 25),
    )

    full_results = {
        "benchmark1_exact_isomer_challenge": res_b1,
        "benchmark2_10k_retrieval": res_b2,
    }

    metrics_file = exp_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nSaved Stage 5 metrics to: {metrics_file}", flush=True)


if __name__ == "__main__":
    main()
