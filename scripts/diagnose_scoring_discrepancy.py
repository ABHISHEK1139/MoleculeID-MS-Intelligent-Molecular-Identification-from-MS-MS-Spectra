"""Diagnostic script to compare Local Benchmark Scoring vs Kaggle Submission Scoring.

Identifies every difference between:
1. Local Benchmark (evaluate_stage15_vs_stage5.py -> 0.4389 MRR)
2. Kaggle Submission (submission_script.py -> 0.144 MRR)

Runs side-by-side on the exact same 200 benchmark queries.
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

from src.core.canonical_benchmark import CanonicalBenchmark
from src.data.spectrum_dataset import spectrum_to_coarse_bins
from src.models.molecule_encoder import MoleculeGNN
from src.models.reranker import CrossModalReranker
from src.models.spectrum_encoder import SpectrumEncoder


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    bm = CanonicalBenchmark(subset_size=10000, n_benchmark_queries=200, split_seed=42, benchmark_seed=123)
    queries = bm.benchmark_queries
    cand_db = bm.cand_db
    cand_mols = cand_db.valid_mols
    mol_to_idx = bm.mol_to_idx

    iso_indices = [i for i, q in enumerate(queries) if q.is_isomer_query]

    # Models
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

    cand_graphs = [cand_db.mol_graphs[m] for m in cand_mols]
    z_mols_list = []
    with torch.no_grad():
        for i in range(0, len(cand_graphs), 256):
            bg = Batch.from_data_list(cand_graphs[i:i + 256]).to(device)
            z_mols_list.append(mol_encoder(bg))
    cand_embs = torch.cat(z_mols_list, dim=0)

    train_spectra_tensors = []
    train_mol_ids = []
    for s_info, mol_id in bm.train_ds.samples:
        binned = spectrum_to_coarse_bins(s_info["mz"], s_info["intensity"])
        train_spectra_tensors.append(torch.from_numpy(binned))
        train_mol_ids.append(mol_id)
    lib_tensors = F.normalize(torch.stack(train_spectra_tensors, dim=0).to(device), dim=-1)

    query_specs = torch.stack([q.spec_tensor for q in queries], dim=0).to(device)
    with torch.no_grad():
        z_queries = spec_encoder(query_specs)

    query_coarse = F.normalize(
        torch.stack([q.spec_tensor[:1480] for q in queries], dim=0).to(device),
        dim=-1,
    )
    cos_matrix = torch.mm(query_coarse, lib_tensors.T).cpu().numpy()

    # We evaluate 4 scoring schemes on the exact same 200 benchmark queries:
    # Scheme 1: evaluate_stage15_vs_stage5.py (Original Local Benchmark -> gave 0.4389)
    # Scheme 2: submission_script.py EXACT (Kaggle old: tau=0.45, W=2.00, +4.50*cos^2)
    # Scheme 3: submission_script.py with tau=0.90 (Calibrated override)
    # Scheme 4: submission_script.py with tau=1.01 (Pure Neural, no false analog override)

    schemes = {
        "1. Original Local (tau=0.70/0.75, 0.5*s_mass+0.5*p)": [],
        "2. Old Kaggle Submission (tau=0.45 query route, +4.5*cos^2)": [],
        "3. Calibrated Router (tau=0.90 candidate gate)": [],
        "4. Pure Stage 5 Neural (No library override)": [],
    }
    iso_schemes = {k: [] for k in schemes}

    RERANKER_A = 4.2645
    RERANKER_B = -1.0935

    for i, q in enumerate(queries):
        true_mol = q.true_mol
        matches = q.matches
        if len(matches) == 0:
            for k in schemes:
                schemes[k].append(0.0)
                if q.is_isomer_query:
                    iso_schemes[k].append(0.0)
            continue

        matched_mols = [m.mol for m in matches]
        matched_indices = [mol_to_idx[m.mol] for m in matches]
        best_lib_idx = int(np.argmax(cos_matrix[i]))
        max_cos = float(cos_matrix[i, best_lib_idx])
        best_lib_mol = train_mol_ids[best_lib_idx]

        ppm_errors = np.array([m.ppm_error for m in matches], dtype=np.float32)
        tiers = np.array([m.tier for m in matches], dtype=np.int32)
        tier_weights = np.where(tiers == 1, 1.0, 0.50).astype(np.float32)
        s_mass = np.exp(-ppm_errors / 10.0) * tier_weights

        top_cand_k = min(len(matches), 50)
        initial_order = np.argsort(-s_mass)[:top_cand_k]
        sub_matches = [matches[k] for k in initial_order]
        sub_indices = [matched_indices[k] for k in initial_order]
        sub_mols = [m.mol for m in sub_matches]

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

        # Scheme 1: evaluate_stage15_vs_stage5.py
        s1_fused = 0.50 * s_mass[initial_order] + 0.50 * calib_prob
        if max_cos >= 0.70 and best_lib_mol in sub_mols:
            s1_fused[sub_mols.index(best_lib_mol)] += 3.0
        r1_order = np.argsort(-s1_fused)
        r1_ranked = [sub_mols[k] for k in r1_order][:25]
        if max_cos >= 0.75 and best_lib_mol not in r1_ranked:
            r1_ranked = [best_lib_mol] + r1_ranked[:24]

        # Scheme 2: Old Kaggle Submission (tau=0.45 query route)
        s2_fused = 1.50 * s_mass[initial_order] + 2.00 * calib_prob
        if max_cos >= 0.45:  # route = "library"
            if best_lib_mol in sub_mols:
                s2_fused[sub_mols.index(best_lib_mol)] += 4.50 * (max_cos ** 2)
            else:
                s2_fused = np.append(s2_fused, 4.50 * (max_cos ** 2))
                sub_mols_ext = sub_mols + [best_lib_mol]
        else:
            sub_mols_ext = sub_mols
        r2_order = np.argsort(-s2_fused)
        r2_ranked = [(sub_mols_ext[k] if max_cos >= 0.45 and best_lib_mol not in sub_mols else sub_mols[k]) for k in r2_order][:25]
        if max_cos >= 0.45 and best_lib_mol not in r2_ranked:
            r2_ranked = [best_lib_mol] + r2_ranked[:24]

        # Scheme 3: Calibrated Router (tau=0.90 candidate gate)
        s3_fused = 1.50 * s_mass[initial_order] + 2.00 * calib_prob
        if max_cos >= 0.90 and best_lib_mol in sub_mols:
            s3_fused[sub_mols.index(best_lib_mol)] += 4.50 * (max_cos ** 2)
        elif max_cos >= 0.35 and best_lib_mol in sub_mols:
            s3_fused[sub_mols.index(best_lib_mol)] += 0.75 * max_cos
        r3_order = np.argsort(-s3_fused)
        r3_ranked = [sub_mols[k] for k in r3_order][:25]
        if max_cos >= 0.90 and best_lib_mol not in r3_ranked:
            r3_ranked = [best_lib_mol] + r3_ranked[:24]

        # Scheme 4: Pure Stage 5 Neural (tau=1.01, no override)
        s4_fused = 1.50 * s_mass[initial_order] + 2.00 * calib_prob
        r4_order = np.argsort(-s4_fused)
        r4_ranked = [sub_mols[k] for k in r4_order][:25]

        for s_name, r_list in [
            ("1. Original Local (tau=0.70/0.75, 0.5*s_mass+0.5*p)", r1_ranked),
            ("2. Old Kaggle Submission (tau=0.45 query route, +4.5*cos^2)", r2_ranked),
            ("3. Calibrated Router (tau=0.90 candidate gate)", r3_ranked),
            ("4. Pure Stage 5 Neural (No library override)", r4_ranked),
        ]:
            r = r_list.index(true_mol) + 1 if true_mol in r_list else 0
            rr = 1.0 / r if r > 0 else 0.0
            schemes[s_name].append(rr)
            if q.is_isomer_query:
                iso_schemes[s_name].append(rr)

    print("\n" + "=" * 80)
    print("  EXACT HEAD-TO-HEAD ON FROZEN 200 BENCHMARK QUERIES (Step 3 Diagnosis)")
    print("=" * 80)
    print(f"{'Scoring Scheme':<55} | {'All 200 MRR':<12} | {'Isomer MRR':<12}")
    print("-" * 80)
    for k in schemes:
        all_mrr = float(np.mean(schemes[k]))
        iso_mrr = float(np.mean(iso_schemes[k]))
        print(f"{k:<55} | {all_mrr:<12.4f} | {iso_mrr:<12.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
