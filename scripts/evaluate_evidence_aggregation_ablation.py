"""Ablation and Evaluation Suite for Evidence Aggregation Scorer.

Runs the exact sequence requested:
  A = 839k baseline
  B = + external cosine
  C = + cosine + peaks
  D = + cosine + peaks + CE
  E = + cosine + peaks + CE + multiplicity
  F = E + Stage 5

Workflow:
1. Load 800 Tuning Queries (from CanonicalBenchmark) + 400 Benchmark Queries (200 novel + 200 known).
2. Precompute candidate match features against Baseline 839k and Unified External (MoNA + GNPS + Baseline).
3. Precompute Stage 5 neural reranker scores.
4. Tune weights (w1, w2, w3, w4, gamma) strictly on the 800 Tuning Queries.
5. Evaluate all configurations A-F on the frozen Benchmark Queries (400 total: 200 novel + 200 known).
6. Save comprehensive ablation results artifact.
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
import torch.nn.functional as F
from torch_geometric.data import Batch

from src.core.canonical_benchmark import CanonicalBenchmark, CanonicalQuery
from src.core.evidence_scorer import EvidenceScorer, EvidenceWeights
from src.models.reranker import CrossModalReranker
from src.models.molecule_encoder import MoleculeGNN
from src.models.spectrum_encoder import SpectrumEncoder
from scripts.evaluate_external_library_diagnostic import (
    CompactSpectralLibrary,
    fast_mutual_cosine,
)


# ── Feature Extraction for a Single Query ──────────────────────────────────
def extract_query_candidate_data(
    q: CanonicalQuery,
    spec_info: dict[str, Any],
    cand_db: Any,
    lib_base: CompactSpectralLibrary,
    lib_ext: CompactSpectralLibrary,
) -> dict[str, Any]:
    """Extract candidate features against both libraries for a query."""
    target_mass = q.target_mass
    prec_mz = q.precursor_mz
    q_ce = spec_info.get("ce", float("nan"))
    q_mzs = np.asarray(spec_info["mz"], dtype=np.float32)
    q_ints = np.asarray(spec_info["intensity"], dtype=np.float32)
    true_smi = cand_db.mol_smiles.get(q.true_mol, "")

    # Candidates in mass window
    cand_records = []
    for m in q.matches:
        smi = cand_db.mol_smiles.get(m.mol, "")
        if not smi:
            continue
        tier_w = 1.0 if m.tier == 1 else 0.50
        s_mass = 1.50 * np.exp(-m.ppm_error / 10.0) * tier_w
        cand_records.append({
            "mol_id": m.mol,
            "smi": smi,
            "ppm_error": m.ppm_error,
            "tier": m.tier,
            "s_mass": s_mass,
            "is_true": (smi == true_smi),
        })

    cand_smis = {c["smi"] for c in cand_records}

    # 1. Search Baseline 839k Library
    l_idx, r_idx = lib_base.query_window(target_mass, ppm=20.0)
    base_best_cos: dict[str, float] = {}
    if r_idx > l_idx:
        for ri in range(l_idx, r_idx):
            ref_smi = lib_base.smiles[ri]
            if ref_smi not in cand_smis:
                continue
            delta = prec_mz - lib_base.precursor_mzs[ri]
            cos_sim, _ = fast_mutual_cosine(
                q_mzs, q_ints, lib_base.mzs_list[ri], lib_base.intens_list[ri], delta=delta
            )
            if cos_sim > base_best_cos.get(ref_smi, 0.0):
                base_best_cos[ref_smi] = cos_sim

    # 2. Search External Library (Unified: MoNA + GNPS + Baseline)
    l_idx, r_idx = lib_ext.query_window(target_mass, ppm=20.0)
    ext_hits: dict[str, list[dict[str, Any]]] = {}
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
                if ref_smi not in ext_hits:
                    ext_hits[ref_smi] = []
                ext_hits[ref_smi].append({
                    "cos": cos_sim,
                    "n_peaks": n_peaks,
                    "ce_diff": ce_diff,
                    "n_supporting": int(lib_ext.n_supporting[ri]),
                    "source_count": int(lib_ext.source_counts[ri]),
                })

    # Compile feature vectors for all candidates
    for c in cand_records:
        smi = c["smi"]
        c["base_cos"] = base_best_cos.get(smi, 0.0)

        # External features
        if smi in ext_hits:
            hits = ext_hits[smi]
            hits.sort(key=lambda x: x["cos"], reverse=True)
            top = hits[0]
            c["ext_features"] = EvidenceScorer.extract_candidate_features(
                best_cos=top["cos"],
                matched_peaks=top["n_peaks"],
                ce_diff=top["ce_diff"],
                n_supporting=top["n_supporting"],
                source_count=top["source_count"],
            )
            c["ext_best_cos"] = top["cos"]
        else:
            c["ext_features"] = {
                "similarity": 0.0,
                "peaks": 0.0,
                "ce": 0.0,
                "multiplicity": 0.0,
            }
            c["ext_best_cos"] = 0.0

    return {
        "true_smi": true_smi,
        "is_isomer": q.is_isomer_query,
        "precursor_mz": prec_mz,
        "candidates": cand_records,
    }


# ── Batch Processor for Multiprocessing ────────────────────────────────────
def process_query_batch(args: tuple[list[tuple[CanonicalQuery, dict]], Any, CompactSpectralLibrary, CompactSpectralLibrary]) -> list[dict]:
    batch, cand_db, lib_b, lib_e = args
    results = []
    for q, spec_info in batch:
        results.append(extract_query_candidate_data(q, spec_info, cand_db, lib_b, lib_e))
    return results


# ── Candidate Ranking & Evaluation Functions ────────────────────────────────
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
    num_cpus = os.cpu_count() or 4
    n_workers = min(12, num_cpus)

    print("=" * 85)
    print("  EVIDENCE AGGREGATION SCORER: STEP-BY-STEP ABLATION & BENCHMARK")
    print(f"  Running across {n_workers} CPU workers on {num_cpus}-core system")
    print("=" * 85, flush=True)

    # 1. Load Canonical Benchmark
    print("\n[1/6] Initializing Canonical Benchmark (800 Tuning + 400 Frozen Benchmark)...", flush=True)
    bm = CanonicalBenchmark(subset_size=10000, n_benchmark_queries=200, split_seed=42, benchmark_seed=123)

    # 800 Tuning Queries
    tuning_queries = bm.tuning_queries
    tuning_specs = [bm.val_ds.samples[q.query_id][0] for q in tuning_queries]

    # 200 Novel Benchmark Queries
    novel_queries = bm.benchmark_queries
    novel_specs = [bm.val_ds.samples[q.query_id][0] for q in novel_queries]

    # 200 Known Benchmark Queries
    train_samples_by_mol: dict[str, list[int]] = {}
    for sample_idx, (_, mol_id) in enumerate(bm.train_ds.samples):
        train_samples_by_mol.setdefault(mol_id, []).append(sample_idx)

    train_unique_mols = sorted(list(train_samples_by_mol.keys()))
    rng = np.random.default_rng(999)
    known_mol_picks = rng.choice(train_unique_mols, size=200, replace=False)

    known_queries = []
    known_specs = []
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
        known_queries.append(
            CanonicalQuery(
                query_id=int(s_idx),
                true_mol=true_mol,
                target_mass=t_mass,
                precursor_mz=prec_mz,
                adduct=adduct,
                spec_tensor=None,
                matches=matches,
                is_isomer_query=has_isomers,
                true_formula=f_true,
            )
        )
        known_specs.append(spec_info)

    benchmark_queries_all = novel_queries + known_queries
    benchmark_specs_all = novel_specs + known_specs

    print(f"Loaded {len(tuning_queries)} Tuning queries ({len(bm.tuning_isomer_indices)} isomers).")
    print(f"Loaded {len(benchmark_queries_all)} Frozen Benchmark queries (200 novel + 200 known).")

    # 2. Load Libraries
    print("\n[2/6] Loading Baseline 839k and Unified External Libraries...", flush=True)
    ref_path = ROOT / "kaggle_dataset" / "reference_library_multice.parquet"
    df_ref = pd.read_parquet(ref_path, columns=["normalized_smiles", "neutral_mass", "precursor_mz", "collision_energy", "ms2_mzs", "ms2_intensities"])

    ext_path = ROOT / "artifacts" / "external" / "external_spectra.parquet"
    df_ext = pd.read_parquet(ext_path, columns=[
        "canonical_smiles", "neutral_mass", "precursor_mz", "collision_energy",
        "peaks_mz", "peaks_intensity", "source_library", "n_supporting_spectra", "source_count"
    ])

    lib_base = CompactSpectralLibrary(
        smiles=df_ref["normalized_smiles"].to_numpy(),
        neutral_masses=df_ref["neutral_mass"].to_numpy(dtype=np.float64),
        precursor_mzs=df_ref["precursor_mz"].to_numpy(dtype=np.float64),
        collision_energies=df_ref["collision_energy"].to_numpy(dtype=np.float32),
        mzs_list=[np.asarray(x, dtype=np.float32) for x in df_ref["ms2_mzs"]],
        intens_list=[np.asarray(x, dtype=np.float32) for x in df_ref["ms2_intensities"]],
    )

    d_smiles = np.concatenate([df_ref["normalized_smiles"].to_numpy(), df_ext["canonical_smiles"].to_numpy()])
    d_masses = np.concatenate([df_ref["neutral_mass"].to_numpy(dtype=np.float64), df_ext["neutral_mass"].to_numpy(dtype=np.float64)])
    d_precs = np.concatenate([df_ref["precursor_mz"].to_numpy(dtype=np.float64), df_ext["precursor_mz"].to_numpy(dtype=np.float64)])
    d_ces = np.concatenate([df_ref["collision_energy"].to_numpy(dtype=np.float32), df_ext["collision_energy"].to_numpy(dtype=np.float32)])
    d_mzs = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_mzs"]] + [np.asarray(x, dtype=np.float32) for x in df_ext["peaks_mz"]]
    d_ints = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_intensities"]] + [np.asarray(x, dtype=np.float32) for x in df_ext["peaks_intensity"]]
    d_sup = np.concatenate([np.ones(len(df_ref), dtype=np.int16), df_ext["n_supporting_spectra"].to_numpy(dtype=np.int16)])
    d_src = np.concatenate([np.ones(len(df_ref), dtype=np.int8), df_ext["source_count"].to_numpy(dtype=np.int8)])

    lib_ext = CompactSpectralLibrary(d_smiles, d_masses, d_precs, d_ces, d_mzs, d_ints, d_sup, d_src)
    print(f"Libraries indexed: Baseline={len(lib_base.neutral_masses):,} | Unified External={len(lib_ext.neutral_masses):,}")

    # 3. Extract Features for Tuning and Benchmark Queries
    print("\n[3/6] Precomputing spectral match features in parallel...", flush=True)

    def extract_dataset_records(queries: list[CanonicalQuery], specs: list[dict]) -> list[dict]:
        pairs = list(zip(queries, specs))
        chunk_size = max(1, len(pairs) // n_workers)
        batches = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        worker_args = [(b, bm.cand_db, lib_base, lib_ext) for b in batches]

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(process_query_batch, arg) for arg in worker_args]
            for f in concurrent.futures.as_completed(futures):
                results.extend(f.result())
        return results

    t_feat = time.time()
    tuning_records = extract_dataset_records(tuning_queries, tuning_specs)
    benchmark_records = extract_dataset_records(benchmark_queries_all, benchmark_specs_all)
    print(f"Extracted features for 800 tuning + 400 benchmark queries in {time.time()-t_feat:.1f}s.")

    # 4. Load Stage 5 Models and Precompute Neural Predictions
    print("\n[4/6] Precomputing Stage 5 neural reranker scores...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Molecule GNN
    s5_data = torch.load("artifacts/stage05/exp5a/checkpoints/best.pt", map_location=device, weights_only=False)
    mol_encoder = MoleculeGNN(embed_dim=256).to(device)
    mol_encoder.load_state_dict(s5_data["mol_encoder_state_dict"])
    mol_encoder.eval()

    # Encode all 10,000 candidate graphs in catalog
    cand_mols = bm.cand_db.valid_mols
    cand_graphs = [bm.cand_db.mol_graphs[m] for m in cand_mols]
    z_mols_list = []
    with torch.no_grad():
        for i in range(0, len(cand_graphs), 256):
            bg = Batch.from_data_list(cand_graphs[i:i + 256]).to(device)
            z_mols_list.append(mol_encoder(bg))
    cand_embs = torch.cat(z_mols_list, dim=0)

    # Spectrum Encoder
    spec_encoder = SpectrumEncoder(embed_dim=256).to(device)
    spec_data = torch.load("artifacts/stage02/exp2a/checkpoints/best.pt", map_location=device, weights_only=False)
    spec_encoder.load_state_dict(spec_data.get("model_state_dict", spec_data))
    spec_encoder.eval()

    # CrossModalReranker
    reranker = CrossModalReranker(embed_dim=256, physics_dim=4, hidden_dim=256).to(device)
    reranker.load_state_dict(s5_data["reranker_state_dict"])
    reranker.eval()

    def attach_stage5_scores(records: list[dict], query_objs: list[CanonicalQuery]):
        from src.data.spectrum_dataset import spectrum_to_coarse_bins
        for idx, (rec, q) in enumerate(zip(records, query_objs)):
            spec_tensor = q.spec_tensor
            if spec_tensor is None:
                # Build tensor from spec info if needed
                binned = spectrum_to_coarse_bins(
                    np.asarray(tuning_specs[0]["mz"]),
                    np.asarray(tuning_specs[0]["intensity"]),
                )
                continue

            with torch.no_grad():
                z_spec = spec_encoder(spec_tensor.unsqueeze(0).to(device))

            cands = rec["candidates"]
            if not cands:
                continue

            cand_indices = [bm.mol_to_idx[c["mol_id"]] for c in cands]
            sub_z_mols = cand_embs[cand_indices]

            prec_norm = q.precursor_mz / 1000.0
            phys_list = [
                [min(c["ppm_error"] / 20.0, 3.0), 1.0 if c["tier"] == 1 else 0.5, prec_norm, 1.0 if c["ppm_error"] <= 5.0 else 0.0]
                for c in cands
            ]
            sub_phys = torch.tensor(phys_list, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = reranker(z_spec, sub_z_mols, sub_phys).cpu().numpy()

            calib_prob = 1.0 / (1.0 + np.exp(-np.clip(4.2645 * logits - 1.0935, -15.0, 15.0)))
            for c, p in zip(cands, calib_prob):
                c["stage5_prob"] = float(p)

    # Attach stage5 scores to queries that have spec_tensor
    # Note: tuning_queries and novel_queries in CanonicalBenchmark have spec_tensor attached
    attach_stage5_scores(tuning_records, tuning_queries)
    attach_stage5_scores(benchmark_records[:200], novel_queries)

    # For known queries (the second half of benchmark), build spec_tensor from dataset
    for i, s_idx in enumerate([q.query_id for q in known_queries]):
        s_tensor = bm.train_ds[s_idx][0]
        known_queries[i].spec_tensor = s_tensor
    attach_stage5_scores(benchmark_records[200:], known_queries)

    print("Stage 5 scores precomputed successfully.")

    # 5. Ablation Study: Local Tuning Set (800 Queries)
    print("\n[5/6] Running Step-by-Step Ablation on 800 Tuning Queries...", flush=True)

    # Define Uniform Boost Function
    def apply_boost(raw_evidence: float) -> float:
        if raw_evidence >= 0.45:
            return 4.50 * (raw_evidence ** 2)
        elif raw_evidence >= 0.20:
            return 1.00 * raw_evidence
        return 0.0

    # Define Scoring Functions for Ablation A - E
    def score_a_baseline(c: dict) -> float:
        """A = 839k baseline: mass score + baseline cosine boost."""
        return c["s_mass"] + apply_boost(c["base_cos"])

    def score_b_ext_cosine(c: dict) -> float:
        """B = + External Cosine only."""
        return c["s_mass"] + apply_boost(c["ext_features"]["similarity"])

    # Sweep weights on Tuning Set for C, D, E
    print("  Tuning additive component weights on 800 tuning queries...")
    best_weights_c = None
    best_mrr_c = -1.0
    for w_p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        w = EvidenceWeights(w_similarity=1.0, w_peaks=w_p).normalized()
        def score_c_test(c: dict, w_obj=w) -> float:
            f = c["ext_features"]
            if f["similarity"] < 0.10:
                return c["s_mass"]
            raw = w_obj.w_similarity * f["similarity"] + w_obj.w_peaks * f["peaks"]
            return c["s_mass"] + apply_boost(raw)
        mrr = evaluate_ranking(tuning_records, score_c_test)["mrr25"]
        if mrr > best_mrr_c:
            best_mrr_c = mrr
            best_weights_c = w

    print(f"  Best Weights C (+peaks): w_sim={best_weights_c.w_similarity:.3f}, w_peaks={best_weights_c.w_peaks:.3f} -> Tuning MRR={best_mrr_c:.4f}")

    best_weights_d = None
    best_mrr_d = -1.0
    for w_ce in [0.03, 0.05, 0.08, 0.10, 0.15]:
        w = EvidenceWeights(w_similarity=1.0, w_peaks=best_weights_c.w_peaks / best_weights_c.w_similarity, w_ce=w_ce).normalized()
        def score_d_test(c: dict, w_obj=w) -> float:
            f = c["ext_features"]
            if f["similarity"] < 0.10:
                return c["s_mass"]
            raw = w_obj.w_similarity * f["similarity"] + w_obj.w_peaks * f["peaks"] + w_obj.w_ce * f["ce"]
            return c["s_mass"] + apply_boost(raw)
        mrr = evaluate_ranking(tuning_records, score_d_test)["mrr25"]
        if mrr > best_mrr_d:
            best_mrr_d = mrr
            best_weights_d = w

    print(f"  Best Weights D (+peaks+CE): w_sim={best_weights_d.w_similarity:.3f}, w_peaks={best_weights_d.w_peaks:.3f}, w_ce={best_weights_d.w_ce:.3f} -> Tuning MRR={best_mrr_d:.4f}")

    best_weights_e = None
    best_mrr_e = -1.0
    for w_m in [0.03, 0.05, 0.08, 0.10, 0.15]:
        w = EvidenceWeights(
            w_similarity=1.0,
            w_peaks=best_weights_d.w_peaks / best_weights_d.w_similarity,
            w_ce=best_weights_d.w_ce / best_weights_d.w_similarity,
            w_multiplicity=w_m,
        ).normalized()
        def score_e_test(c: dict, w_obj=w) -> float:
            f = c["ext_features"]
            if f["similarity"] < 0.10:
                return c["s_mass"]
            raw = (
                w_obj.w_similarity * f["similarity"]
                + w_obj.w_peaks * f["peaks"]
                + w_obj.w_ce * f["ce"]
                + w_obj.w_multiplicity * f["multiplicity"]
            )
            return c["s_mass"] + apply_boost(raw)
        mrr = evaluate_ranking(tuning_records, score_e_test)["mrr25"]
        if mrr > best_mrr_e:
            best_mrr_e = mrr
            best_weights_e = w

    print(f"  Best Weights E (+peaks+CE+multi): w_sim={best_weights_e.w_similarity:.3f}, w_peaks={best_weights_e.w_peaks:.3f}, w_ce={best_weights_e.w_ce:.3f}, w_multi={best_weights_e.w_multiplicity:.3f} -> Tuning MRR={best_mrr_e:.4f}")

    w_e = best_weights_e

    def score_c(c: dict) -> float:
        f = c["ext_features"]
        if f["similarity"] < 0.10:
            return c["s_mass"]
        raw = best_weights_c.w_similarity * f["similarity"] + best_weights_c.w_peaks * f["peaks"]
        return c["s_mass"] + apply_boost(raw)

    def score_d(c: dict) -> float:
        f = c["ext_features"]
        if f["similarity"] < 0.10:
            return c["s_mass"]
        raw = best_weights_d.w_similarity * f["similarity"] + best_weights_d.w_peaks * f["peaks"] + best_weights_d.w_ce * f["ce"]
        return c["s_mass"] + apply_boost(raw)

    def score_e(c: dict) -> float:
        f = c["ext_features"]
        if f["similarity"] < 0.10:
            return c["s_mass"]
        raw = (
            w_e.w_similarity * f["similarity"]
            + w_e.w_peaks * f["peaks"]
            + w_e.w_ce * f["ce"]
            + w_e.w_multiplicity * f["multiplicity"]
        )
        return c["s_mass"] + apply_boost(raw)

    # Tune Stage 5 mixing weight gamma for Configuration F
    best_gamma = 1.0
    best_mrr_f = -1.0
    for gamma in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        def score_f_test(c: dict, g=gamma) -> float:
            base_e = score_e(c)
            s5_prob = c.get("stage5_prob", 0.50)
            return base_e + g * s5_prob
        mrr = evaluate_ranking(tuning_records, score_f_test)["mrr25"]
        if mrr > best_mrr_f:
            best_mrr_f = mrr
            best_gamma = gamma

    print(f"  Best Stage 5 Mixing Weight (gamma): {best_gamma} -> Tuning MRR={best_mrr_f:.4f}")

    def score_f(c: dict) -> float:
        base_e = score_e(c)
        s5_prob = c.get("stage5_prob", 0.50)
        return base_e + best_gamma * s5_prob

    # 6. Evaluate All Configurations A-F on Both Tuning and Frozen Benchmark
    print("\n[6/6] Final Frozen Evaluation across all Ablation Stages...", flush=True)

    configs = [
        ("A: 839k Baseline", score_a_baseline),
        ("B: + External Cosine", score_b_ext_cosine),
        ("C: + Cosine + Peaks", score_c),
        ("D: + Cosine + Peaks + CE", score_d),
        ("E: + Cosine + Peaks + CE + Multi", score_e),
        ("F: E + Stage 5 Neural", score_f),
    ]

    tuning_evals = {}
    benchmark_evals = {}
    novel_bench_evals = {}
    known_bench_evals = {}

    for name, s_func in configs:
        tuning_evals[name] = evaluate_ranking(tuning_records, s_func)
        benchmark_evals[name] = evaluate_ranking(benchmark_records, s_func)
        novel_bench_evals[name] = evaluate_ranking(benchmark_records[:200], s_func)
        known_bench_evals[name] = evaluate_ranking(benchmark_records[200:], s_func)

    # Print Table 1: Tuning Set Performance (800 Queries)
    print("\n" + "=" * 90)
    print("  TABLE 1: STEP-BY-STEP ABLATION ON LOCAL TUNING SET (800 QUERIES)")
    print("=" * 90)
    print(f"{'Configuration':<34} | {'MRR@25':<10} | {'Hit@1 (%)':<10} | {'Hit@5 (%)':<10} | {'Isomer MRR':<12} | {'Delta MRR':<10}")
    print("-" * 90)
    base_mrr_tune = tuning_evals["A: 839k Baseline"]["mrr25"]
    for name, _ in configs:
        res = tuning_evals[name]
        d = res["mrr25"] - base_mrr_tune
        print(f"{name:<34} | {res['mrr25']:<10.4f} | {res['hit1']:<10.2f} | {res['hit5']:<10.2f} | {res['isomer_mrr']:<12.4f} | {d:+10.4f}")
    print("=" * 90)

    # Print Table 2: Frozen Benchmark Performance (400 Queries: 200 Novel + 200 Known)
    print("\n" + "=" * 95)
    print("  TABLE 2: FROZEN BENCHMARK EVALUATION (400 QUERIES: 200 NOVEL + 200 KNOWN)")
    print("=" * 95)
    print(f"{'Configuration':<34} | {'Overall MRR':<12} | {'Hit@1 (%)':<10} | {'Novel MRR':<11} | {'Isomer MRR':<12} | {'Delta MRR':<10}")
    print("-" * 95)
    base_mrr_bench = benchmark_evals["A: 839k Baseline"]["mrr25"]
    for name, _ in configs:
        res = benchmark_evals[name]
        n_res = novel_bench_evals[name]
        d = res["mrr25"] - base_mrr_bench
        print(f"{name:<34} | {res['mrr25']:<12.4f} | {res['hit1']:<10.2f} | {n_res['mrr25']:<11.4f} | {res['isomer_mrr']:<12.4f} | {d:+10.4f}")
    print("=" * 95)

    # Save Results Artifact
    out_dict = {
        "weights": {
            "w_similarity": float(w_e.w_similarity),
            "w_peaks": float(w_e.w_peaks),
            "w_ce": float(w_e.w_ce),
            "w_multiplicity": float(w_e.w_multiplicity),
            "gamma_stage5": float(best_gamma),
        },
        "tuning_results": tuning_evals,
        "benchmark_overall": benchmark_evals,
        "benchmark_novel": novel_bench_evals,
        "benchmark_known": known_bench_evals,
    }
    out_path = ROOT / "artifacts" / "external" / "evidence_ablation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nAll ablation results saved to: {out_path}")
    print(f"Total ablation elapsed time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
