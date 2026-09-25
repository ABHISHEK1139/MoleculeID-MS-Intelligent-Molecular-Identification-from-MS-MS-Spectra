"""Phase 2 Diagnostic Gate: 4-Way External Spectral Library Evaluation.

Evaluates 4 systems head-to-head on the EXACT same 400 frozen benchmark queries:
  A. Baseline: Existing 839k multi-CE library
  B. +MoNA:    Existing + MoNA (536k spectra)
  C. +GNPS:    Existing + GNPS (8.5k spectra)
  D. +Both:    Existing + MoNA + GNPS (545k spectra)

Strictly identical conditions:
- Same 400 queries (200 Novel held-out + 200 Known in-catalog)
- Same candidate universe
- Same +-20 ppm mass filtering
- Same peak preprocessing
- Same modified cosine matching
- Same CE matching logic (5 eV tolerance bonus)
- Same ranking and router logic

Computes:
1. Core performance table (MRR@25, Hit@1, Hit@5, Hit@25, Coverage, Direct hits, Isomer MRR, Novel MRR)
2. Multiplicity analysis (1 vs 2+ vs 3+ supporting spectra, source corroboration)
3. Sub-cohort breakdown (In-277k catalog vs Novel, Isomer vs Non-isomer)
"""
from __future__ import annotations

import concurrent.futures
import json
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

from src.core.canonical_benchmark import CanonicalBenchmark, CanonicalQuery


# ── Fast Mutual Cosine Matching ───────────────────────────────────────────
def fast_mutual_cosine(
    q_mzs: np.ndarray,
    q_ints: np.ndarray,
    r_mzs: np.ndarray,
    r_ints: np.ndarray,
    delta: float = 0.0,
    tol: float = 0.01,
) -> tuple[float, int]:
    """Mutual 1-to-1 peak-matching modified cosine, strictly in [0, 1.0]."""
    if q_mzs.size == 0 or r_mzs.size == 0:
        return 0.0, 0

    q_norm = np.linalg.norm(q_ints)
    r_norm = np.linalg.norm(r_ints)
    if q_norm <= 0 or r_norm <= 0:
        return 0.0, 0

    qi = (q_ints / q_norm).astype(np.float32)
    li = (r_ints / r_norm).astype(np.float32)

    best_dot = 0.0
    best_cnt = 0
    shifts = [0.0] if abs(delta) < 1e-4 else [0.0, delta]

    for s in shifts:
        qs = q_mzs + s
        order = np.argsort(qs)
        q_sorted = qs[order]
        qi_sorted = qi[order]

        lo = np.searchsorted(q_sorted, r_mzs - tol, side="left")
        hi = np.searchsorted(q_sorted, r_mzs + tol, side="right")

        dot = 0.0
        cnt = 0
        matched_q: set[int] = set()
        for j in range(r_mzs.size):
            a, b = int(lo[j]), int(hi[j])
            if b > a:
                if b - a == 1:
                    m_idx = a
                else:
                    diffs = np.abs(q_sorted[a:b] - r_mzs[j])
                    m_idx = a + int(np.argmin(diffs))
                if m_idx not in matched_q:
                    matched_q.add(m_idx)
                    dot += float(li[j] * qi_sorted[m_idx])
                    cnt += 1

        if dot > best_dot:
            best_dot = dot
            best_cnt = cnt

    return min(1.0, float(best_dot)), best_cnt


# ── Spectral Reference Library Container ───────────────────────────────────
class CompactSpectralLibrary:
    """Pre-sorted in-memory spectral library with binary-search by mass."""

    def __init__(
        self,
        smiles: np.ndarray,
        neutral_masses: np.ndarray,
        precursor_mzs: np.ndarray,
        collision_energies: np.ndarray,
        mzs_list: list[np.ndarray],
        intens_list: list[np.ndarray],
        n_supporting: np.ndarray | None = None,
        source_counts: np.ndarray | None = None,
    ):
        order = np.argsort(neutral_masses)
        self.smiles = smiles[order]
        self.neutral_masses = neutral_masses[order]
        self.precursor_mzs = precursor_mzs[order]
        self.collision_energies = collision_energies[order]
        self.mzs_list = [mzs_list[i] for i in order]
        self.intens_list = [intens_list[i] for i in order]
        self.n_supporting = n_supporting[order] if n_supporting is not None else np.ones(len(order), dtype=np.int16)
        self.source_counts = source_counts[order] if source_counts is not None else np.ones(len(order), dtype=np.int8)

    def query_window(self, mass: float, ppm: float = 20.0) -> tuple[int, int]:
        l_idx = np.searchsorted(self.neutral_masses, mass * (1.0 - ppm * 1e-6))
        r_idx = np.searchsorted(self.neutral_masses, mass * (1.0 + ppm * 1e-6))
        return int(l_idx), int(r_idx)


# ── Query Worker Function ──────────────────────────────────────────────────
def evaluate_query_on_library(
    q_data: dict[str, Any],
    lib: CompactSpectralLibrary,
) -> dict[str, Any]:
    """Evaluate a single query on a given spectral library."""
    target_mass = q_data["target_mass"]
    prec_mz = q_data["precursor_mz"]
    q_ce = q_data["ce"]
    q_mzs = q_data["q_mzs"]
    q_ints = q_data["q_intens"]
    true_smi = q_data["true_smi"]
    candidates = q_data["candidates"]  # list of dicts: {'smi': ..., 'ppm_err': ..., 'tier': ...}

    # 1. Physics baseline score for candidates
    cand_scores = {}
    for c in candidates:
        smi = c["smi"]
        ppm_err = c["ppm_err"]
        tier = c["tier"]
        tier_w = 1.0 if tier == 1 else 0.50
        s_mass = 1.50 * np.exp(-ppm_err / 10.0) * tier_w
        cand_scores[smi] = s_mass

    # 2. Spectral search in +-20 ppm window
    l_idx, r_idx = lib.query_window(target_mass, ppm=20.0)

    mol_spectral_hits: dict[str, list[dict[str, Any]]] = {}
    best_query_cos = 0.0

    if r_idx > l_idx:
        for ri in range(l_idx, r_idx):
            ref_smi = lib.smiles[ri]
            ref_prec = lib.precursor_mzs[ri]
            ref_mzs = lib.mzs_list[ri]
            ref_intens = lib.intens_list[ri]
            r_ce = lib.collision_energies[ri]
            n_sup = int(lib.n_supporting[ri])
            src_cnt = int(lib.source_counts[ri])

            delta = prec_mz - ref_prec
            cos_sim, n_peaks = fast_mutual_cosine(q_mzs, q_ints, ref_mzs, ref_intens, delta=delta)

            ce_match = False
            ce_diff = float("nan")
            if np.isfinite(q_ce) and np.isfinite(r_ce):
                ce_diff = abs(q_ce - r_ce)
                ce_match = ce_diff <= 5.0

            if cos_sim > 0.10:
                if ref_smi not in mol_spectral_hits:
                    mol_spectral_hits[ref_smi] = []
                mol_spectral_hits[ref_smi].append({
                    "cos": cos_sim,
                    "n_peaks": n_peaks,
                    "ce_match": ce_match,
                    "ce_diff": ce_diff,
                    "n_supporting": n_sup,
                    "source_count": src_cnt,
                })
                if cos_sim > best_query_cos:
                    best_query_cos = cos_sim

    # 3. Aggregate per-molecule spectral evidence
    spectral_matches: dict[str, tuple[float, int, dict[str, Any]]] = {}

    for ref_smi, hits in mol_spectral_hits.items():
        hits.sort(key=lambda x: x["cos"], reverse=True)
        top_hit = hits[0]

        if len(hits) == 1:
            agg_cos = top_hit["cos"]
            agg_peaks = top_hit["n_peaks"]
            ce_bonus = 1.10 if top_hit["ce_match"] else 1.00
        else:
            agg_cos = 0.65 * hits[0]["cos"] + 0.35 * hits[1]["cos"]
            agg_peaks = max(hits[0]["n_peaks"], hits[1]["n_peaks"])
            any_ce = any(h["ce_match"] for h in hits[:2])
            ce_bonus = 1.12 if any_ce else 1.00
            if hits[1]["cos"] > 0.30:
                ce_bonus *= 1.08  # multi-spectrum consistency bonus

        spectral_matches[ref_smi] = (agg_cos * ce_bonus, agg_peaks, top_hit)

    best_agg_cos = max((v[0] for v in spectral_matches.values()), default=0.0)

    # 4. Hybrid Scoring & Dynamic Router
    if best_agg_cos >= 0.45:
        route = "library"
    elif best_agg_cos >= 0.20:
        route = "partial"
    else:
        route = "physics"

    for ref_smi, (agg_cos, n_peaks, _) in spectral_matches.items():
        peak_bonus = 0.60 + 0.40 * min(1.0, n_peaks / 5.0)
        s_spec = agg_cos * peak_bonus
        curr_s = cand_scores.get(ref_smi, 0.50)

        if route == "library":
            cand_scores[ref_smi] = curr_s + 4.50 * (s_spec ** 2)
        elif route == "partial":
            cand_scores[ref_smi] = curr_s + 2.50 * (s_spec ** 2)
        else:
            cand_scores[ref_smi] = curr_s + 1.00 * s_spec

    # 5. Rank candidates
    ranked = sorted(cand_scores.keys(), key=lambda s: cand_scores[s], reverse=True)

    rank = 0
    if true_smi in ranked:
        rank = ranked.index(true_smi) + 1

    recip_rank = 1.0 / rank if 1 <= rank <= 25 else 0.0

    has_coverage = len(mol_spectral_hits) > 0
    direct_correct_hit = False
    true_hit_info = None
    if true_smi in mol_spectral_hits:
        true_hits = mol_spectral_hits[true_smi]
        if any(h["cos"] >= 0.50 for h in true_hits):
            direct_correct_hit = True
            true_hit_info = true_hits[0]

    # Collect matched candidates info for multiplicity analysis
    candidate_match_records = []
    for s_mol, (_, _, hit_dict) in spectral_matches.items():
        candidate_match_records.append({
            "is_true_mol": (s_mol == true_smi),
            "cos": hit_dict["cos"],
            "n_peaks": hit_dict["n_peaks"],
            "ce_diff": hit_dict["ce_diff"],
            "n_supporting": hit_dict["n_supporting"],
            "source_count": hit_dict["source_count"],
        })

    return {
        "rank": rank,
        "mrr": recip_rank,
        "hit1": 1.0 if rank == 1 else 0.0,
        "hit5": 1.0 if 1 <= rank <= 5 else 0.0,
        "hit25": 1.0 if 1 <= rank <= 25 else 0.0,
        "covered": has_coverage,
        "direct_hit": direct_correct_hit,
        "candidate_matches": candidate_match_records,
    }


# ── Batch Worker for Multiprocessing ───────────────────────────────────────
def evaluate_batch(args: tuple[list[dict], CompactSpectralLibrary]) -> list[dict]:
    query_batch, lib = args
    results = []
    for q in query_batch:
        results.append(evaluate_query_on_library(q, lib))
    return results


# ── Main Diagnostic Suite ──────────────────────────────────────────────────
def main():
    t_start = time.time()
    num_cpus = os.cpu_count() or 4
    n_workers = min(12, num_cpus)

    print("=" * 80)
    print("  PHASE 2 DIAGNOSTIC GATE: 4-WAY EXTERNAL SPECTRAL LIBRARY BENCHMARK")
    print(f"  Running in parallel across {n_workers} CPU workers on {num_cpus}-core system")
    print("=" * 80, flush=True)

    # 1. Load Canonical Benchmark (200 novel queries + 200 known queries = 400 total)
    print("\n[1/5] Ingesting 400 Canonical Benchmark Queries...", flush=True)
    bm = CanonicalBenchmark(subset_size=10000, n_benchmark_queries=200, split_seed=42, benchmark_seed=123)
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

    all_raw_queries = known_queries + novel_queries
    print(f"Loaded {len(known_queries)} Known queries + {len(novel_queries)} Novel queries = {len(all_raw_queries)} total.")

    # 2. Check 277k candidate presence
    cand_277k_path = ROOT / "kaggle_dataset" / "candidate_library.parquet"
    cand_277k_smiles = set()
    if cand_277k_path.exists():
        cand_277k_smiles = set(pd.read_parquet(cand_277k_path, columns=["normalized_smiles"])["normalized_smiles"])
        print(f"Loaded 277k candidate universe: {len(cand_277k_smiles):,} molecules.")

    # Package query dictionaries for multi-processing
    packaged_queries = []
    for i, q in enumerate(all_raw_queries):
        is_known = (i < len(known_queries))
        sample_idx = q.query_id
        ds = bm.train_ds if is_known else bm.val_ds
        spec_info = ds.samples[sample_idx][0]

        true_smi = bm.cand_db.mol_smiles.get(q.true_mol, "")
        cand_list = []
        for m in q.matches:
            c_smi = bm.cand_db.mol_smiles.get(m.mol, "")
            if c_smi:
                cand_list.append({"smi": c_smi, "ppm_err": m.ppm_error, "tier": m.tier})

        packaged_queries.append({
            "idx": i,
            "regime": "known" if is_known else "novel",
            "is_novel_regime": not is_known,
            "true_smi": true_smi,
            "target_mass": q.target_mass,
            "precursor_mz": q.precursor_mz,
            "adduct": q.adduct,
            "ce": spec_info.get("ce", float("nan")),
            "q_mzs": np.asarray(spec_info["mz"], dtype=np.float32),
            "q_intens": np.asarray(spec_info["intensity"], dtype=np.float32),
            "candidates": cand_list,
            "is_isomer": q.is_isomer_query,
            "in_277k": true_smi in cand_277k_smiles,
        })

    n_in_277k = sum(1 for q in packaged_queries if q["in_277k"])
    n_novel_277k = len(packaged_queries) - n_in_277k
    n_isomers = sum(1 for q in packaged_queries if q["is_isomer"])
    print(f"Cohort Breakdown: {n_in_277k} present in 277k catalog | {n_novel_277k} absent from 277k | {n_isomers} multi-isomer cases.")

    # 3. Load Reference Library and External Library
    print("\n[2/5] Loading and building the 4 comparison libraries...", flush=True)
    t_lib = time.time()

    ref_path = ROOT / "kaggle_dataset" / "reference_library_multice.parquet"
    df_ref = pd.read_parquet(ref_path, columns=["normalized_smiles", "neutral_mass", "precursor_mz", "collision_energy", "ms2_mzs", "ms2_intensities"])

    ext_path = ROOT / "artifacts" / "external" / "external_spectra.parquet"
    df_ext = pd.read_parquet(ext_path, columns=[
        "canonical_smiles", "neutral_mass", "precursor_mz", "collision_energy",
        "peaks_mz", "peaks_intensity", "source_library", "n_supporting_spectra", "source_count"
    ])

    print(f"Loaded {len(df_ref):,} baseline spectra and {len(df_ext):,} external spectra in {time.time()-t_lib:.1f}s.")

    # Build Library A (Baseline)
    lib_a = CompactSpectralLibrary(
        smiles=df_ref["normalized_smiles"].to_numpy(),
        neutral_masses=df_ref["neutral_mass"].to_numpy(dtype=np.float64),
        precursor_mzs=df_ref["precursor_mz"].to_numpy(dtype=np.float64),
        collision_energies=df_ref["collision_energy"].to_numpy(dtype=np.float32),
        mzs_list=[np.asarray(x, dtype=np.float32) for x in df_ref["ms2_mzs"]],
        intens_list=[np.asarray(x, dtype=np.float32) for x in df_ref["ms2_intensities"]],
    )

    # Build Library B (Baseline + MoNA)
    df_mona = df_ext[df_ext["source_library"] == "MoNA"]
    b_smiles = np.concatenate([df_ref["normalized_smiles"].to_numpy(), df_mona["canonical_smiles"].to_numpy()])
    b_masses = np.concatenate([df_ref["neutral_mass"].to_numpy(dtype=np.float64), df_mona["neutral_mass"].to_numpy(dtype=np.float64)])
    b_precs = np.concatenate([df_ref["precursor_mz"].to_numpy(dtype=np.float64), df_mona["precursor_mz"].to_numpy(dtype=np.float64)])
    b_ces = np.concatenate([df_ref["collision_energy"].to_numpy(dtype=np.float32), df_mona["collision_energy"].to_numpy(dtype=np.float32)])
    b_mzs = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_mzs"]] + [np.asarray(x, dtype=np.float32) for x in df_mona["peaks_mz"]]
    b_ints = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_intensities"]] + [np.asarray(x, dtype=np.float32) for x in df_mona["peaks_intensity"]]
    b_sup = np.concatenate([np.ones(len(df_ref), dtype=np.int16), df_mona["n_supporting_spectra"].to_numpy(dtype=np.int16)])
    b_src = np.concatenate([np.ones(len(df_ref), dtype=np.int8), df_mona["source_count"].to_numpy(dtype=np.int8)])

    lib_b = CompactSpectralLibrary(b_smiles, b_masses, b_precs, b_ces, b_mzs, b_ints, b_sup, b_src)

    # Build Library C (Baseline + GNPS)
    df_gnps = df_ext[df_ext["source_library"].str.startswith("GNPS")]
    c_smiles = np.concatenate([df_ref["normalized_smiles"].to_numpy(), df_gnps["canonical_smiles"].to_numpy()])
    c_masses = np.concatenate([df_ref["neutral_mass"].to_numpy(dtype=np.float64), df_gnps["neutral_mass"].to_numpy(dtype=np.float64)])
    c_precs = np.concatenate([df_ref["precursor_mz"].to_numpy(dtype=np.float64), df_gnps["precursor_mz"].to_numpy(dtype=np.float64)])
    c_ces = np.concatenate([df_ref["collision_energy"].to_numpy(dtype=np.float32), df_gnps["collision_energy"].to_numpy(dtype=np.float32)])
    c_mzs = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_mzs"]] + [np.asarray(x, dtype=np.float32) for x in df_gnps["peaks_mz"]]
    c_ints = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_intensities"]] + [np.asarray(x, dtype=np.float32) for x in df_gnps["peaks_intensity"]]
    c_sup = np.concatenate([np.ones(len(df_ref), dtype=np.int16), df_gnps["n_supporting_spectra"].to_numpy(dtype=np.int16)])
    c_src = np.concatenate([np.ones(len(df_ref), dtype=np.int8), df_gnps["source_count"].to_numpy(dtype=np.int8)])

    lib_c = CompactSpectralLibrary(c_smiles, c_masses, c_precs, c_ces, c_mzs, c_ints, c_sup, c_src)

    # Build Library D (Baseline + MoNA + GNPS = Both)
    d_smiles = np.concatenate([df_ref["normalized_smiles"].to_numpy(), df_ext["canonical_smiles"].to_numpy()])
    d_masses = np.concatenate([df_ref["neutral_mass"].to_numpy(dtype=np.float64), df_ext["neutral_mass"].to_numpy(dtype=np.float64)])
    d_precs = np.concatenate([df_ref["precursor_mz"].to_numpy(dtype=np.float64), df_ext["precursor_mz"].to_numpy(dtype=np.float64)])
    d_ces = np.concatenate([df_ref["collision_energy"].to_numpy(dtype=np.float32), df_ext["collision_energy"].to_numpy(dtype=np.float32)])
    d_mzs = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_mzs"]] + [np.asarray(x, dtype=np.float32) for x in df_ext["peaks_mz"]]
    d_ints = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_intensities"]] + [np.asarray(x, dtype=np.float32) for x in df_ext["peaks_intensity"]]
    d_sup = np.concatenate([np.ones(len(df_ref), dtype=np.int16), df_ext["n_supporting_spectra"].to_numpy(dtype=np.int16)])
    d_src = np.concatenate([np.ones(len(df_ref), dtype=np.int8), df_ext["source_count"].to_numpy(dtype=np.int8)])

    lib_d = CompactSpectralLibrary(d_smiles, d_masses, d_precs, d_ces, d_mzs, d_ints, d_sup, d_src)

    systems = [
        ("Baseline (839k)", lib_a),
        ("+ MoNA", lib_b),
        ("+ GNPS", lib_c),
        ("+ Both (MoNA+GNPS)", lib_d),
    ]

    print("Libraries ready:")
    for name, lib in systems:
        print(f"  {name:<22s}: {len(lib.neutral_masses):>10,} indexed spectra")

    # 4. Run Evaluation in Parallel across Systems
    print("\n[3/5] Evaluating all 4 systems in parallel across 400 queries...", flush=True)

    system_results = {}

    def run_single_system(sys_tuple: tuple[str, CompactSpectralLibrary]) -> tuple[str, list[dict]]:
        sys_name, lib_obj = sys_tuple
        t_sys = time.time()
        res = [evaluate_query_on_library(q, lib_obj) for q in packaged_queries]
        print(f"  System '{sys_name}' finished in {time.time() - t_sys:.1f}s ({len(res)} queries evaluated)", flush=True)
        return sys_name, res

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_single_system, s) for s in systems]
        for f in concurrent.futures.as_completed(futures):
            s_name, s_res = f.result()
            system_results[s_name] = s_res

    # 5. Compute Comprehensive Metrics
    print("\n[4/5] Computing Benchmark Performance Metrics & Multiplicity...", flush=True)

    def calc_metrics(res_list: list[dict], mask: list[bool] | None = None) -> dict[str, float]:
        if mask is not None:
            sub = [r for r, m in zip(res_list, mask) if m]
        else:
            sub = res_list

        n = len(sub)
        if n == 0:
            return {"coverage": 0.0, "direct_hits": 0.0, "hit1": 0.0, "mrr25": 0.0, "hit5": 0.0, "hit25": 0.0}

        coverage = np.mean([1.0 if r["covered"] else 0.0 for r in sub]) * 100
        direct_hits = np.sum([1.0 if r["direct_hit"] else 0.0 for r in sub])
        hit1 = np.mean([r["hit1"] for r in sub]) * 100
        mrr25 = np.mean([r["mrr"] for r in sub])
        hit5 = np.mean([r["hit5"] for r in sub]) * 100
        hit25 = np.mean([r["hit25"] for r in sub]) * 100

        return {
            "coverage": coverage,
            "direct_hits": int(direct_hits),
            "hit1": hit1,
            "mrr25": mrr25,
            "hit5": hit5,
            "hit25": hit25,
        }

    # Masks for query subsets
    novel_mask = [q["is_novel_regime"] for q in packaged_queries]
    known_mask = [not q["is_novel_regime"] for q in packaged_queries]
    isomer_mask = [q["is_isomer"] for q in packaged_queries]
    in_277k_mask = [q["in_277k"] for q in packaged_queries]
    absent_277k_mask = [not q["in_277k"] for q in packaged_queries]

    # Metrics table
    metrics_summary = {}
    for sys_name in system_results:
        res = system_results[sys_name]
        overall = calc_metrics(res)
        novel = calc_metrics(res, novel_mask)
        known = calc_metrics(res, known_mask)
        isomer = calc_metrics(res, isomer_mask)
        in_277k = calc_metrics(res, in_277k_mask)
        absent_277k = calc_metrics(res, absent_277k_mask)

        metrics_summary[sys_name] = {
            "Overall": overall,
            "Novel": novel,
            "Known": known,
            "Isomer": isomer,
            "In_277K": in_277k,
            "Absent_277K": absent_277k,
        }

    # Print Primary Table
    print("\n" + "=" * 95)
    print("  EXACT HEAD-TO-HEAD BENCHMARK: EXTERNAL SPECTRAL LIBRARY DIAGNOSTIC (400 QUERIES)")
    print("=" * 95)
    headers = ["Metric", "Baseline (839k)", "+ MoNA", "+ GNPS", "+ Both (MoNA+GNPS)"]
    print(f"{headers[0]:<30} | {headers[1]:<14} | {headers[2]:<14} | {headers[3]:<14} | {headers[4]:<18}")
    print("-" * 100)

    b_res = metrics_summary["Baseline (839k)"]
    m_res = metrics_summary["+ MoNA"]
    g_res = metrics_summary["+ GNPS"]
    both_res = metrics_summary["+ Both (MoNA+GNPS)"]

    rows = [
        ("Query Spectral Coverage", f"{b_res['Overall']['coverage']:.1f}%", f"{m_res['Overall']['coverage']:.1f}%", f"{g_res['Overall']['coverage']:.1f}%", f"{both_res['Overall']['coverage']:.1f}%"),
        ("Direct Correct-Spectrum Hits", f"{b_res['Overall']['direct_hits']}", f"{m_res['Overall']['direct_hits']}", f"{g_res['Overall']['direct_hits']}", f"{both_res['Overall']['direct_hits']}"),
        ("Rank-1 Accuracy (Hit@1)", f"{b_res['Overall']['hit1']:.2f}%", f"{m_res['Overall']['hit1']:.2f}%", f"{g_res['Overall']['hit1']:.2f}%", f"{both_res['Overall']['hit1']:.2f}%"),
        ("MRR@25 (Overall)", f"{b_res['Overall']['mrr25']:.4f}", f"{m_res['Overall']['mrr25']:.4f}", f"{g_res['Overall']['mrr25']:.4f}", f"{both_res['Overall']['mrr25']:.4f}"),
        ("Hit@5", f"{b_res['Overall']['hit5']:.2f}%", f"{m_res['Overall']['hit5']:.2f}%", f"{g_res['Overall']['hit5']:.2f}%", f"{both_res['Overall']['hit5']:.2f}%"),
        ("Hit@25", f"{b_res['Overall']['hit25']:.2f}%", f"{m_res['Overall']['hit25']:.2f}%", f"{g_res['Overall']['hit25']:.2f}%", f"{both_res['Overall']['hit25']:.2f}%"),
        ("Isomer MRR (Hard Cases)", f"{b_res['Isomer']['mrr25']:.4f}", f"{m_res['Isomer']['mrr25']:.4f}", f"{g_res['Isomer']['mrr25']:.4f}", f"{both_res['Isomer']['mrr25']:.4f}"),
        ("Novel-Query MRR (Held-out)", f"{b_res['Novel']['mrr25']:.4f}", f"{m_res['Novel']['mrr25']:.4f}", f"{g_res['Novel']['mrr25']:.4f}", f"{both_res['Novel']['mrr25']:.4f}"),
        ("Known-Query MRR (In-Library)", f"{b_res['Known']['mrr25']:.4f}", f"{m_res['Known']['mrr25']:.4f}", f"{g_res['Known']['mrr25']:.4f}", f"{both_res['Known']['mrr25']:.4f}"),
    ]

    for label, v1, v2, v3, v4 in rows:
        print(f"{label:<30} | {v1:<14} | {v2:<14} | {v3:<14} | {v4:<18}")

    print("=" * 95)

    # Print Sub-Cohort Breakdown Table
    print("\n" + "=" * 95)
    print("  SUB-COHORT SLICING: CATALOG & ISOMER GAINS")
    print("=" * 95)
    print(f"{'Cohort':<32} | {'Queries':<8} | {'Baseline MRR':<14} | {'+ Both MRR':<14} | {'Absolute Delta':<14}")
    print("-" * 95)

    cohorts = [
        ("Present in 277K Catalog", sum(in_277k_mask), b_res['In_277K']['mrr25'], both_res['In_277K']['mrr25']),
        ("Absent from 277K Catalog", sum(absent_277k_mask), b_res['Absent_277K']['mrr25'], both_res['Absent_277K']['mrr25']),
        ("Exact Isomer Queries", sum(isomer_mask), b_res['Isomer']['mrr25'], both_res['Isomer']['mrr25']),
        ("Non-Isomer Queries", len(packaged_queries) - sum(isomer_mask),
         calc_metrics(system_results["Baseline (839k)"], [not m for m in isomer_mask])['mrr25'],
         calc_metrics(system_results["+ Both (MoNA+GNPS)"], [not m for m in isomer_mask])['mrr25']),
    ]

    for c_name, c_cnt, m_base, m_both in cohorts:
        d = m_both - m_base
        d_str = f"{d:+.4f}"
        print(f"{c_name:<32} | {c_cnt:<8} | {m_base:<14.4f} | {m_both:<14.4f} | {d_str:<14}")

    print("=" * 95)

    # 6. Multiplicity Analysis (Is Agreement Predictive?)
    print("\n[5/5] Analyzing Multiplicity & Corroboration Signal...", flush=True)

    both_records = []
    for r in system_results["+ Both (MoNA+GNPS)"]:
        both_records.extend(r["candidate_matches"])

    df_cand_matches = pd.DataFrame(both_records)
    if not df_cand_matches.empty:
        print("\n" + "=" * 80)
        print("  MULTIPLICITY ANALYSIS: IS SPECTRAL AGREEMENT PREDICTIVE?")
        print("=" * 80)
        print(f"{'Multiplicity Level':<30} | {'Total Matches':<14} | {'True Positives':<15} | {'Precision (Hit Rate)':<20}")
        print("-" * 80)

        # Multiplicity slices
        s1 = df_cand_matches[df_cand_matches["n_supporting"] == 1]
        s2 = df_cand_matches[df_cand_matches["n_supporting"] >= 2]
        s3 = df_cand_matches[df_cand_matches["n_supporting"] >= 3]
        src_multi = df_cand_matches[df_cand_matches["source_count"] >= 2]

        for label, subset in [
            ("Single supporting spectrum (N=1)", s1),
            ("2+ supporting spectra (N>=2)", s2),
            ("3+ supporting spectra (N>=3)", s3),
            ("Multiple independent sources (S>=2)", src_multi),
        ]:
            tot = len(subset)
            tp = int(subset["is_true_mol"].sum())
            prec = (tp / tot * 100) if tot > 0 else 0.0
            print(f"{label:<30} | {tot:>13,} | {tp:>14,} | {prec:>18.2f}%")

        print("=" * 80)

    # Save complete results artifact
    out_path = ROOT / "artifacts" / "external" / "benchmark_diagnostic_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nDiagnostic results saved to: {out_path}")
    print(f"Total benchmark elapsed time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
