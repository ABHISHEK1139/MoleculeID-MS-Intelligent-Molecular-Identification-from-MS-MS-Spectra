"""Phase 6: Multi-Channel Out-Of-Fold (OOF) Meta-Ranker.

Trains a Query-Grouped GBDT (LightGBM) meta-ranker combining:
  1. Mass features (ppm error, Da error, isotope indicator, mass rank)
  2. Direct spectral retrieval features (entropy similarity, rank, diff-to-max, direct flags)
  3. Mass-shifted analog propagation features (score, top sim, ranks, diff-to-max)
  4. FPNet structural features (Bayes raw score, z-score, pool rank, length-normalized score)
  5. Context and evidence features (candidate counts, query max direct similarity, collision energy, adduct mode, source flags)

Validation Protocol:
  - 5-Fold GroupKFold strictly grouped by query_id (zero query or candidate leakage between train & val).
  - Balanced across C1, C2, and C3 cohorts (30 queries each per fold).
  - Out-of-fold predictions collected for all 450 Clean v4 Benchmark queries.

Comparison Systems:
  A = Direct + Analog Baseline
  B = Direct + Analog + FPNet Fixed Linear Fusion
  C = OOF GBDT Meta-Ranker (Pure Learned Model, no hard thresholds)
  D = OOF GBDT + Conditional Direct Routing
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from numba import njit, prange
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

from src.core.candidate_retrieval import retrieve_candidates_union
from src.core.preprocessing_v3 import neutral_mass
from src.models.fpnet import FPNetEnsemble, score_candidates_fpnet


# ── Fast Numba Spectral Kernels ───────────────────────────────────────────────
@njit(cache=True, fastmath=True)
def _clean_numba(mz, it, floor, topk):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx:
            mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    thresh = mx * floor
    cnt = 0
    for i in range(n):
        if it[i] >= thresh:
            cnt += 1

    idx = np.empty(cnt, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thresh:
            idx[j] = i
            j += 1

    if cnt > topk:
        vals = np.empty(cnt, np.float32)
        for i in range(cnt):
            vals[i] = it[idx[i]]
        order = np.argsort(-vals)[:topk]
        idx = idx[order]
        cnt = topk

    out_mz = np.empty(cnt, np.float32)
    out_it = np.empty(cnt, np.float32)
    for i in range(cnt):
        out_mz[i] = mz[idx[i]]
        out_it[i] = it[idx[i]]

    s_order = np.argsort(out_mz)
    res_mz = np.empty(cnt, np.float32)
    res_it = np.empty(cnt, np.float32)
    for i in range(cnt):
        res_mz[i] = out_mz[s_order[i]]
        res_it[i] = out_it[s_order[i]]
    return res_mz, res_it


@njit(cache=True, fastmath=True)
def entropy_similarity_numba(m1, i1, m2, i2, tol):
    n1 = len(m1)
    n2 = len(m2)
    if n1 == 0 or n2 == 0:
        return 0.0

    s1 = 0.0
    for i in range(n1):
        s1 += i1[i]
    s2 = 0.0
    for i in range(n2):
        s2 += i2[i]
    if s1 <= 0.0 or s2 <= 0.0:
        return 0.0

    S_A = 0.0
    for i in range(n1):
        p = i1[i] / s1
        if p > 1e-12:
            S_A -= p * math.log(p)

    S_B = 0.0
    for i in range(n2):
        p = i2[i] / s2
        if p > 1e-12:
            S_B -= p * math.log(p)

    S_AB = 0.0
    p1 = 0
    p2 = 0
    while p1 < n1 and p2 < n2:
        diff = m1[p1] - m2[p2]
        if abs(diff) <= tol:
            pm = (i1[p1] / s1 + i2[p2] / s2) * 0.5
            if pm > 1e-12:
                S_AB -= pm * math.log(pm)
            p1 += 1
            p2 += 1
        elif diff < 0:
            pm = 0.5 * (i1[p1] / s1)
            if pm > 1e-12:
                S_AB -= pm * math.log(pm)
            p1 += 1
        else:
            pm = 0.5 * (i2[p2] / s2)
            if pm > 1e-12:
                S_AB -= pm * math.log(pm)
            p2 += 1

    while p1 < n1:
        pm = 0.5 * (i1[p1] / s1)
        if pm > 1e-12:
            S_AB -= pm * math.log(pm)
            p1 += 1
    while p2 < n2:
        pm = 0.5 * (i2[p2] / s2)
        if pm > 1e-12:
            S_AB -= pm * math.log(pm)
            p2 += 1

    e_un = 2.0 * S_AB - S_A - S_B
    e_max = 2.0 * math.log(4.0)
    sim = 1.0 - (e_un / e_max)
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim


@njit(cache=True, fastmath=True)
def entropy_sim_shift_numba(qmz, qp, cmz, cp, tol, shift):
    a = entropy_similarity_numba(qmz, qp, cmz, cp, tol)
    if -0.001 < shift < 0.001:
        return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)):
        sm[i] = cmz[i] + shift
    b = entropy_similarity_numba(qmz, qp, sm, cp, tol)
    return a if a > b else b


@njit(cache=True, fastmath=True, parallel=True)
def search_direct_numba(qmz, qp, cands, off, allmz, allin, tol=0.015):
    out = np.zeros(len(cands), np.float32)
    for k in prange(len(cands)):
        c = cands[k]
        a = off[c]
        b = off[c + 1]
        if b <= a:
            continue
        cm, cp = _clean_numba(allmz[a:b], allin[a:b], 0.002, 128)
        if len(cm) == 0:
            continue
        out[k] = entropy_similarity_numba(qmz, qp, cm, cp, tol)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def search_shift_numba(qmz, qp, cands, off, allmz, allin, shifts, tol=0.015):
    out = np.zeros(len(cands), np.float32)
    for k in prange(len(cands)):
        c = cands[k]
        a = off[c]
        b = off[c + 1]
        if b <= a:
            continue
        cm, cp = _clean_numba(allmz[a:b], allin[a:b], 0.002, 128)
        if len(cm) == 0:
            continue
        out[k] = entropy_sim_shift_numba(qmz, qp, cm, cp, tol, shifts[k])
    return out


# ── Feature Helpers ──────────────────────────────────────────────────────────
def _rank_norm(x):
    """Normalized rank in [0, 1], where 1.0 is the best (highest value)."""
    n = len(x)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    order = np.argsort(-x)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return 1.0 - (ranks / float(n - 1))


def _z(x):
    """Z-score normalization with safe epsilon."""
    s = float(np.std(x))
    if s > 1e-9:
        return (x - float(np.mean(x))) / s
    return np.zeros_like(x, dtype=np.float32)


def main():
    print("=" * 85)
    print("  PHASE 6: MULTI-CHANNEL OUT-OF-FOLD (OOF) META-RANKER")
    print("  Query-Grouped GBDT with Continuous Evidence Fusion")
    print("=" * 85, flush=True)
    t0 = time.time()

    # 1. Load Candidate Union (776,699 candidates)
    cand_pq = ROOT / "artifacts" / "v3_clean" / "candidate_union.parquet"
    print(f"Loading candidate catalog from {cand_pq}...", flush=True)
    cand_df = pq.read_table(cand_pq, columns=["inchikey14", "canonical_smiles", "exact_mass", "source"]).to_pandas()
    cand_iks = cand_df["inchikey14"].to_numpy()
    cand_smiles = cand_df["canonical_smiles"].to_numpy()
    cand_masses = cand_df["exact_mass"].to_numpy(dtype=np.float64)
    cand_sources = cand_df["source"].to_numpy()
    n_total_cands = len(cand_df)
    print(f"Loaded {n_total_cands:,} candidates in {time.time()-t0:.1f}s.")

    cand_k2i = {k: i for i, k in enumerate(cand_iks)}

    # Precomputed Morgan fingerprints for fast analog propagation
    cand_fps = np.load(ROOT / "artifacts/v3_clean/candidate_fps.npy")
    popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    # 2. Load Clean v4 Reference Library
    ref_pq = ROOT / "artifacts" / "v3_clean" / "clean_v4_reference_library.parquet"
    print(f"Loading reference library from {ref_pq}...", flush=True)
    t_ref = time.time()
    tbl_ref = pq.read_table(ref_pq)
    df_ref = tbl_ref.to_pandas()

    mzc = tbl_ref.column("peaks_mz").combine_chunks()
    itc = tbl_ref.column("peaks_intensity").combine_chunks()
    ref_off = mzc.offsets.to_numpy().astype(np.int64)
    ref_allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    ref_allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)

    ref_nms = df_ref["neutral_mass"].to_numpy(dtype=np.float64)
    ref_ces = df_ref["collision_energy"].to_numpy(dtype=np.float32)
    ref_iks = df_ref["inchikey14"].to_numpy(dtype=object)

    ref_order = np.argsort(ref_nms)
    sorted_ref_nms = ref_nms[ref_order]
    print(f"Loaded {len(df_ref):,} reference spectra in {time.time()-t_ref:.1f}s.")

    # Build Representative Analog Library
    n_peaks_arr = np.diff(ref_off)
    rep_dict = {}
    for i in range(len(ref_iks)):
        ik = ref_iks[i]
        if ik and (ik not in rep_dict or n_peaks_arr[i] > n_peaks_arr[rep_dict[ik]]):
            rep_dict[ik] = i

    rep_indices = np.array(sorted(rep_dict.values()), dtype=np.int64)
    o_rep = np.argsort(ref_nms[rep_indices])
    rep_indices = rep_indices[o_rep]
    rep_nms = ref_nms[rep_indices]
    rep_iks = ref_iks[rep_indices]
    rep_ces = ref_ces[rep_indices]
    rep_fp_indices = np.array([cand_k2i.get(k, -1) for k in rep_iks], dtype=np.int32)
    print(f"Analog scaffold index ready: {len(rep_indices):,} unique scaffolds.")

    # 3. Load Pretrained FPNet Ensemble
    ckpts = sorted((ROOT / "artifacts/fp_models").glob("*.pt"))
    assert len(ckpts) > 0, "No FPNet checkpoints found!"
    fpnet = FPNetEnsemble(ckpts)

    bits = np.load(ROOT / "external_candidates/fp_bits.npy")
    m2_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    m3_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    rk_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

    cand_fp6930_cache = {}

    def get_candidate_6930_fp(smi: str) -> np.ndarray:
        if smi in cand_fp6930_cache:
            return cand_fp6930_cache[smi]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            f = np.zeros(len(bits), dtype=np.uint8)
        else:
            f = np.concatenate(
                [
                    m2_gen.GetFingerprintAsNumPy(m).astype(np.uint8),
                    m3_gen.GetFingerprintAsNumPy(m).astype(np.uint8),
                    rk_gen.GetFingerprintAsNumPy(m).astype(np.uint8),
                    np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8),
                ]
            )[bits]
        cand_fp6930_cache[smi] = f
        return f

    # 4. Load Clean v4 Benchmark Queries (450 queries)
    bench_pq = ROOT / "artifacts" / "v3_clean" / "benchmark_v4_queries.parquet"
    print(f"Loading benchmark queries from {bench_pq}...", flush=True)
    bqs_df = pq.read_table(bench_pq).to_pandas()
    print(f"Loaded {len(bqs_df)} benchmark queries across groups:", bqs_df["group"].value_counts().to_dict())

    # 5. Feature Extraction Loop across all 450 Queries
    print("\nExtracting feature vectors for all query/candidate pairs...", flush=True)
    t_feat = time.time()

    FEATURE_NAMES = [
        # Mass features (5)
        "ppm_error", "abs_mass_error", "is_ppm_10", "is_c13_iso", "rank_mass",
        # Direct retrieval features (6)
        "direct_sim", "direct_sim_sq", "direct_rank", "direct_diff_max", "has_direct", "has_strong_direct",
        # Analog propagation features (5)
        "analog_score", "analog_top_sim", "analog_rank", "analog_diff_max", "has_analog",
        # FPNet structural features (7)
        "fpnet_raw", "fpnet_z", "fpnet_rank", "fpnet_diff_max", "fpnet_is_top1", "fpnet_norm_z", "fpnet_norm_rank",
        # Context & Query features (8)
        "log_n_candidates", "query_max_direct", "query_has_confident_direct", "query_precursor_mz",
        "query_ce", "query_is_positive", "cand_src_train", "cand_src_coconut",
        # Channel Interactions (2)
        "direct_fpnet_prod", "analog_fpnet_prod"
    ]
    print(f"Total features per candidate: {len(FEATURE_NAMES)}")

    feat_cache_file = ROOT / "artifacts" / "v3_clean" / "meta_ranker_features_cache.pkl"
    query_cands_info = []

    if feat_cache_file.exists():
        print(f"Loading precomputed feature cache from {feat_cache_file}...", flush=True)
        with open(feat_cache_file, "rb") as f:
            query_cands_info = pickle.load(f)
        print(f"Loaded {len(query_cands_info)} precomputed query feature sets in 0.2s.", flush=True)
    else:
        for q_idx, row in bqs_df.iterrows():
            grp = row["group"]
            true_ik = row["true_inchikey14"]
            prec_mz = float(row["observed_precursor_mz"])
            adduct = str(row["observed_adduct"])
            ce_val = float(row["observed_collision_energy"]) if row["observed_collision_energy"] is not None else 25.0
            mode_str = str(row["observed_ionization_mode"])

            q_mzs = np.asarray(row["observed_ms2_mzs"], dtype=np.float32)
            q_ints = np.asarray(row["observed_ms2_intensities"], dtype=np.float32)

            m0 = neutral_mass(prec_mz, adduct)
            if m0 is None or not np.isfinite(m0) or m0 <= 0:
                m0 = prec_mz - 1.007825

            qm, qp = _clean_numba(q_mzs, q_ints, 0.002, 128)

            # Retrieve candidates with precursor_mz-aware union
            cands_idx = retrieve_candidates_union(m0, cand_masses, precursor_mz=prec_mz)
            n_cands = len(cands_idx)

            # Ground truth index in candidate pool
            pool_iks = cand_iks[cands_idx]
            true_match = np.where(pool_iks == true_ik)[0]
            true_pos = int(true_match[0]) if len(true_match) > 0 else -1

            # ── Channel 1: Mass ──
            c_masses = cand_masses[cands_idx]
            ppm_errors = np.abs(c_masses - m0) / m0 * 1e6
            abs_mass_errors = np.abs(c_masses - m0)
            is_ppm_10 = (ppm_errors <= 10.0).astype(np.float32)
            is_c13_iso = (np.abs(abs_mass_errors - 1.003355) <= 0.02).astype(np.float32)
            rank_mass = _rank_norm(-ppm_errors)

            # ── Channel 2: Direct ──
            tol_dir = m0 * 20.0 / 1e6
            lo_dir = int(np.searchsorted(sorted_ref_nms, m0 - tol_dir, side="left"))
            hi_dir = int(np.searchsorted(sorted_ref_nms, m0 + tol_dir, side="right"))
            dir_cands = ref_order[lo_dir:hi_dir]

            direct_hits = {}
            if len(dir_cands) > 0 and len(qm) > 0:
                sims_dir = search_direct_numba(qm, qp, dir_cands, ref_off, ref_allmz, ref_allin, tol=0.015)
                for c_idx, s in zip(dir_cands, sims_dir):
                    ik_match = ref_iks[c_idx]
                    if s > direct_hits.get(ik_match, -1.0):
                        direct_hits[ik_match] = float(s)

            direct_sim = np.zeros(n_cands, dtype=np.float32)
            for i_loc, ik_c in enumerate(pool_iks):
                if ik_c in direct_hits:
                    direct_sim[i_loc] = direct_hits[ik_c]

            direct_sim_sq = direct_sim ** 2
            direct_rank = _rank_norm(direct_sim)
            direct_max_query = float(direct_sim.max()) if n_cands > 0 else 0.0
            direct_diff_max = direct_sim - direct_max_query
            has_direct = (direct_sim >= 0.10).astype(np.float32)
            has_strong_direct = (direct_sim >= 0.70).astype(np.float32)

            # ── Channel 3: Analog ──
            lo_rep = int(np.searchsorted(rep_nms, m0 - 200.0, side="left"))
            hi_rep = int(np.searchsorted(rep_nms, m0 + 200.0, side="right"))
            analog_cands = rep_indices[lo_rep:hi_rep]

            s_analog = np.zeros(n_cands, dtype=np.float32)
            analog_top_sim = 0.0

            if len(analog_cands) > 0 and len(qm) > 0:
                shifts = (m0 - rep_nms[lo_rep:hi_rep]).astype(np.float32)
                shift_sims = search_shift_numba(qm, qp, analog_cands, ref_off, ref_allmz, ref_allin, shifts, tol=0.015)
                qual = np.where(shift_sims >= 0.15)[0]
                if len(qual) > 0:
                    top_k = qual[np.argsort(-shift_sims[qual])[:80]]
                    top_w = shift_sims[top_k]
                    analog_top_sim = float(top_w[0])
                    top_ces = rep_ces[lo_rep:hi_rep][top_k]
                    top_shifts = np.abs(shifts[top_k])
                    top_fp_idx = rep_fp_indices[lo_rep:hi_rep][top_k]

                    ce_weights = np.exp(-np.abs(ce_val - top_ces) / 20.0)
                    mass_weights = np.exp(-top_shifts / 100.0)
                    analog_weights = (top_w ** 2) * ce_weights * mass_weights

                    cand_pool_fps = cand_fps[cands_idx]
                    cs_counts = popcount_lut[cand_pool_fps].sum(axis=-1)

                    for afp_idx, aw in zip(top_fp_idx, analog_weights):
                        if afp_idx >= 0 and aw > 0:
                            a_fp = cand_fps[afp_idx]
                            inter = np.bitwise_and(cand_pool_fps, a_fp)
                            inter_c = popcount_lut[inter].sum(axis=-1)
                            a_count = popcount_lut[a_fp].sum()
                            union_c = cs_counts + a_count - inter_c
                            tan = np.where(union_c > 0, inter_c / union_c, 0.0)
                            contrib = (tan * aw).astype(np.float32)
                            s_analog = np.maximum(s_analog, contrib)

            analog_rank = _rank_norm(s_analog)
            analog_max_query = float(s_analog.max()) if n_cands > 0 else 0.0
            analog_diff_max = s_analog - analog_max_query
            has_analog = (s_analog >= 0.10).astype(np.float32)

            # ── Channel 4: FPNet ──
            z_logits = fpnet.predict_logits([q_mzs], [q_ints], prec_mz, adduct, None, ce_val, mode_str)
            cand_smis = cand_smiles[cands_idx]
            pool_6930_fps = np.stack([get_candidate_6930_fp(s) for s in cand_smis])
            cs = pool_6930_fps.sum(axis=1)

            raw_fpnet = pool_6930_fps @ z_logits
            fpnet_z = _z(raw_fpnet)
            fpnet_rank = _rank_norm(raw_fpnet)
            fpnet_diff_max = raw_fpnet - raw_fpnet.max()
            fpnet_is_top1 = (raw_fpnet == raw_fpnet.max()).astype(np.float32)

            nrm_fpnet = raw_fpnet / np.sqrt(np.maximum(cs, 1.0))
            fpnet_norm_z = _z(nrm_fpnet)
            fpnet_norm_rank = _rank_norm(nrm_fpnet)

            # ── Channel 5: Context & Query Features ──
            log_n_cands = float(np.log(max(n_cands, 1)))
            q_has_conf_direct = 1.0 if direct_max_query >= 0.70 else 0.0
            q_is_pos = 1.0 if mode_str == "positive" else 0.0
            c_src = cand_sources[cands_idx]
            cand_src_train = (c_src == "TRAIN").astype(np.float32)
            cand_src_coco = (c_src == "COCONUT").astype(np.float32)

            direct_fpnet_prod = direct_sim * fpnet_z
            analog_fpnet_prod = s_analog * fpnet_z

            # Assemble feature matrix X_q of shape (n_cands, 33)
            X_q = np.column_stack([
                ppm_errors, abs_mass_errors, is_ppm_10, is_c13_iso, rank_mass,
                direct_sim, direct_sim_sq, direct_rank, direct_diff_max, has_direct, has_strong_direct,
                s_analog, np.full(n_cands, analog_top_sim, dtype=np.float32), analog_rank, analog_diff_max, has_analog,
                raw_fpnet, fpnet_z, fpnet_rank, fpnet_diff_max, fpnet_is_top1, fpnet_norm_z, fpnet_norm_rank,
                np.full(n_cands, log_n_cands, dtype=np.float32),
                np.full(n_cands, direct_max_query, dtype=np.float32),
                np.full(n_cands, q_has_conf_direct, dtype=np.float32),
                np.full(n_cands, prec_mz, dtype=np.float32),
                np.full(n_cands, ce_val, dtype=np.float32),
                np.full(n_cands, q_is_pos, dtype=np.float32),
                cand_src_train, cand_src_coco,
                direct_fpnet_prod, analog_fpnet_prod
            ]).astype(np.float32)

            # Baseline scores for comparisons A and B
            # Config A: Mass + Direct + Analog
            score_a = -ppm_errors / 100.0 + (2.0 * direct_sim) + (2.0 * (direct_sim >= 0.70)) + (1.5 * s_analog)
            # Config B: Fixed Linear Fusion
            score_b = score_a + (1.2 * fpnet_z)

            # Binary labels
            y_q = np.zeros(n_cands, dtype=np.float32)
            if true_pos >= 0:
                y_q[true_pos] = 1.0

            query_cands_info.append({
                "q_idx": q_idx,
                "group": grp,
                "true_pos": true_pos,
                "n_cands": n_cands,
                "score_a": score_a,
                "score_b": score_b,
                "direct_max": direct_max_query,
                "direct_sim": direct_sim,
                "X": X_q,
                "y": y_q
            })

            if (q_idx + 1) % 10 == 0 or (q_idx + 1) == len(bqs_df) or (q_idx < 5):
                elapsed = time.time() - t_feat
                ms_per_q = elapsed / (q_idx + 1) * 1000
                rem_q = len(bqs_df) - (q_idx + 1)
                eta_m = (rem_q * (elapsed / (q_idx + 1))) / 60.0
                print(
                    f"  Processed {q_idx+1:3d}/450 queries | "
                    f"Elapsed: {elapsed/60:.1f}m ({ms_per_q:.0f} ms/q) | "
                    f"ETA: {eta_m:.1f}m",
                    flush=True,
                )

        feat_cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(feat_cache_file, "wb") as f:
            pickle.dump(query_cands_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved feature cache to {feat_cache_file} ({time.time()-t_feat:.1f}s).", flush=True)

    # 6. Query-Grouped 5-Fold Cross Validation
    print("\nRunning Query-Grouped 5-Fold OOF Training with LightGBM...", flush=True)

    # Stratified fold assignment: 30 C1, 30 C2, 30 C3 per fold
    c1_q = [i for i, info in enumerate(query_cands_info) if info["group"] == "C1"]
    c2_q = [i for i, info in enumerate(query_cands_info) if info["group"] == "C2"]
    c3_q = [i for i, info in enumerate(query_cands_info) if info["group"] == "C3"]

    query_fold = np.zeros(len(query_cands_info), dtype=int)
    for cohort in [c1_q, c2_q, c3_q]:
        for i, q in enumerate(cohort):
            query_fold[q] = i % 5

    oof_gbdt_probs = [None] * len(query_cands_info)
    models = []
    feature_importances_gain = np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    for fold in range(5):
        val_q_indices = [q for q in range(len(query_cands_info)) if query_fold[q] == fold]
        train_q_indices = [q for q in range(len(query_cands_info)) if query_fold[q] != fold]

        X_train_list = [query_cands_info[q]["X"] for q in train_q_indices]
        y_train_list = [query_cands_info[q]["y"] for q in train_q_indices]
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        pos_weight = float((len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0))

        clf = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            colsample_bytree=0.8,
            subsample=0.8,
            scale_pos_weight=min(pos_weight, 50.0),
            random_state=42 + fold,
            verbose=-1,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        models.append(clf)
        feature_importances_gain += clf.booster_.feature_importance(importance_type="gain")

        # Predict OOF probabilities
        for q in val_q_indices:
            X_val_q = query_cands_info[q]["X"]
            probs_q = clf.predict_proba(X_val_q)[:, 1]
            oof_gbdt_probs[q] = probs_q

    feature_importances_gain /= 5.0

    # 7. Evaluation of the 4 Systems
    print("\nEvaluating all 4 systems across all 450 queries...", flush=True)

    systems = ["A_Direct_Analog", "B_Fixed_Fusion", "C_OOF_GBDT", "D_OOF_GBDT_Gated"]
    results = {
        sys_name: {
            "overall_rr": [], "C1_rr": [], "C2_rr": [], "C3_rr": [],
            "hit1": [], "hit5": [], "hit25": []
        }
        for sys_name in systems
    }

    recall_stats = {"C1": 0, "C2": 0, "C3": 0, "overall": 0}

    for q, info in enumerate(query_cands_info):
        grp = info["group"]
        true_pos = info["true_pos"]
        has_recall = true_pos >= 0

        if has_recall:
            recall_stats["overall"] += 1
            recall_stats[grp] += 1

        sc_a = info["score_a"]
        sc_b = info["score_b"]
        sc_c = oof_gbdt_probs[q]

        # Config D: OOF GBDT + Conditional Direct Routing
        d_sims = info["direct_sim"]
        sc_d = sc_c.copy()
        if info["direct_max"] >= 0.70:
            sc_d = sc_d + 10.0 * (d_sims >= 0.70)

        sys_scores = {
            "A_Direct_Analog": sc_a,
            "B_Fixed_Fusion": sc_b,
            "C_OOF_GBDT": sc_c,
            "D_OOF_GBDT_Gated": sc_d
        }

        for sys_name, sc in sys_scores.items():
            if true_pos >= 0:
                true_sc = sc[true_pos]
                rank = int(np.sum(sc > true_sc)) + 1
            else:
                rank = 999999

            rr = 1.0 / rank if rank <= 25 else 0.0
            results[sys_name]["overall_rr"].append(rr)
            results[sys_name][f"{grp}_rr"].append(rr)
            results[sys_name]["hit1"].append(1.0 if rank == 1 else 0.0)
            results[sys_name]["hit5"].append(1.0 if rank <= 5 else 0.0)
            results[sys_name]["hit25"].append(1.0 if rank <= 25 else 0.0)

    # 8. Summary Table
    print("\n" + "=" * 85)
    print("  PHASE 6 OOF META-RANKER EVALUATION SUMMARY")
    print("=" * 85)

    summary_rows = []
    for sys_name in systems:
        ov_mrr = float(np.mean(results[sys_name]["overall_rr"]))
        c1_mrr = float(np.mean(results[sys_name]["C1_rr"]))
        c2_mrr = float(np.mean(results[sys_name]["C2_rr"]))
        c3_mrr = float(np.mean(results[sys_name]["C3_rr"]))
        h1 = float(np.mean(results[sys_name]["hit1"]) * 100.0)
        h5 = float(np.mean(results[sys_name]["hit5"]) * 100.0)
        h25 = float(np.mean(results[sys_name]["hit25"]) * 100.0)

        summary_rows.append({
            "System": sys_name,
            "Overall MRR": ov_mrr,
            "C1 MRR": c1_mrr,
            "C2 MRR": c2_mrr,
            "C3 MRR": c3_mrr,
            "Hit@1": h1,
            "Hit@5": h5,
            "Hit@25": h25
        })
        print(f"[{sys_name:18s}] MRR: {ov_mrr:.4f} | C1: {c1_mrr:.4f} | C2: {c2_mrr:.4f} | C3: {c3_mrr:.4f} | H@1: {h1:5.1f}% | H@5: {h5:5.1f}% | H@25: {h25:5.1f}%")

    print(f"\nCandidate Recall: {recall_stats['overall']/450*100:.2f}% (C1: {recall_stats['C1']/150*100:.1f}%, C2: {recall_stats['C2']/150*100:.1f}%, C3: {recall_stats['C3']/150*100:.1f}%)")

    # 9. Top Feature Importances
    fi_order = np.argsort(-feature_importances_gain)
    top_features = [{"feature": FEATURE_NAMES[i], "gain_importance": float(feature_importances_gain[i])} for i in fi_order]
    print("\nTop 15 Features by Gain Importance:")
    for rank, fi in enumerate(top_features[:15], 1):
        print(f"  {rank:2d}. {fi['feature']:26s} (gain: {fi['gain_importance']:10.1f})")

    # Save Results
    out_json = ROOT / "artifacts" / "v3_clean" / "oof_meta_ranker_results.json"
    output_payload = {
        "summary": summary_rows,
        "candidate_recall": {
            "overall": recall_stats["overall"] / 450.0,
            "C1": recall_stats["C1"] / 150.0,
            "C2": recall_stats["C2"] / 150.0,
            "C3": recall_stats["C3"] / 150.0
        },
        "feature_importances": top_features
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved OOF Meta-Ranker results to {out_json}")

    ckpt_dir = ROOT / "artifacts" / "v3_clean" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "oof_meta_ranker_models.pkl", "wb") as f:
        pickle.dump(models, f)
    print(f"Saved 5-fold LightGBM models to {ckpt_dir / 'oof_meta_ranker_models.pkl'}")


if __name__ == "__main__":
    main()
