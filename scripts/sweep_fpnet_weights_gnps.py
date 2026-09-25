"""Phase C: Comprehensive FPNet Weight & Normalization Sweep on 50-Molecule External GNPS Cohort.

Mirrors exact verified pipeline from verify_meta_ranker_three_tests.py (Test B).
Saves intermediate scores and tests:
  w_fpnet in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0]
across 4 normalization functions.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

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
import pyarrow.parquet as pq
import torch
from numba import njit, prange
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

from src.core.candidate_retrieval import retrieve_candidates_union
from src.core.preprocessing_v3 import neutral_mass
from src.models.fpnet import FPNetEnsemble, score_candidates_fpnet


# Numba Spectral Kernels matching verify_meta_ranker_three_tests.py
@njit(cache=True, fastmath=True)
def _clean_numba(mz, it, floor=0.002, topk=128):
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
def entropy_similarity_numba(m1, i1, m2, i2, tol=0.015):
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

    sim = 1.0 - (2.0 * S_AB - S_A - S_B) / 1.3862943611198906
    return max(0.0, min(1.0, sim))


@njit(cache=True, fastmath=True)
def entropy_sim_shift_numba(qmz, qp, cmz, cp, tol=0.015, shift=0.0):
    a = entropy_similarity_numba(qmz, qp, cmz, cp, tol)
    if abs(shift) < 0.005:
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


def _z(x):
    s = float(np.std(x))
    if s > 1e-9:
        return (x - float(np.mean(x))) / s
    return np.zeros_like(x, dtype=np.float32)


def compute_metrics(ranks):
    rr = [1.0 / r if r <= 25 else 0.0 for r in ranks]
    mrr = float(np.mean(rr))
    h1 = float(np.mean([1.0 if r == 1 else 0.0 for r in ranks]) * 100)
    h5 = float(np.mean([1.0 if r <= 5 else 0.0 for r in ranks]) * 100)
    h25 = float(np.mean([1.0 if r <= 25 else 0.0 for r in ranks]) * 100)
    return {"mrr": mrr, "hit1": h1, "hit5": h5, "hit25": h25}


def main():
    print("=" * 80)
    print("  PHASE C: SYSTEMATIC FPNet WEIGHT & NORMALIZATION SWEEP ON EXTERNAL GNPS")
    print("=" * 80)

    # 1. Load train InChIKey14s for strict quarantine verification
    print("[1/5] Auditing train InChIKey14 set...", flush=True)
    train_iks = set(
        pq.read_table(ROOT / "dataset/train.parquet", columns=["inchikey14"])
        .to_pandas()["inchikey14"]
        .dropna()
        .unique()
    )
    print(f"      Competition train molecules: {len(train_iks):,}")

    # 2. Load Candidates & Precomputed Data
    print("[2/5] Loading 776k candidate catalog and representations...", flush=True)
    cand_df = pq.read_table(
        ROOT / "artifacts/v3_clean/candidate_union.parquet",
        columns=["inchikey14", "canonical_smiles", "exact_mass", "source"],
    ).to_pandas()
    cand_iks = cand_df["inchikey14"].to_numpy()
    cand_smiles = cand_df["canonical_smiles"].to_numpy()
    cand_masses = cand_df["exact_mass"].to_numpy(dtype=np.float64)
    cand_sources = cand_df["source"].to_numpy()
    cand_k2i = {k: i for i, k in enumerate(cand_iks)}
    cand_fps = np.load(ROOT / "artifacts/v3_clean/candidate_fps.npy")
    popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    # 3. Load Clean Reference Library
    print("[3/5] Loading Clean Reference Library for direct & analog search...", flush=True)
    tbl_ref = pq.read_table(ROOT / "artifacts/v3_clean/clean_v4_reference_library.parquet")
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

    # Build Representative Analog Library
    n_peaks_arr = np.diff(ref_off)
    rep_dict = {}
    for i in range(len(ref_iks)):
        ik = ref_iks[i]
        if ik and (ik not in rep_dict or n_peaks_arr[i] > n_peaks_arr[rep_dict[ik]]):
            rep_dict[ik] = i
    rep_indices = np.array(sorted(rep_dict.values()), dtype=np.int64)
    rep_nms = ref_nms[rep_indices]
    rep_iks = ref_iks[rep_indices]
    rep_ces = ref_ces[rep_indices]
    o_rep = np.argsort(rep_nms)
    rep_indices = rep_indices[o_rep]
    rep_nms = rep_nms[o_rep]
    rep_iks = rep_iks[o_rep]
    rep_ces = rep_ces[o_rep]
    rep_fp_indices = np.array([cand_k2i.get(k, -1) for k in rep_iks], dtype=np.int32)

    # 4. Load Neural FPNet Ensemble
    print("[4/5] Loading Neural FPNet Ensemble...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fpnet = FPNetEnsemble(
        [ROOT / "artifacts/fp_models/fp_single_aug.pt", ROOT / "artifacts/fp_models/fp_merged_m1.pt"],
        device=device,
    )
    bits = np.load(ROOT / "external_candidates/fp_bits.npy")
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    mfpgen3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)

    cand_fp6930_cache: dict[str, np.ndarray] = {}

    def get_candidate_6930_fp(smi: str) -> np.ndarray:
        if smi in cand_fp6930_cache:
            return cand_fp6930_cache[smi]
        m = Chem.MolFromSmiles(smi)
        if not m:
            fp_6930 = np.zeros(len(bits), dtype=np.float32)
        else:
            v1 = np.asarray(mfpgen.GetFingerprint(m), dtype=np.uint8)
            v2 = np.asarray(mfpgen3.GetFingerprint(m), dtype=np.uint8)
            v3 = np.asarray(rdkgen.GetFingerprint(m), dtype=np.uint8)
            v4 = np.asarray(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)
            full = np.concatenate([v1, v2, v3, v4])
            fp_6930 = full[bits].astype(np.float32)
        cand_fp6930_cache[smi] = fp_6930
        return fp_6930

    # 5. Extract 50 Strictly Novel GNPS Queries (0% Train Overlap)
    print("[5/5] Extracting 50 novel GNPS queries...", flush=True)
    df_gnps = pq.read_table(ROOT / "artifacts/external/gnps_spectra.parquet").to_pandas()
    gnps_evaluated = []
    for idx_row, row in df_gnps.iterrows():
        smi = row["canonical_smiles"]
        if not smi or pd.isna(smi):
            continue
        m = Chem.MolFromSmiles(smi)
        if not m:
            continue
        ik_full = Chem.MolToInchiKey(m)
        if not ik_full or len(ik_full) < 14:
            continue
        ik = ik_full[:14]

        # Strictest check: 0% overlap with competition train set
        if ik in train_iks or ik not in cand_k2i:
            continue

        p_mz = float(row["precursor_mz"])
        adduct = str(row.get("precursor_type", "[M+H]+"))
        if not adduct or adduct == "None" or adduct not in ["[M+H]+", "[M-H]-"]:
            adduct = "[M+H]+" if row.get("ion_mode", "P") == "P" else "[M-H]-"

        m0 = neutral_mass(p_mz, adduct)
        if m0 is None or not np.isfinite(m0) or m0 <= 0:
            m0 = p_mz - 1.007825

        true_m = float(Descriptors.ExactMolWt(m))
        if abs(m0 - true_m) > 0.05:
            continue

        cands_idx = retrieve_candidates_union(m0, cand_masses, precursor_mz=p_mz)
        if len(cands_idx) < 5 or cand_k2i[ik] not in cands_idx:
            continue

        q_mzs = np.asarray(row["peaks_mz"], dtype=np.float32)
        q_ints = np.asarray(row["peaks_intensity"], dtype=np.float32)
        if len(q_mzs) == 0:
            continue

        ce_val = float(row["collision_energy"]) if pd.notna(row.get("collision_energy")) and float(row["collision_energy"]) > 0 else 25.0
        mode_str = "positive" if row.get("ion_mode", "P") == "P" else "negative"

        gnps_evaluated.append({
            "ik": ik, "smi": smi, "p_mz": p_mz, "adduct": adduct, "m0": m0,
            "q_mzs": q_mzs, "q_ints": q_ints, "ce_val": ce_val, "mode_str": mode_str,
            "cands_idx": cands_idx
        })
        if len(gnps_evaluated) >= 50:
            break

    print(f"      Successfully isolated {len(gnps_evaluated)} unseen GNPS molecules (0% train overlap).")

    # Evaluate & Cache Candidate Scores for all 50 Queries
    print("Computing channels for 50 GNPS queries...", flush=True)
    gnps_records = []
    t_start_gnps = time.time()

    for i_q, q in enumerate(gnps_evaluated):
        ik = q["ik"]
        m0 = q["m0"]
        p_mz = q["p_mz"]
        ce_val = q["ce_val"]
        mode_str = q["mode_str"]
        cands_idx = q["cands_idx"]
        n_cands = len(cands_idx)
        pool_iks = cand_iks[cands_idx]
        true_pos = int(np.where(cands_idx == cand_k2i[ik])[0][0])

        qm, qp = _clean_numba(q["q_mzs"], q["q_ints"], 0.002, 128)

        # ── Channel 1: Mass & Source Prior ──
        c_masses = cand_masses[cands_idx]
        ppm_errors = np.abs(c_masses - m0) / m0 * 1e6
        cand_srcs = cand_sources[cands_idx]
        cand_src_train = (cand_srcs == "TRAIN").astype(np.float32)
        cand_src_coco = (cand_srcs == "COCONUT").astype(np.float32)
        source_priors = cand_src_train * 0.05 + cand_src_coco * 0.02

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

        # ── Channel 3: Analog ──
        lo_rep = int(np.searchsorted(rep_nms, m0 - 200.0, side="left"))
        hi_rep = int(np.searchsorted(rep_nms, m0 + 200.0, side="right"))
        analog_cands = rep_indices[lo_rep:hi_rep]

        s_analog = np.zeros(n_cands, dtype=np.float32)
        if len(analog_cands) > 0 and len(qm) > 0:
            shifts = (m0 - rep_nms[lo_rep:hi_rep]).astype(np.float32)
            shift_sims = search_shift_numba(qm, qp, analog_cands, ref_off, ref_allmz, ref_allin, shifts, tol=0.015)
            qual = np.where(shift_sims >= 0.15)[0]
            if len(qual) > 0:
                top_k = qual[np.argsort(-shift_sims[qual])[:80]]
                top_w = shift_sims[top_k]
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

        # ── Channel 4: FPNet ──
        z_logits = fpnet.predict_logits([q["q_mzs"]], [q["q_ints"]], p_mz, q["adduct"], None, ce_val, mode_str)
        cand_smis = cand_smiles[cands_idx]
        pool_6930_fps = np.stack([get_candidate_6930_fp(s) for s in cand_smis])
        cs = pool_6930_fps.sum(axis=1)

        score_fpnet_raw = score_candidates_fpnet(pool_6930_fps, z_logits, normalize=True)
        fpnet_z = _z(score_fpnet_raw)

        # ── Physical Baseline Score ──
        d_gated = np.where(direct_sim >= 0.10, 2.0 * direct_sim + np.where(direct_sim >= 0.70, 2.0, 0.0), 0.0)
        score_direct_analog = -ppm_errors / 100.0 + source_priors + d_gated + 1.5 * s_analog

        gnps_records.append({
            "true_pos": true_pos,
            "score_direct_analog": score_direct_analog,
            "score_fpnet_raw": score_fpnet_raw,
            "fpnet_z": fpnet_z,
            "cs": cs,
            "n_cands": n_cands
        })

        if (i_q + 1) % 10 == 0:
            print(f"  Processed {i_q+1}/50 queries in {time.time()-t_start_gnps:.1f}s...", flush=True)

    print(f"Precomputation complete for all 50 queries in {time.time()-t_start_gnps:.1f}s!\n")

    # =========================================================================
    # SYSTEMATIC SWEEPS ACROSS WEIGHTS & NORMALIZATIONS
    # =========================================================================
    weights = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.2, 1.5, 2.0]
    all_sweep_results = []

    # 1. Direct + Analog Baseline
    base_ranks = []
    for r in gnps_records:
        sc = r["score_direct_analog"]
        rank = int(np.sum(sc > sc[r["true_pos"]])) + 1
        base_ranks.append(rank)
    m_base = compute_metrics(base_ranks)
    print("=" * 80)
    print(f"PHYSICAL BASELINE (w_fpnet = 0.0, Direct + Analog):")
    print(f"  MRR: {m_base['mrr']:.4f} | Hit@1: {m_base['hit1']:5.1f}% | Hit@5: {m_base['hit5']:5.1f}% | Hit@25: {m_base['hit25']:5.1f}%")
    print("=" * 80)

    # 2. Standard Z-Score: (score - mu) / sigma
    print("\n--- Scheme 1: Standard Query-Level Z-Score: z = (dot - mu) / (sigma + eps) ---")
    for w in weights:
        ranks = []
        for r in gnps_records:
            sc = r["score_direct_analog"] + w * r["fpnet_z"]
            rank = int(np.sum(sc > sc[r["true_pos"]])) + 1
            ranks.append(rank)
        m = compute_metrics(ranks)
        all_sweep_results.append({"scheme": "standard_zscore", "w_fpnet": w, **m})
        print(f"  w_fpnet = {w:4.2f} -> MRR: {m['mrr']:.4f} | Hit@1: {m['hit1']:5.1f}% | Hit@5: {m['hit5']:5.1f}% | Hit@25: {m['hit25']:5.1f}%")

    # 3. Soft-Clipped Z-Score: sigma >= 0.5, clip [-2.5, 2.5]
    print("\n--- Scheme 2: Robust Clipped Z-Score (sigma floor = 0.5, clip [-2.5, 2.5]) ---")
    for w in weights:
        ranks = []
        for r in gnps_records:
            raw = r["score_fpnet_raw"]
            mu = float(np.mean(raw))
            sig = max(float(np.std(raw)), 0.5)
            z_clip = np.clip((raw - mu) / sig, -2.5, 2.5)
            sc = r["score_direct_analog"] + w * z_clip
            rank = int(np.sum(sc > sc[r["true_pos"]])) + 1
            ranks.append(rank)
        m = compute_metrics(ranks)
        all_sweep_results.append({"scheme": "clipped_zscore", "w_fpnet": w, **m})
        print(f"  w_fpnet = {w:4.2f} -> MRR: {m['mrr']:.4f} | Hit@1: {m['hit1']:5.1f}% | Hit@5: {m['hit5']:5.1f}% | Hit@25: {m['hit25']:5.1f}%")

    # 4. Bit-Size Normalized Score: raw / sqrt(max(bits, 1))
    print("\n--- Scheme 3: Bit-Size Normalized Score: (raw / sqrt(bits)) ---")
    for w in weights:
        ranks = []
        for r in gnps_records:
            raw = r["score_fpnet_raw"]
            cs = r["cs"]
            norm_raw = raw / np.sqrt(np.maximum(cs, 1.0))
            z_norm = _z(norm_raw)
            sc = r["score_direct_analog"] + w * z_norm
            rank = int(np.sum(sc > sc[r["true_pos"]])) + 1
            ranks.append(rank)
        m = compute_metrics(ranks)
        all_sweep_results.append({"scheme": "bit_norm_zscore", "w_fpnet": w, **m})
        print(f"  w_fpnet = {w:4.2f} -> MRR: {m['mrr']:.4f} | Hit@1: {m['hit1']:5.1f}% | Hit@5: {m['hit5']:5.1f}% | Hit@25: {m['hit25']:5.1f}%")

    # 5. Min-Max Scaling [0, 1]
    print("\n--- Scheme 4: Query Min-Max Scaling [0, 1] ---")
    for w in weights:
        ranks = []
        for r in gnps_records:
            raw = r["score_fpnet_raw"]
            mi = float(np.min(raw))
            ma = float(np.max(raw))
            rng = max(ma - mi, 1e-6)
            minmax = (raw - mi) / rng
            sc = r["score_direct_analog"] + w * minmax
            rank = int(np.sum(sc > sc[r["true_pos"]])) + 1
            ranks.append(rank)
        m = compute_metrics(ranks)
        all_sweep_results.append({"scheme": "minmax", "w_fpnet": w, **m})
        print(f"  w_fpnet = {w:4.2f} -> MRR: {m['mrr']:.4f} | Hit@1: {m['hit1']:5.1f}% | Hit@5: {m['hit5']:5.1f}% | Hit@25: {m['hit25']:5.1f}%")

    # Save complete results
    out_file = ROOT / "artifacts/v3_clean/fpnet_external_weight_sweep.json"
    with open(out_file, "w") as f:
        json.dump(all_sweep_results, f, indent=2)
    print(f"\nSaved empirical sweep results to {out_file}")


if __name__ == "__main__":
    main()
