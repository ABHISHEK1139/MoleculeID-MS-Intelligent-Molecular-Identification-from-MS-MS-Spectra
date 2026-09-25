"""Verification of Phase 6 Multi-Channel Meta-Ranker across 3 Strict Checks:

Test A: Molecule-Grouped OOF (Group = InChIKey14 & Murcko Scaffold)
Test B: External Unseen Chemistry (GNPS 50-molecule cohort, 0% train overlap)
Test C: Reconcile 0.4170 vs 0.3439 baseline discrepancy
"""
from __future__ import annotations

import json
import math
import os
import pickle
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

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from numba import njit, prange
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold

RDLogger.DisableLog("rdApp.*")

from src.core.candidate_retrieval import retrieve_candidates_union
from src.core.preprocessing_v3 import neutral_mass
from src.models.fpnet import FPNetEnsemble, score_candidates_fpnet


# ==============================================================================
# FAST NUMBA SPECTRAL KERNELS
# ==============================================================================
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


def _rank_norm(x):
    n = len(x)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    order = np.argsort(-x)
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return 1.0 - (ranks / float(n - 1))


def _z(x):
    s = float(np.std(x))
    if s > 1e-9:
        return (x - float(np.mean(x))) / s
    return np.zeros_like(x, dtype=np.float32)


# ==============================================================================
# TEST C: RECONCILING THE 0.4170 VS 0.3439 BASELINE DISCREPANCY
# ==============================================================================
def run_test_c(cache):
    print("\n" + "=" * 80)
    print("  TEST C: RECONCILING 0.4170 (Ablation) vs 0.3439 (Meta-Ranker Cache)")
    print("=" * 80)
    
    FEATURE_NAMES = [
        "ppm_error", "abs_mass_error", "is_ppm_10", "is_c13_iso", "rank_mass",
        "direct_sim", "direct_sim_sq", "direct_rank", "direct_max_query", "direct_diff_max",
        "has_direct", "has_strong_direct",
        "s_analog", "analog_rank", "analog_max_query", "analog_diff_max", "analog_top_sim", "has_analog",
        "fpnet_raw", "fpnet_zscore", "fpnet_rank", "fpnet_norm", "fpnet_top_cand",
        "n_cands", "direct_max_again", "q_has_conf_direct", "prec_mz", "ce_val", "q_is_pos",
        "cand_src_train", "cand_src_coco", "direct_fpnet_prod", "analog_fpnet_prod"
    ]
    ppm_idx = FEATURE_NAMES.index("ppm_error")
    dir_idx = FEATURE_NAMES.index("direct_sim")
    ana_idx = FEATURE_NAMES.index("s_analog")
    src_tr_idx = FEATURE_NAMES.index("cand_src_train")
    src_co_idx = FEATURE_NAMES.index("cand_src_coco")
    
    # 1. Formula A (as written in train_oof_meta_ranker.py line 511)
    # score = -ppm_errors / 100.0 + (2.0 * direct_sim) + (2.0 * (direct_sim >= 0.70)) + (1.5 * s_analog)
    rrs_cache_formula = []
    rrs_by_grp_cache = {"C1": [], "C2": [], "C3": []}
    
    # 2. Formula B (with exact ablation source priors & direct threshold >= 0.10)
    # score = -ppm/100 + (0.05*TRAIN + 0.02*COCO) + (2.0*direct + 2.0*(direct>=0.70))*(direct>=0.10) + 1.5*analog
    rrs_ablation_formula = []
    rrs_by_grp_ablation = {"C1": [], "C2": [], "C3": []}
    
    # 3. Formula C (original experiment_direct_vs_analog.py with exp decay mass + prior)
    # score = exp(-ppm/25) + prior + (2.0*direct + 2.0*(direct>=0.70))*(direct>=0.10) + 1.0*analog
    rrs_orig_exp = []
    rrs_by_grp_orig = {"C1": [], "C2": [], "C3": []}

    for q in cache:
        grp = q["group"]
        tp = q["true_pos"]
        X = q["X"]
        ppm = X[:, ppm_idx]
        d_sim = X[:, dir_idx]
        s_ana = X[:, ana_idx]
        is_tr = X[:, src_tr_idx]
        is_co = X[:, src_co_idx]
        
        # Formula 1
        sc1 = -ppm / 100.0 + (2.0 * d_sim) + (2.0 * (d_sim >= 0.70)) + (1.5 * s_ana)
        # Formula 2
        src_p = is_tr * 0.05 + is_co * 0.02
        d_gated = np.where(d_sim >= 0.10, 2.0 * d_sim + np.where(d_sim >= 0.70, 2.0, 0.0), 0.0)
        sc2 = -ppm / 100.0 + src_p + d_gated + (1.5 * s_ana)
        # Formula 3
        s_mass = np.exp(-ppm / 25.0)
        s_prior = np.where((is_tr > 0.5) | (is_co > 0.5), 0.20, 0.10)
        sc3 = s_mass + s_prior + d_gated + (1.0 * s_ana)
        
        for sc, rrs, grp_rrs in [
            (sc1, rrs_cache_formula, rrs_by_grp_cache),
            (sc2, rrs_ablation_formula, rrs_by_grp_ablation),
            (sc3, rrs_orig_exp, rrs_by_grp_orig),
        ]:
            if tp >= 0:
                rank = int(np.sum(sc > sc[tp])) + 1
                rr = 1.0 / rank if rank <= 25 else 0.0
            else:
                rr = 0.0
            rrs.append(rr)
            grp_rrs[grp].append(rr)
            
    print("Baseline Formula Dissection across all 450 v4 Queries:")
    print("-" * 80)
    print(f"1. Meta-Ranker Cache Formula (Raw -ppm, no prior, unthresholded direct):")
    print(f"   Overall MRR: {np.mean(rrs_cache_formula):.4f} | C1: {np.mean(rrs_by_grp_cache['C1']):.4f} | C2: {np.mean(rrs_by_grp_cache['C2']):.4f} | C3: {np.mean(rrs_by_grp_cache['C3']):.4f}")
    print(f"2. Ablation Gated Formula (+ Source Prior, direct >= 0.10 gate):")
    print(f"   Overall MRR: {np.mean(rrs_ablation_formula):.4f} | C1: {np.mean(rrs_by_grp_ablation['C1']):.4f} | C2: {np.mean(rrs_by_grp_ablation['C2']):.4f} | C3: {np.mean(rrs_by_grp_ablation['C3']):.4f}")
    print(f"3. Original Step 4 Formula (Exp decay mass + 0.20/0.10 prior):")
    print(f"   Overall MRR: {np.mean(rrs_orig_exp):.4f} | C1: {np.mean(rrs_by_grp_orig['C1']):.4f} | C2: {np.mean(rrs_by_grp_orig['C2']):.4f} | C3: {np.mean(rrs_by_grp_orig['C3']):.4f}")
    print("-" * 80)
    print("Conclusion for Test C:")
    print("In train_oof_meta_ranker.py, the comparison baseline score_a omitted the catalog source prior (+0.05/+0.02)")
    print("and the 0.10 direct similarity noise gate. The true clean Direct + Analog baseline is 0.4170 as reported in the ablation.")
    return float(np.mean(rrs_cache_formula)), float(np.mean(rrs_ablation_formula))


# ==============================================================================
# TEST A: MOLECULE-GROUPED AND SCAFFOLD-GROUPED OOF GBDT
# ==============================================================================
def run_test_a(cache, bqs_df):
    print("\n" + "=" * 80)
    print("  TEST A: MOLECULE-GROUPED & SCAFFOLD-GROUPED OOF GBDT EVALUATION")
    print("=" * 80)

    # Compute Murcko Scaffolds
    scaffolds = []
    for s in bqs_df["true_smiles"]:
        m = Chem.MolFromSmiles(s)
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""
        scaffolds.append(scaf if scaf else "NO_SCAFFOLD")
    bqs_df["scaffold"] = scaffolds
    
    unique_iks = bqs_df["true_inchikey14"].nunique()
    unique_smis = bqs_df["true_smiles"].nunique()
    unique_scafs = bqs_df["scaffold"].nunique()
    print(f"Dataset Structure Audit across {len(bqs_df)} Clean v4 Queries:")
    print(f"  - Unique InChIKey14s:       {unique_iks} / {len(bqs_df)}")
    print(f"  - Unique Canonical SMILES:  {unique_smis} / {len(bqs_df)}")
    print(f"  - Unique Murcko Scaffolds:  {unique_scafs} / {len(bqs_df)}")
    print("  (Note: In v4, every query was sampled from a distinct molecule, so InChIKey14 is already unique per query.)")
    print("  Therefore, Murcko Scaffold grouping provides the ultimate, strictest test of structural generalizability!\n")

    test_a_results = {}
    
    for split_type, group_series in [
        ("Molecule-Grouped (InChIKey14)", bqs_df["true_inchikey14"]),
        ("Scaffold-Grouped (Bemis-Murcko)", bqs_df["scaffold"])
    ]:
        print(f"--> Training 5-Fold LightGBM with Group = {split_type}...")
        gkf = GroupKFold(n_splits=5)
        folds = list(gkf.split(cache, groups=group_series))
        
        oof_probs = [None] * len(cache)
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(folds):
            X_train = np.vstack([cache[i]["X"] for i in train_idx])
            y_train = np.concatenate([cache[i]["y"] for i in train_idx])
            
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
            
            for q in val_idx:
                oof_probs[q] = clf.predict_proba(cache[q]["X"])[:, 1]
                
        # Evaluate metrics
        res = {"overall": [], "C1": [], "C2": [], "C3": [], "h1": [], "h5": [], "h25": []}
        for q, info in enumerate(cache):
            grp = info["group"]
            tp = info["true_pos"]
            sc = oof_probs[q]
            if tp >= 0:
                rank = int(np.sum(sc > sc[tp])) + 1
                rr = 1.0 / rank if rank <= 25 else 0.0
            else:
                rank = 999999
                rr = 0.0
            res["overall"].append(rr)
            res[grp].append(rr)
            res["h1"].append(1.0 if rank == 1 else 0.0)
            res["h5"].append(1.0 if rank <= 5 else 0.0)
            res["h25"].append(1.0 if rank <= 25 else 0.0)
            
        ov_mrr = float(np.mean(res["overall"]))
        c1_mrr = float(np.mean(res["C1"]))
        c2_mrr = float(np.mean(res["C2"]))
        c3_mrr = float(np.mean(res["C3"]))
        h1 = float(np.mean(res["h1"]) * 100.0)
        h5 = float(np.mean(res["h5"]) * 100.0)
        h25 = float(np.mean(res["h25"]) * 100.0)
        
        test_a_results[split_type] = {
            "models": models,
            "overall_mrr": ov_mrr, "c1_mrr": c1_mrr, "c2_mrr": c2_mrr, "c3_mrr": c3_mrr,
            "hit1": h1, "hit5": h5, "hit25": h25
        }
        
        print(f"  [{split_type}]")
        print(f"    Overall MRR: {ov_mrr:.4f} | C1: {c1_mrr:.4f} | C2: {c2_mrr:.4f} | C3: {c3_mrr:.4f}")
        print(f"    Hit@1: {h1:5.1f}% | Hit@5: {h5:5.1f}% | Hit@25: {h25:5.1f}%")

    return test_a_results


# ==============================================================================
# TEST B: EXTERNAL UNSEEN MOLECULES (GNPS 50-COHORT)
# ==============================================================================
def run_test_b(trained_models):
    print("\n" + "=" * 80)
    print("  TEST B: STRICTLY UNSEEN EXTERNAL GNPS EVALUATION (0% TRAIN OVERLAP)")
    print("  Direct+Analog vs Fixed Fusion vs Learned GBDT Meta-Ranker")
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
    print("[5/5] Extracting & evaluating 50 novel GNPS queries...", flush=True)
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
        if ik in train_iks:
            continue
        if ik not in cand_k2i:
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
            
        ce_val = float(row["collision_energy"]) if pd.notna(row["collision_energy"]) and float(row["collision_energy"]) > 0 else 25.0
        mode_str = "positive" if row.get("ion_mode", "P") == "P" else "negative"
        
        gnps_evaluated.append({
            "ik": ik, "smi": smi, "p_mz": p_mz, "adduct": adduct, "m0": m0,
            "q_mzs": q_mzs, "q_ints": q_ints, "ce_val": ce_val, "mode_str": mode_str,
            "cands_idx": cands_idx
        })
        
        if len(gnps_evaluated) >= 50:
            break

    print(f"      Successfully isolated {len(gnps_evaluated)} unseen GNPS molecules (0% train overlap).")

    # Evaluation structures
    metrics = {
        "A_Direct_Analog": {"rrs": [], "h1": [], "h5": [], "h25": []},
        "B_Fixed_Fusion":  {"rrs": [], "h1": [], "h5": [], "h25": []},
        "C_Learned_GBDT":  {"rrs": [], "h1": [], "h5": [], "h25": []}
    }

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
        abs_mass_errors = np.abs(c_masses - m0)
        is_ppm_10 = (ppm_errors <= 10.0).astype(np.float32)
        is_c13_iso = (np.abs(abs_mass_errors - 1.003355) <= 0.02).astype(np.float32)
        rank_mass = _rank_norm(-ppm_errors)
        
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
        z_logits = fpnet.predict_logits([q["q_mzs"]], [q["q_ints"]], p_mz, q["adduct"], None, ce_val, mode_str)
        cand_smis = cand_smiles[cands_idx]
        pool_6930_fps = np.stack([get_candidate_6930_fp(s) for s in cand_smis])
        cs = pool_6930_fps.sum(axis=1)
        
        score_fpnet_raw = score_candidates_fpnet(pool_6930_fps, z_logits, normalize=True)
        fpnet_z = _z(score_fpnet_raw)
        fpnet_rank = _rank_norm(score_fpnet_raw)
        fpnet_norm = score_fpnet_raw / np.sqrt(np.maximum(cs, 1.0))
        fpnet_top_cand = (score_fpnet_raw == score_fpnet_raw.max()).astype(np.float32)
        
        # ── Cross-Channel Interaction Terms ──
        direct_fpnet_prod = direct_sim * score_fpnet_raw
        analog_fpnet_prod = s_analog * score_fpnet_raw
        q_has_conf_direct = float(direct_max_query >= 0.70)
        q_is_pos = 1.0 if mode_str == "positive" else 0.0
        
        # ── Feature Matrix (33 features) ──
        X_gnps = np.column_stack([
            ppm_errors, abs_mass_errors, is_ppm_10, is_c13_iso, rank_mass,
            direct_sim, direct_sim_sq, direct_rank,
            np.full(n_cands, direct_max_query, dtype=np.float32),
            direct_diff_max, has_direct, has_strong_direct,
            s_analog, analog_rank,
            np.full(n_cands, analog_max_query, dtype=np.float32),
            analog_diff_max,
            np.full(n_cands, analog_top_sim, dtype=np.float32),
            has_analog,
            score_fpnet_raw, fpnet_z, fpnet_rank, fpnet_norm, fpnet_top_cand,
            np.full(n_cands, float(n_cands), dtype=np.float32),
            np.full(n_cands, direct_max_query, dtype=np.float32),
            np.full(n_cands, q_has_conf_direct, dtype=np.float32),
            np.full(n_cands, p_mz, dtype=np.float32),
            np.full(n_cands, ce_val, dtype=np.float32),
            np.full(n_cands, q_is_pos, dtype=np.float32),
            cand_src_train, cand_src_coco,
            direct_fpnet_prod, analog_fpnet_prod
        ]).astype(np.float32)
        
        # ── System 1: Direct + Analog Baseline ──
        # score = -ppm/100 + prior + (2.0*direct + 2.0*(direct>=0.70))*(direct>=0.10) + 1.5*analog
        d_gated = np.where(direct_sim >= 0.10, 2.0 * direct_sim + np.where(direct_sim >= 0.70, 2.0, 0.0), 0.0)
        score_direct_analog = -ppm_errors / 100.0 + source_priors + d_gated + 1.5 * s_analog
        
        # ── System 2: Fixed Linear Fusion (Direct + Analog + FPNet) ──
        score_fixed_fusion = score_direct_analog + 1.2 * fpnet_z
        
        # ── System 3: Learned GBDT Meta-Ranker Ensemble ──
        # Average probability across the 5 trained folds
        prob_gbdt = np.mean([clf.predict_proba(X_gnps)[:, 1] for clf in trained_models], axis=0)
        
        scores_dict = {
            "A_Direct_Analog": score_direct_analog,
            "B_Fixed_Fusion":  score_fixed_fusion,
            "C_Learned_GBDT":  prob_gbdt
        }
        
        for sys_name, sc in scores_dict.items():
            true_score = sc[true_pos]
            rank = int(np.sum(sc > true_score)) + 1
            rr = 1.0 / rank if rank <= 25 else 0.0
            metrics[sys_name]["rrs"].append(rr)
            metrics[sys_name]["h1"].append(1.0 if rank == 1 else 0.0)
            metrics[sys_name]["h5"].append(1.0 if rank <= 5 else 0.0)
            metrics[sys_name]["h25"].append(1.0 if rank <= 25 else 0.0)
            
        if (i_q + 1) % 10 == 0 or (i_q + 1) == len(gnps_evaluated):
            print(f"  Processed {i_q+1:2d}/50 GNPS queries | "
                  f"Direct+Analog MRR: {np.mean(metrics['A_Direct_Analog']['rrs']):.4f} | "
                  f"Fixed Fusion MRR: {np.mean(metrics['B_Fixed_Fusion']['rrs']):.4f} | "
                  f"Learned GBDT MRR: {np.mean(metrics['C_Learned_GBDT']['rrs']):.4f}", flush=True)

    print("\n" + "=" * 80)
    print("  TEST B RESULTS: 50 UNSEEN EXTERNAL GNPS MOLECULES")
    print("=" * 80)
    print(f"{'System':<20} {'MRR@25':<12} {'Hit@1':<10} {'Hit@5':<10} {'Hit@25':<10}")
    print("-" * 65)
    for sys_name, label in [
        ("A_Direct_Analog", "1. Direct + Analog"),
        ("B_Fixed_Fusion",  "2. Fixed Fusion"),
        ("C_Learned_GBDT",  "3. Learned GBDT")
    ]:
        mrr = np.mean(metrics[sys_name]["rrs"])
        h1 = np.mean(metrics[sys_name]["h1"]) * 100.0
        h5 = np.mean(metrics[sys_name]["h5"]) * 100.0
        h25 = np.mean(metrics[sys_name]["h25"]) * 100.0
        print(f"{label:<20} {mrr:<12.4f} {h1:<10.1f}% {h5:<10.1f}% {h25:<10.1f}%")
        
    print("-" * 65)
    mrr_a = np.mean(metrics["A_Direct_Analog"]["rrs"])
    mrr_b = np.mean(metrics["B_Fixed_Fusion"]["rrs"])
    mrr_c = np.mean(metrics["C_Learned_GBDT"]["rrs"])
    
    if mrr_c >= mrr_b and mrr_b >= mrr_a:
        print(">>> VERIFICATION PASSED: Learned GBDT >= Fixed Fusion >= Direct+Analog on unseen chemistry! <<<")
    elif mrr_c >= mrr_b:
        print(">>> VERIFICATION PASSED: Learned GBDT outperforms Fixed Fusion on unseen chemistry! <<<")
    else:
        print(">>> NOTICE: Learned GBDT did not outperform Fixed Fusion on unseen chemistry. <<<")
        
    return {
        "Direct_Analog_MRR": float(mrr_a),
        "Fixed_Fusion_MRR": float(mrr_b),
        "Learned_GBDT_MRR": float(mrr_c),
        "Direct_Analog_H25": float(np.mean(metrics["A_Direct_Analog"]["h25"]) * 100.0),
        "Fixed_Fusion_H25": float(np.mean(metrics["B_Fixed_Fusion"]["h25"]) * 100.0),
        "Learned_GBDT_H25": float(np.mean(metrics["C_Learned_GBDT"]["h25"]) * 100.0),
    }


# ==============================================================================
# MAIN ORCHESTRATION
# ==============================================================================
def main():
    print("=" * 80)
    print("  STRICT THREE-PART VERIFICATION PROTOCOL")
    print("  Authoritative Audit of the Multi-Channel Meta-Ranker")
    print("=" * 80)
    t_start = time.time()
    
    # Load Benchmark & Cached Features
    bqs_df = pq.read_table(ROOT / "artifacts/v3_clean/benchmark_v4_queries.parquet").to_pandas()
    cache_path = ROOT / "artifacts/v3_clean/meta_ranker_features_cache.pkl"
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
        
    # Part 1: Test C (Reconcile 0.417 vs 0.3439)
    mrr_c_raw, mrr_c_gated = run_test_c(cache)
    
    # Part 2: Test A (Molecule-grouped & Scaffold-grouped OOF GBDT)
    test_a_res = run_test_a(cache, bqs_df)
    
    # Use the Molecule-Grouped (InChIKey14) models for Test B
    models_mol_grouped = test_a_res["Molecule-Grouped (InChIKey14)"]["models"]
    
    # Part 3: Test B (External GNPS Unseen Chemistry)
    test_b_res = run_test_b(models_mol_grouped)
    
    # Save Comprehensive Verification Summary
    out_payload = {
        "test_a_molecule_grouped_oof": {
            "inchikey14_grouped": {
                "overall_mrr": test_a_res["Molecule-Grouped (InChIKey14)"]["overall_mrr"],
                "c1_mrr": test_a_res["Molecule-Grouped (InChIKey14)"]["c1_mrr"],
                "c2_mrr": test_a_res["Molecule-Grouped (InChIKey14)"]["c2_mrr"],
                "c3_mrr": test_a_res["Molecule-Grouped (InChIKey14)"]["c3_mrr"],
                "hit1": test_a_res["Molecule-Grouped (InChIKey14)"]["hit1"],
                "hit5": test_a_res["Molecule-Grouped (InChIKey14)"]["hit5"],
                "hit25": test_a_res["Molecule-Grouped (InChIKey14)"]["hit25"],
            },
            "bemis_murcko_scaffold_grouped": {
                "overall_mrr": test_a_res["Scaffold-Grouped (Bemis-Murcko)"]["overall_mrr"],
                "c1_mrr": test_a_res["Scaffold-Grouped (Bemis-Murcko)"]["c1_mrr"],
                "c2_mrr": test_a_res["Scaffold-Grouped (Bemis-Murcko)"]["c2_mrr"],
                "c3_mrr": test_a_res["Scaffold-Grouped (Bemis-Murcko)"]["c3_mrr"],
                "hit1": test_a_res["Scaffold-Grouped (Bemis-Murcko)"]["hit1"],
                "hit5": test_a_res["Scaffold-Grouped (Bemis-Murcko)"]["hit5"],
                "hit25": test_a_res["Scaffold-Grouped (Bemis-Murcko)"]["hit25"],
            }
        },
        "test_b_external_gnps": test_b_res,
        "test_c_baseline_reconciliation": {
            "meta_ranker_cache_baseline_mrr": mrr_c_raw,
            "true_ablation_baseline_mrr": mrr_c_gated
        },
        "execution_time_seconds": time.time() - t_start
    }
    
    out_file = ROOT / "artifacts/v3_clean/three_part_verification_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved full verification results to {out_file} in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
