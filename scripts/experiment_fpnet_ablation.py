"""Phase 5 Step 3 Experiment: 5-Way Ablation (Mass vs Direct vs Analog vs FPNet vs Ensemble).

Evaluates 5 clean configurations on the repaired Clean v4 Benchmark:
  Configuration A: Mass + Prior
  Configuration B: Mass + Direct Spectral Retrieval
  Configuration C: Mass + Direct + Mass-Shifted Analog Propagation
  Configuration D: FPNet Only (CSI:FingerID-style Spectrum-to-Fingerprint Transformer)
  Configuration E: Direct + Analog + FPNet Full Fusion

Tracks:
  Overall MRR, C1 MRR, C2 MRR, C3 MRR, Hit@1, Hit@5, Hit@25, Candidate Recall.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit, prange
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

from src.core.candidate_retrieval import retrieve_candidates_union
from src.core.preprocessing_v3 import neutral_mass
from src.models.fpnet import FPNetEnsemble, score_candidates_fpnet

DEFAULT_OUT_DIR = ROOT / "artifacts" / "v4_clean"
RETRIEVAL_PARAMETERS = {
    "ppm_windows": [20.0, 50.0, 100.0],
    "use_c13_isotopes": True,
    "c13_ppm": 30.0,
    "nominal_tol": 0.5,
    "early_stop": False,
    "union": True,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, str | int]:
    return {"path": str(path), "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _provenance(out_dir: Path) -> dict[str, object]:
    split = out_dir / "clean_v4_split.json"
    reference = out_dir / "clean_v4_reference_library.parquet"
    candidate = out_dir / "candidate_union.parquet"
    query = out_dir / "benchmark_v4_queries.parquet"
    manifest = out_dir / "candidate_manifest.json"
    split_data = {}
    if split.exists():
        try:
            with open(split, "r", encoding="utf-8") as handle:
                split_data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            split_data = {}
    if not candidate.exists():
        candidate_value = split_data.get("candidate_catalog", {}).get("path", "")
        if candidate_value and Path(candidate_value).exists():
            candidate = Path(candidate_value)
    if not manifest.exists() and candidate.exists():
        manifest = candidate.parent / "candidate_manifest.json"
    return {
        "split": _artifact_record(split) if split.exists() else {},
        "reference": _artifact_record(reference) if reference.exists() else {},
        "candidate": _artifact_record(candidate) if candidate.exists() else {},
        "queries": _artifact_record(query) if query.exists() else {},
        "candidate_manifest": _artifact_record(manifest) if manifest.exists() else {},
    }


def _resolve_out_dir(value: str | Path | None) -> Path:
    if value is None:
        return DEFAULT_OUT_DIR
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


# ── Numba Fast Spectral Kernels ───────────────────────────────────────────────
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
        order = np.argsort(vals)[cnt - topk :]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[order[i]]
        k2.sort()
        idx = k2
        cnt = topk

    om = np.empty(cnt, np.float32)
    oi = np.empty(cnt, np.float32)
    s = 0.0
    for i in range(cnt):
        om[i] = mz[idx[i]]
        v = it[idx[i]]
        oi[i] = v
        s += v
    if s > 0:
        for i in range(cnt):
            oi[i] /= s

    # Entropy weighting for low-entropy spectra
    S = 0.0
    for i in range(cnt):
        if oi[i] > 0:
            S -= oi[i] * np.log(oi[i])
    if S < 3.0:
        w = 0.25 + 0.25 * S
        s2 = 0.0
        for i in range(cnt):
            oi[i] = oi[i] ** w
            s2 += oi[i]
        if s2 > 0:
            for i in range(cnt):
                oi[i] /= s2

    return om, oi


@njit(cache=True, fastmath=True)
def entropy_sim_numba(qmz, qp, cmz, cp, tol):
    i = 0
    j = 0
    n = len(qmz)
    m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    SAB = 0.0
    tot = 0.0
    buf = np.empty(n + m, np.float64)
    b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]
            i += 1
            b += 1
        elif d > tol:
            buf[b] = cp[j]
            j += 1
            b += 1
        else:
            buf[b] = qp[i] + cp[j]
            i += 1
            j += 1
            b += 1
    while i < n:
        buf[b] = qp[i]
        i += 1
        b += 1
    while j < m:
        buf[b] = cp[j]
        j += 1
        b += 1
    for x in range(b):
        tot += buf[x]
    if tot <= 0:
        return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0:
            SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True)
def entropy_sim_shift_numba(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim_numba(qmz, qp, cmz, cp, tol)
    if -0.001 < shift < 0.001:
        return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)):
        sm[i] = cmz[i] + shift
    b = entropy_sim_numba(qmz, qp, sm, cp, tol)
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
        out[k] = entropy_sim_numba(qmz, qp, cm, cp, tol)
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


def _z(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def run_5way_ablation(output_dir: str | Path | None = None):
    t0 = time.time()
    out_dir = _resolve_out_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("5-WAY ABLATION: Mass vs Direct vs Analog vs FPNet vs Full Ensemble")
    print("=" * 80)

    print("[1/5] Loading Candidate Catalog...", flush=True)
    cand_path = out_dir / "candidate_union.parquet"
    cand_df = pq.read_table(
        cand_path, columns=["inchikey", "exact_mass", "source", "canonical_smiles"]
    ).to_pandas()
    cand_masses = cand_df["exact_mass"].to_numpy(dtype=np.float64)
    cand_iks = cand_df["inchikey"].astype(str).to_numpy()
    cand_smiles = cand_df["canonical_smiles"].to_numpy()
    cand_sources = cand_df["source"].to_numpy()
    cand_k2i = {}
    for index, key in enumerate(cand_iks):
        cand_k2i.setdefault(key, index)
    cand_fps = np.load(out_dir / "candidate_fps.npy")
    popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    print("[2/5] Loading Clean Reference Library...", flush=True)
    ref_path = out_dir / "clean_v4_reference_library.parquet"
    tbl_ref = pq.read_table(ref_path)
    df_ref = tbl_ref.to_pandas()

    mzc = tbl_ref.column("peaks_mz").combine_chunks()
    itc = tbl_ref.column("peaks_intensity").combine_chunks()
    ref_off = mzc.offsets.to_numpy().astype(np.int64)
    ref_allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    ref_allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)

    ref_nms = df_ref["neutral_mass"].to_numpy(dtype=np.float64)
    ref_ces = df_ref["collision_energy"].to_numpy(dtype=np.float32)
    ref_iks = df_ref["inchikey"].astype(str).to_numpy(dtype=object)

    ref_order = np.argsort(ref_nms)
    sorted_ref_nms = ref_nms[ref_order]
    print(f"      Indexed {len(df_ref):,} clean reference spectra.", flush=True)

    # Build Representative Analog Library
    print("      Building representative scaffold library for analog search...", flush=True)
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

    # Sort reps by neutral mass
    o_rep = np.argsort(rep_nms)
    rep_indices = rep_indices[o_rep]
    rep_nms = rep_nms[o_rep]
    rep_iks = rep_iks[o_rep]
    rep_ces = rep_ces[o_rep]
    rep_fp_indices = np.array([cand_k2i.get(k, -1) for k in rep_iks], dtype=np.int32)
    print(f"      Representative library ready: {len(rep_indices):,} unique scaffolds.", flush=True)

    # 3. Load Neural FPNet Ensemble
    print("[3/5] Loading Pretrained FPNet Ensemble...", flush=True)
    ckpts = sorted((ROOT / "artifacts/fp_models").glob("*.pt"))
    assert len(ckpts) > 0, "No FPNet checkpoints found in artifacts/fp_models!"
    fpnet = FPNetEnsemble(ckpts)

    # FPNet bit definitions & RDKit generators
    bits = np.load(ROOT / "external_candidates/fp_bits.npy")
    m2_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    m3_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    rk_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

    # On-the-fly candidate fingerprint cache
    cand_fp6930_cache: dict[str, np.ndarray] = {}

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

    # 4. Load Clean Benchmark Queries
    print("[4/5] Loading Clean v4 Benchmark Queries...", flush=True)
    bqs_path = out_dir / "benchmark_v4_queries.parquet"
    bqs_df = pq.read_table(bqs_path).to_pandas()
    print(f"      Benchmark queries loaded: {len(bqs_df)} rows across groups:", bqs_df["group"].value_counts().to_dict())

    # Evaluation containers for 5 configurations
    configs = ["A_Mass", "B_Mass_Direct", "C_Direct_Analog", "D_FPNet_Only", "E_Full_Ensemble"]
    reciprocal_ranks = {cfg: defaultdict(list) for cfg in configs}
    hit1 = {cfg: defaultdict(list) for cfg in configs}
    hit5 = {cfg: defaultdict(list) for cfg in configs}
    hit25 = {cfg: defaultdict(list) for cfg in configs}
    hits_recall = defaultdict(list)

    print(f"\n[5/5] Executing 5-Way Ablation across {len(bqs_df)} Benchmark Queries...", flush=True)
    t_eval = time.time()

    for q_idx, row in bqs_df.iterrows():
        grp = row["group"]
        true_ik = row["true_inchikey"]
        prec_mz = float(row["observed_precursor_mz"])
        adduct = str(row["observed_adduct"])
        ce_val = float(row["observed_collision_energy"]) if row["observed_collision_energy"] is not None else 25.0
        mode_str = str(row["observed_ionization_mode"])

        q_mzs = np.asarray(row["observed_ms2_mzs"], dtype=np.float32)
        q_ints = np.asarray(row["observed_ms2_intensities"], dtype=np.float32)

        # 1. Unprivileged neutral mass derivation
        m0 = neutral_mass(prec_mz, adduct)
        if m0 is None or not np.isfinite(m0) or m0 <= 0:
            m0 = prec_mz - 1.007825

        # 2. Candidate Retrieval Union (No early stopping, 20/50/100 ppm + 13C isotope + nominal)
        cands_idx = retrieve_candidates_union(
            m0,
            cand_masses,
            ppm_windows=tuple(RETRIEVAL_PARAMETERS["ppm_windows"]),
            use_c13_isotopes=RETRIEVAL_PARAMETERS["use_c13_isotopes"],
            c13_ppm=RETRIEVAL_PARAMETERS["c13_ppm"],
            precursor_mz=prec_mz,
            nominal_tol=RETRIEVAL_PARAMETERS["nominal_tol"],
        )
        n_cands = len(cands_idx)

        # Candidate recall tracking
        pool_iks = cand_iks[cands_idx]
        found = true_ik in set(pool_iks)
        hits_recall["overall"].append(1.0 if found else 0.0)
        hits_recall[grp].append(1.0 if found else 0.0)

        if not found or n_cands == 0:
            for cfg in configs:
                reciprocal_ranks[cfg]["overall"].append(0.0)
                reciprocal_ranks[cfg][grp].append(0.0)
                hit1[cfg]["overall"].append(0.0)
                hit1[cfg][grp].append(0.0)
                hit5[cfg]["overall"].append(0.0)
                hit5[cfg][grp].append(0.0)
                hit25[cfg]["overall"].append(0.0)
                hit25[cfg][grp].append(0.0)
            continue

        true_pos = np.where(pool_iks == true_ik)[0][0]

        # Clean query spectrum for Numba searches
        qm, qp = _clean_numba(q_mzs, q_ints, 0.002, 128)

        # -------------------------------------------------------------
        # CONFIGURATION A: Mass + Source Prior
        # -------------------------------------------------------------
        cand_pool_masses = cand_masses[cands_idx]
        ppm_errors = np.abs(cand_pool_masses - m0) / m0 * 1e6

        cand_pool_sources = cand_sources[cands_idx]
        source_priors = np.zeros(n_cands, dtype=np.float32)
        for i_c, src in enumerate(cand_pool_sources):
            if src == "TRAIN":
                source_priors[i_c] = 0.05
            elif src == "COCONUT":
                source_priors[i_c] = 0.02

        score_a = -ppm_errors / 100.0 + source_priors

        # -------------------------------------------------------------
        # CONFIGURATION B: Mass + Direct Spectral Retrieval
        # -------------------------------------------------------------
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

        s_direct = np.zeros(n_cands, dtype=np.float32)
        for i_local, ik_cand in enumerate(pool_iks):
            if ik_cand in direct_hits:
                d_sim = direct_hits[ik_cand]
                if d_sim >= 0.10:
                    s_direct[i_local] = 2.0 * d_sim
                    if d_sim >= 0.70:
                        s_direct[i_local] += 2.0  # Decisive high-confidence boost

        score_b = score_a + s_direct

        # -------------------------------------------------------------
        # CONFIGURATION C: Mass + Direct + Mass-Shifted Analog Propagation
        # -------------------------------------------------------------
        lo_rep = int(np.searchsorted(rep_nms, m0 - 200.0, side="left"))
        hi_rep = int(np.searchsorted(rep_nms, m0 + 200.0, side="right"))
        analog_cands = rep_indices[lo_rep:hi_rep]

        s_analog = np.zeros(n_cands, dtype=np.float32)
        if len(analog_cands) > 0 and len(qm) > 0:
            shifts = (m0 - rep_nms[lo_rep:hi_rep]).astype(np.float32)
            shift_sims = search_shift_numba(qm, qp, analog_cands, ref_off, ref_allmz, ref_allin, shifts, tol=0.015)

            qualifying = np.where(shift_sims >= 0.15)[0]
            if len(qualifying) > 0:
                top_k_qual = qualifying[np.argsort(-shift_sims[qualifying])[:80]]
                top_analog_sims = shift_sims[top_k_qual]
                top_analog_ces = rep_ces[lo_rep:hi_rep][top_k_qual]
                top_analog_shifts = np.abs(shifts[top_k_qual])
                top_analog_fp_idx = rep_fp_indices[lo_rep:hi_rep][top_k_qual]

                ce_weights = np.exp(-np.abs(ce_val - top_analog_ces) / 20.0)
                mass_weights = np.exp(-top_analog_shifts / 100.0)
                analog_weights = (top_analog_sims**2) * ce_weights * mass_weights

                cand_pool_fps = cand_fps[cands_idx]  # (n_cands, 256)
                cs_counts = popcount_lut[cand_pool_fps].sum(axis=-1)

                for a_i, (afp_idx, aw) in enumerate(zip(top_analog_fp_idx, analog_weights)):
                    if afp_idx >= 0 and aw > 0:
                        a_fp = cand_fps[afp_idx]
                        inter = np.bitwise_and(cand_pool_fps, a_fp)
                        inter_c = popcount_lut[inter].sum(axis=-1)
                        a_count = popcount_lut[a_fp].sum()
                        union_c = cs_counts + a_count - inter_c
                        tan = np.where(union_c > 0, inter_c / union_c, 0.0)
                        contrib = (tan * aw).astype(np.float32)
                        s_analog = np.maximum(s_analog, contrib)

        score_c = score_b + 1.5 * s_analog

        # -------------------------------------------------------------
        # CONFIGURATION D: FPNet Only (6,930-bit Transformer)
        # -------------------------------------------------------------
        z_logits = fpnet.predict_logits(
            [q_mzs],
            [q_ints],
            prec_mz,
            adduct,
            instrument_type=None,
            ce_ev=ce_val,
            ionization_mode=mode_str,
        )

        cand_smis = cand_smiles[cands_idx]
        pool_6930_fps = np.stack([get_candidate_6930_fp(s) for s in cand_smis])  # (n_cands, 6930)

        # Normalized Bayes score
        score_fpnet_raw = score_candidates_fpnet(pool_6930_fps, z_logits, normalize=True)
        score_d = score_fpnet_raw

        # -------------------------------------------------------------
        # CONFIGURATION E: Direct + Analog + FPNet Full Fusion
        # -------------------------------------------------------------
        # Z-score the FPNet signal across the retrieved candidates
        z_fpnet = _z(score_fpnet_raw)

        # Calibrated fusion:
        # - High direct hit (>= 0.70) remains decisive (+4.0 boost)
        # - Analog evidence provides scaffold consistency (+1.5 * s_analog)
        # - FPNet provides strong structural likelihood (+1.2 * z_fpnet)
        # - Mass prior breaks degenerate ties
        score_e = score_a + s_direct + 1.5 * s_analog + 1.2 * z_fpnet

        # -------------------------------------------------------------
        # Rank evaluation across all configurations
        # -------------------------------------------------------------
        all_scores = {
            "A_Mass": score_a,
            "B_Mass_Direct": score_b,
            "C_Direct_Analog": score_c,
            "D_FPNet_Only": score_d,
            "E_Full_Ensemble": score_e,
        }

        for cfg, sc in all_scores.items():
            true_score = sc[true_pos]
            rank = int(np.sum(sc > true_score)) + 1
            rr = 1.0 / rank if rank <= 25 else 0.0
            reciprocal_ranks[cfg]["overall"].append(rr)
            reciprocal_ranks[cfg][grp].append(rr)
            hit1[cfg]["overall"].append(1.0 if rank == 1 else 0.0)
            hit1[cfg][grp].append(1.0 if rank == 1 else 0.0)
            hit5[cfg]["overall"].append(1.0 if rank <= 5 else 0.0)
            hit5[cfg][grp].append(1.0 if rank <= 5 else 0.0)
            hit25[cfg]["overall"].append(1.0 if rank <= 25 else 0.0)
            hit25[cfg][grp].append(1.0 if rank <= 25 else 0.0)

        if (q_idx + 1) % 50 == 0 or (q_idx + 1) == len(bqs_df):
            elapsed = time.time() - t_eval
            curr_c = _mean(reciprocal_ranks["C_Direct_Analog"]["overall"])
            curr_d = _mean(reciprocal_ranks["D_FPNet_Only"]["overall"])
            curr_e = _mean(reciprocal_ranks["E_Full_Ensemble"]["overall"])
            print(
                f"  Evaluated {q_idx+1:3d}/{len(bqs_df)} queries | "
                f"C_MRR: {curr_c:.4f} | D_MRR: {curr_d:.4f} | E_MRR: {curr_e:.4f} | "
                f"Elapsed: {elapsed:.1f}s ({elapsed/(q_idx+1)*1000:.1f} ms/q)",
                flush=True,
            )

    # -----------------------------------------------------------------
    # Aggregate and Print Final Results
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5-WAY ABLATION BENCHMARK RESULTS")
    print("=" * 80)

    summary_rows = []
    for cfg in configs:
        row_dict = {
            "System": cfg,
            "Overall MRR": _mean(reciprocal_ranks[cfg]["overall"]),
            "C1 MRR": _mean(reciprocal_ranks[cfg]["C1"]),
            "C2 MRR": _mean(reciprocal_ranks[cfg]["C2"]),
            "C3 MRR": _mean(reciprocal_ranks[cfg]["C3"]),
            "Hit@1": _mean(hit1[cfg]["overall"]) * 100.0,
            "Hit@5": _mean(hit5[cfg]["overall"]) * 100.0,
            "Hit@25": _mean(hit25[cfg]["overall"]) * 100.0,
        }
        summary_rows.append(row_dict)

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCandidate Recall Gate:")
    for grp in ["C1", "C2", "C3", "overall"]:
        rec = _mean(hits_recall[grp]) * 100.0
        n_tot = len(hits_recall[grp])
        n_hit = int(np.sum(hits_recall[grp]))
        print(f"  {grp:8s}: {rec:6.2f}% ({n_hit}/{n_tot})")

    # Save Results as JSON and Markdown Artifact
    provenance = _provenance(out_dir)
    res_payload = {
        "summary": summary_rows,
        "candidate_recall": {grp: _mean(hits_recall[grp]) for grp in ["C1", "C2", "C3", "overall"]},
        "molecule_key": "inchikey",
        "retrieval_parameters": RETRIEVAL_PARAMETERS,
        "provenance": provenance,
        "split_sha256": provenance["split"].get("sha256", ""),
        "reference_sha256": provenance["reference"].get("sha256", ""),
        "candidate_sha256": provenance["candidate"].get("sha256", ""),
    }
    with open(out_dir / "fpnet_ablation_results.json", "w") as f:
        json.dump(res_payload, f, indent=2)

    print(f"\n[SUCCESS] Ablation completed in {time.time()-t0:.1f}s.")
    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)
    args = parser.parse_args()
    run_5way_ablation(args.output_dir)
