"""Phase 4 Step 4 Experiment: Mass vs Direct vs Direct + Analog.

Evaluates 3 clean configurations on the repaired Clean v4 Benchmark:
  Configuration A: Mass + Source Prior
  Configuration B: Mass + Direct Spectral Retrieval (full InChIKey matched)
  Configuration C: Mass + Direct + Mass-Shifted Analog Propagation

Analog parameters (per specification):
  - +-200 Da mass shift window
  - Top 80-100 analogs
  - Spectral shift score (entropy_sim_shift)
  - Morgan Tanimoto similarity to candidate pool
  - Delta-mass weight
  - Polarity matching
  - CE compatibility (exp(-|delta_CE| / 20.0))
  - Soft feature (NOT hard override)

Reports:
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
import pyarrow.parquet as pq
from numba import njit, prange

from src.core.candidate_retrieval import retrieve_candidates_union
from src.core.preprocessing_v3 import neutral_mass

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
    if cnt == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    om = np.empty(cnt, np.float32)
    oi = np.empty(cnt, np.float32)
    pos = 0
    for i in range(n):
        if it[i] >= thresh:
            om[pos] = mz[i]
            oi[pos] = it[i]
            pos += 1

    if cnt > topk:
        idx = np.argsort(oi)[::-1][:topk]
        idx = np.sort(idx)
        om = om[idx]
        oi = oi[idx]

    s = np.sum(oi)
    if s > 0:
        oi = oi / s
    return om, oi


@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol=0.015):
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
    SAB = 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0:
            SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if -0.001 < shift < 0.001:
        return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)):
        sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b


@njit(cache=True, fastmath=True, parallel=True)
def search_direct_numba(qmz, qp, cand, off, allmz, allin, tol=0.015, floor=0.002, topk=256):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]
        a = off[c]
        b = off[c + 1]
        if b <= a:
            continue
        cm, cp = _clean_numba(allmz[a:b], allin[a:b], floor, topk)
        if len(cm) == 0:
            continue
        out[k] = entropy_sim(qmz, qp, cm, cp, tol)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def search_shift_numba(qmz, qp, cand, off, allmz, allin, shift, tol=0.015, floor=0.002, topk=256):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]
        a = off[c]
        b = off[c + 1]
        if b <= a:
            continue
        cm, cp = _clean_numba(allmz[a:b], allin[a:b], floor, topk)
        if len(cm) == 0:
            continue
        out[k] = entropy_sim_shift(qmz, qp, cm, cp, tol, shift[k])
    return out


def main(output_dir: str | Path | None = None):
    print("=" * 85)
    print("  EXPERIMENT: MASS VS DIRECT VS DIRECT + ANALOG PROPAGATION")
    print("  Quarantine: Clean v4 Benchmark (full InChIKey identity)")
    print("=" * 85, flush=True)

    out_dir = _resolve_out_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_pq = out_dir / "candidate_union.parquet"
    fps_npy = out_dir / "candidate_fps.npy"

    print(f"Loading candidate catalog from {cand_pq}...", flush=True)
    df_cand = pq.read_table(cand_pq, columns=["inchikey", "canonical_smiles", "exact_mass", "source"]).to_pandas()
    cand_masses = df_cand["exact_mass"].to_numpy(dtype=np.float64)
    cand_iks = df_cand["inchikey"].astype(str).to_numpy()
    cand_sources = df_cand["source"].to_numpy()
    cand_fps = np.load(fps_npy)
    cand_ik_to_idx = {}
    for index, key in enumerate(cand_iks):
        cand_ik_to_idx.setdefault(key, index)
    print(f"Loaded {len(df_cand):,} candidates ({cand_fps.nbytes / 1e6:.1f} MB fingerprints).")

    # 2. Load Clean v4 Reference Library
    ref_pq = out_dir / "clean_v4_reference_library.parquet"
    print(f"Loading clean reference library from {ref_pq}...", flush=True)
    tbl_ref = pq.read_table(ref_pq)
    df_ref = tbl_ref.to_pandas()

    mzc = tbl_ref.column("peaks_mz").combine_chunks()
    itc = tbl_ref.column("peaks_intensity").combine_chunks()
    ref_off = mzc.offsets.to_numpy().astype(np.int64)
    ref_allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    ref_allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)

    ref_nms = df_ref["neutral_mass"].to_numpy(dtype=np.float64)
    ref_ces = df_ref["collision_energy"].to_numpy(dtype=np.float32)
    ref_iks = df_ref["inchikey"].astype(str).to_numpy(dtype=object)

    # Sort reference library by neutral mass for fast binary window searches
    ref_order = np.argsort(ref_nms)
    sorted_ref_nms = ref_nms[ref_order]

    print(f"Indexed {len(df_ref):,} clean reference spectra.")

    # 3. Build Analog Representative Index (single richest spectrum per full InChIKey)
    print("Building analog scaffold representatives index...", flush=True)
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

    rep_order = np.argsort(rep_nms)
    rep_indices = rep_indices[rep_order]
    rep_nms = rep_nms[rep_order]
    rep_iks = rep_iks[rep_order]
    rep_ces = rep_ces[rep_order]

    # Map representative full InChIKey to candidate fingerprint index for fast vector bitwise Tanimoto
    rep_fp_indices = np.array([cand_ik_to_idx.get(ik, -1) for ik in rep_iks], dtype=np.int64)
    print(f"Analog index ready: {len(rep_indices):,} unique scaffold representatives.")

    # Popcount LUT for fast bitwise Tanimoto
    popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    # 4. Load Benchmark Queries
    queries_pq = out_dir / "benchmark_v4_queries.parquet"
    print(f"Loading benchmark queries from {queries_pq}...", flush=True)
    df_queries = pq.read_table(queries_pq).to_pandas()
    n_queries = len(df_queries)
    print(f"Loaded {n_queries} benchmark queries (C1/C2/C3).")

    # Accumulators for the 3 configurations
    configs = ["A_MassPrior", "B_MassDirect", "C_MassDirectAnalog"]
    recips = {cfg: [] for cfg in configs}
    recips_grp = {cfg: defaultdict(list) for cfg in configs}
    hits_1 = {cfg: [] for cfg in configs}
    hits_5 = {cfg: [] for cfg in configs}
    hits_25 = {cfg: [] for cfg in configs}

    hits_recall = []
    hits_recall_grp = defaultdict(list)

    print("\nRunning Back-to-Back Evaluation Across All 3 Configurations...", flush=True)
    t_eval = time.time()

    for qi in range(n_queries):
        row = df_queries.iloc[qi]
        grp = row["group"]
        true_ik = row["true_inchikey"]

        prec_mz = float(row["observed_precursor_mz"])
        adduct = str(row["observed_adduct"])
        ce_val = float(row["observed_collision_energy"])
        q_mzs = np.asarray(row["observed_ms2_mzs"], dtype=np.float32)
        q_ints = np.asarray(row["observed_ms2_intensities"], dtype=np.float32)

        # 1. Unprivileged neutral mass derivation
        m0 = neutral_mass(prec_mz, adduct)
        if m0 is None or not np.isfinite(m0) or m0 <= 0:
            m0 = prec_mz - 1.007825

        # 2. Candidate Retrieval Union (No early stopping, 20/50/100 ppm + 13C isotope)
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
        hits_recall.append(1.0 if found else 0.0)
        hits_recall_grp[grp].append(1.0 if found else 0.0)

        if n_cands == 0:
            for cfg in configs:
                recips[cfg].append(0.0)
                recips_grp[cfg][grp].append(0.0)
                hits_1[cfg].append(0.0)
                hits_5[cfg].append(0.0)
                hits_25[cfg].append(0.0)
            continue

        # Clean query spectrum for Numba kernels
        qm, qp = _clean_numba(q_mzs, q_ints, floor=0.002, topk=256)

        # -------------------------------------------------------------
        # CONFIGURATION A: Mass + Prior
        # -------------------------------------------------------------
        cand_sub_masses = cand_masses[cands_idx]
        cand_sub_sources = cand_sources[cands_idx]

        ppm_errs = np.abs(cand_sub_masses - m0) / m0 * 1e6
        s_mass = np.exp(-ppm_errs / 25.0).astype(np.float32)
        s_prior = np.where(np.isin(cand_sub_sources, ["TRAIN", "COCONUT"]), 0.20, 0.10).astype(np.float32)
        score_a = s_mass + s_prior

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

        # Attach direct score to candidate pool via full InChIKey
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

            # Filter top 80 analogs with shift similarity >= 0.15
            qualifying = np.where(shift_sims >= 0.15)[0]
            if len(qualifying) > 0:
                top_k_qual = qualifying[np.argsort(-shift_sims[qualifying])[:80]]
                top_analog_sims = shift_sims[top_k_qual]
                top_analog_ces = rep_ces[lo_rep:hi_rep][top_k_qual]
                top_analog_shifts = np.abs(shifts[top_k_qual])
                top_analog_fp_idx = rep_fp_indices[lo_rep:hi_rep][top_k_qual]

                # CE compatibility and mass shift decay weights
                ce_weights = np.exp(-np.abs(ce_val - top_analog_ces) / 20.0)
                mass_weights = np.exp(-top_analog_shifts / 100.0)
                analog_weights = (top_analog_sims ** 2) * ce_weights * mass_weights

                # Vectorized bitwise Tanimoto against candidate pool
                cand_pool_fps = cand_fps[cands_idx]  # (n_cands, 256)
                cs_counts = popcount_lut[cand_pool_fps].sum(axis=-1)  # (n_cands,)

                for a_i, (afp_idx, aw) in enumerate(zip(top_analog_fp_idx, analog_weights)):
                    if afp_idx >= 0 and aw > 0:
                        a_fp = cand_fps[afp_idx]  # (256,)
                        inter = np.bitwise_and(cand_pool_fps, a_fp)
                        inter_c = popcount_lut[inter].sum(axis=-1)
                        a_count = popcount_lut[a_fp].sum()
                        union_c = cs_counts + a_count - inter_c
                        tan = np.where(union_c > 0, inter_c / union_c, 0.0)
                        contrib = (tan * aw).astype(np.float32)
                        s_analog = np.maximum(s_analog, contrib)

        score_c = score_b + 1.0 * s_analog

        # -------------------------------------------------------------
        # Rank and Accumulate Metrics
        # -------------------------------------------------------------
        for cfg, sc in [("A_MassPrior", score_a), ("B_MassDirect", score_b), ("C_MassDirectAnalog", score_c)]:
            ranked_indices = np.argsort(-sc)[:25]
            ranked_iks = [pool_iks[idx] for idx in ranked_indices]
            rank = ranked_iks.index(true_ik) + 1 if true_ik in ranked_iks else 0
            rr = 1.0 / rank if 1 <= rank <= 25 else 0.0

            recips[cfg].append(rr)
            recips_grp[cfg][grp].append(rr)
            hits_1[cfg].append(1.0 if rank == 1 else 0.0)
            hits_5[cfg].append(1.0 if 1 <= rank <= 5 else 0.0)
            hits_25[cfg].append(1.0 if 1 <= rank <= 25 else 0.0)

        if (qi + 1) % 50 == 0 or (qi + 1) == n_queries:
            elapsed = time.time() - t_eval
            cur_a = np.mean(recips["A_MassPrior"])
            cur_b = np.mean(recips["B_MassDirect"])
            cur_c = np.mean(recips["C_MassDirectAnalog"])
            print(
                f"  [{qi+1:3d}/{n_queries}] Elapsed: {elapsed:.1f}s | "
                f"A(Mass): {cur_a:.4f} | B(+Direct): {cur_b:.4f} | C(+Analog): {cur_c:.4f}",
                flush=True,
            )

    eval_time = time.time() - t_eval
    print(f"\nCompleted evaluation in {eval_time:.1f}s ({eval_time/n_queries*1000:.1f} ms/query).")

    # -----------------------------------------------------------------
    # Final Structured Report
    # -----------------------------------------------------------------
    rec_all = _mean(hits_recall)
    rec_c1 = _mean(hits_recall_grp["C1"])
    rec_c2 = _mean(hits_recall_grp["C2"])
    rec_c3 = _mean(hits_recall_grp["C3"])

    print("\n" + "=" * 95)
    print("  PHASE 4 STEP 4: DIRECT VS ANALOG PROPAGATION EXPERIMENT REPORT")
    print("=" * 95)
    print(f"{'Configuration':<26} {'Overall MRR':<13} {'C1 MRR':<11} {'C2 MRR':<11} {'C3 MRR':<11} {'Recall':<10}")
    print("-" * 95)

    report_data = {}
    for cfg, label in [
        ("A_MassPrior", "Mass + prior"),
        ("B_MassDirect", "+ Direct"),
        ("C_MassDirectAnalog", "+ Direct + Analog"),
    ]:
        ov_mrr = _mean(recips[cfg])
        c1_mrr = _mean(recips_grp[cfg]["C1"])
        c2_mrr = _mean(recips_grp[cfg]["C2"])
        c3_mrr = _mean(recips_grp[cfg]["C3"])
        h1 = _mean(hits_1[cfg]) * 100
        h5 = _mean(hits_5[cfg]) * 100
        h25 = _mean(hits_25[cfg]) * 100

        report_data[label] = {
            "overall_mrr": ov_mrr,
            "c1_mrr": c1_mrr,
            "c2_mrr": c2_mrr,
            "c3_mrr": c3_mrr,
            "hit1": h1,
            "hit5": h5,
            "hit25": h25,
            "candidate_recall": rec_all * 100,
        }
        print(f"{label:<26} {ov_mrr:<13.4f} {c1_mrr:<11.4f} {c2_mrr:<11.4f} {c3_mrr:<11.4f} {rec_all*100:<10.2f}%")

    print("=" * 95)
    print(f"\nRecall Breakdown: Overall = {rec_all*100:.2f}%, C1 = {rec_c1*100:.2f}%, C2 = {rec_c2*100:.2f}%, C3 = {rec_c3*100:.2f}%")
    print("\nHit Rates:")
    for label, d in report_data.items():
        print(f"  {label:<22} -> Hit@1: {d['hit1']:5.2f}%, Hit@5: {d['hit5']:5.2f}%, Hit@25: {d['hit25']:5.2f}%")
    print("=" * 95)

    out_json = out_dir / "experiment_direct_vs_analog_results.json"
    report_data["molecule_key"] = "inchikey"
    report_data["retrieval_parameters"] = RETRIEVAL_PARAMETERS
    provenance = _provenance(out_dir)
    report_data["provenance"] = provenance
    report_data["split_sha256"] = provenance["split"].get("sha256", "")
    report_data["reference_sha256"] = provenance["reference"].get("sha256", "")
    report_data["candidate_sha256"] = provenance["candidate"].get("sha256", "")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)
    args = parser.parse_args()
    main(args.output_dir)
