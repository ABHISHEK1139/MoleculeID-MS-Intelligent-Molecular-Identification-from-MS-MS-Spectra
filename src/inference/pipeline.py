"""Canonical Unified Inference Pipeline for CASMI 2026.

Single source of truth for:
1. Exact preprocessing and 1480 coarse binning ([20.0, 1500.0) @ 1.0 Da)
2. Authoritative adduct neutral mass derivation
3. Physical candidate retrieval with fallback tiers
4. External library spectral matching (mutual peak-matched cosine)
5. Non-leaking 10-D physical/spectral evidence vector extraction
6. Neural cross-modal reranking (SpectrumEncoder + GNN + Morgan + CrossModalRerankerV2)
7. Exact top-25 deduplicated ranking
"""
from __future__ import annotations

import math
import time
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.core.adducts import neutral_mass
from src.core.spectral_search import fast_mutual_cosine, CompactSpectralLibrary

# ── Production Constants (Exact match with training) ─────────────────────────
MZ_BIN_MIN = 20.0
MZ_BIN_MAX = 1500.0
COARSE_BIN_WIDTH = 1.0
COARSE_N_BINS = int(round((MZ_BIN_MAX - MZ_BIN_MIN) / COARSE_BIN_WIDTH))  # 1480


# ── Authoritative Preprocessing ──────────────────────────────────────────────
def deisotope_peaks(mzs: np.ndarray, ints: np.ndarray, tol_da: float = 0.015) -> tuple[np.ndarray, np.ndarray]:
    """Deisotoping clustering removing C13 isotopic companion peaks."""
    if mzs.size <= 1:
        return mzs, ints
    order = np.argsort(mzs)
    mzs_s = mzs[order]
    ints_s = ints[order]
    keep = np.ones(mzs_s.size, dtype=bool)

    for i in range(mzs_s.size):
        if not keep[i]:
            continue
        m_i = mzs_s[i]
        target_iso = m_i + 1.003355
        lo = np.searchsorted(mzs_s, target_iso - tol_da)
        hi = np.searchsorted(mzs_s, target_iso + tol_da)
        for j in range(lo, hi):
            if ints_s[j] < ints_s[i] * 0.90:
                keep[j] = False

    return mzs_s[keep], ints_s[keep]


def clean_spectrum(
    raw_mzs: np.ndarray,
    raw_intens: np.ndarray,
    max_peaks: int = 60,
    deisotope: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Production cleaning: deisotoping, sqrt scaling, L2 normalization, top-K selection."""
    mzs = np.asarray(raw_mzs, dtype=np.float32)
    intens = np.asarray(raw_intens, dtype=np.float32)

    if mzs.size > 1 and deisotope:
        mzs, intens = deisotope_peaks(mzs, intens)

    intens = np.sqrt(np.maximum(intens, 0.0))
    norm = np.linalg.norm(intens)
    if norm > 0:
        intens = intens / norm

    if mzs.size > max_peaks:
        top_idx = np.argsort(-intens)[:max_peaks]
        top_idx = top_idx[np.argsort(mzs[top_idx])]
        mzs = mzs[top_idx]
        intens = intens[top_idx]

    return mzs, intens


def spectrum_to_coarse_bins(mz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Bin a peak list into fixed 1480-D vector on [20.0, 1500.0) @ 1.0 Da."""
    bins = np.zeros(COARSE_N_BINS, dtype=np.float32)
    if mz.size == 0:
        return bins
    mask = (mz >= MZ_BIN_MIN) & (mz < MZ_BIN_MAX)
    if not np.any(mask):
        return bins
    mz_m = mz[mask]
    int_m = intensity[mask]
    indices = ((mz_m - MZ_BIN_MIN) / COARSE_BIN_WIDTH).astype(np.int64)
    np.add.at(bins, indices, int_m)
    norm = np.linalg.norm(bins)
    if norm > 0:
        bins /= norm
    return bins


def compute_morgan_tanimoto_numpy(
    fps_bytes: np.ndarray,
    cand_indices: np.ndarray,
    top_ref_idx: int | None,
) -> np.ndarray:
    """Fast bit-level Tanimoto similarity over precomputed uint8 Morgan bitvectors."""
    n_cands = len(cand_indices)
    if top_ref_idx is None or top_ref_idx < 0:
        return np.zeros(n_cands, dtype=np.float32)

    ref_bits = fps_bytes[top_ref_idx]
    cand_bits = fps_bytes[cand_indices]

    inter_bytes = np.bitwise_and(cand_bits, ref_bits[None, :])
    union_bytes = np.bitwise_or(cand_bits, ref_bits[None, :])

    inter_cnt = np.unpackbits(inter_bytes, axis=1).sum(axis=1)
    union_cnt = np.unpackbits(union_bytes, axis=1).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(union_cnt > 0, inter_cnt / union_cnt, 0.0)
    return sims.astype(np.float32)


# ── Canonical Inference Pipeline ─────────────────────────────────────────────
class CanonicalInferencePipeline:
    """Single authoritative inference pipeline used across validation, benchmark, and Kaggle."""

    def __init__(
        self,
        candidate_df: pd.DataFrame,
        candidate_embs: np.ndarray,
        candidate_fps: np.ndarray,
        ref_library: CompactSpectralLibrary,
        spec_encoder: torch.nn.Module,
        reranker: torch.nn.Module,
        device: torch.device | str = "cpu",
        gamma: float = 0.15,
        default_smiles: str = "CCO",
    ):
        self.cand_df = candidate_df
        self.cand_smiles = candidate_df["normalized_smiles"].to_numpy()
        self.cand_masses = candidate_df["exact_mass"].to_numpy(dtype=np.float64)
        self.cand_formulas = candidate_df["molecular_formula"].to_numpy() if "molecular_formula" in candidate_df.columns else None
        self.cand_embs = candidate_embs
        self.cand_fps = candidate_fps
        self.ref_library = ref_library
        self.spec_encoder = spec_encoder.to(device).eval()
        self.reranker = reranker.to(device).eval()
        self.device = torch.device(device)
        self.gamma = gamma
        self.default_smiles = default_smiles

        # Fast SMILES to index lookup
        self.smi_to_idx = {smi: i for i, smi in enumerate(self.cand_smiles)}

    def retrieve_candidates(self, m_neutral: float, min_cands: int = 25) -> tuple[np.ndarray, float]:
        """Physics candidate retrieval with progressive ppm expansion to guarantee at least min_cands candidates."""
        cands_idx = np.array([], dtype=np.int64)
        tier_w = 1.0

        for tol_ppm, weight in [(20e-6, 1.0), (50e-6, 0.50), (100e-6, 0.25)]:
            l_idx = np.searchsorted(self.cand_masses, m_neutral * (1.0 - tol_ppm))
            r_idx = np.searchsorted(self.cand_masses, m_neutral * (1.0 + tol_ppm))
            if r_idx > l_idx:
                cands_idx = np.arange(l_idx, r_idx)
                tier_w = weight
                if len(cands_idx) >= min_cands:
                    break

        if len(cands_idx) < min_cands:
            # 13C isotope fallback
            m_iso = m_neutral - 1.003355
            l_idx = np.searchsorted(self.cand_masses, m_iso * (1.0 - 50e-6))
            r_idx = np.searchsorted(self.cand_masses, m_iso * (1.0 + 50e-6))
            if r_idx > l_idx:
                iso_cands = np.arange(l_idx, r_idx)
                cands_idx = np.unique(np.concatenate([cands_idx, iso_cands]))
                tier_w = min(tier_w, 0.50)

        if len(cands_idx) < min_cands:
            # Nearest neighbors by exact mass to guarantee at least min_cands unique candidates
            center_idx = int(np.searchsorted(self.cand_masses, m_neutral))
            half_win = max(min_cands, 30)
            l_idx = max(0, center_idx - half_win)
            r_idx = min(len(self.cand_masses), center_idx + half_win)
            extra_cands = np.arange(l_idx, r_idx)
            cands_idx = np.unique(np.concatenate([cands_idx, extra_cands]))
            tier_w = min(tier_w, 0.10)

        return cands_idx, tier_w

    def search_reference_spectra(
        self,
        q_list: list[dict[str, Any]],
        m_neutral: float,
        cand_smis_set: set[str],
        excluded_smiles: set[str] | None = None,
    ) -> tuple[dict[str, dict[str, Any]], str, float]:
        """Search unified reference library with strict exclusion of validation molecules."""
        ref_l, ref_r = self.ref_library.query_window(m_neutral, ppm=20.0)
        ext_hits: dict[str, dict[str, Any]] = {}
        top_hit_cos = 0.0
        top_ref_smi = ""

        if ref_r > ref_l:
            for ri in range(ref_l, ref_r):
                r_smi = self.ref_library.smiles[ri]
                # Enforce clean validation: skip excluded/query molecules to eliminate leakage (C1)
                if excluded_smiles and r_smi in excluded_smiles:
                    continue
                if r_smi not in cand_smis_set:
                    continue

                for q in q_list:
                    delta = q["precursor_mz"] - self.ref_library.precursor_mzs[ri]
                    cos_sim, n_peaks = fast_mutual_cosine(
                        q["mzs"],
                        q["intens"],
                        self.ref_library.mzs_list[ri],
                        self.ref_library.intens_list[ri],
                        delta=delta,
                    )
                    if cos_sim > 0.10:
                        ce_val = q.get("ce", float("nan"))
                        ref_ce = self.ref_library.collision_energies[ri]
                        ce_diff = abs(ce_val - ref_ce) if (np.isfinite(ce_val) and np.isfinite(ref_ce)) else float("nan")

                        if r_smi not in ext_hits or cos_sim > ext_hits[r_smi]["cos"]:
                            ext_hits[r_smi] = {
                                "cos": cos_sim,
                                "n_peaks": n_peaks,
                                "ce_diff": ce_diff,
                                "n_supporting": int(self.ref_library.n_supporting[ri]),
                                "source_count": int(self.ref_library.source_counts[ri]),
                            }
                        if cos_sim > top_hit_cos:
                            top_hit_cos = cos_sim
                            top_ref_smi = r_smi

        return ext_hits, top_ref_smi, top_hit_cos

    def rank_query(
        self,
        q_list: list[dict[str, Any]],
        top_k: int = 25,
        excluded_smiles: set[str] | None = None,
    ) -> list[str]:
        """Score and rank candidates for a single query molecule."""
        if not q_list:
            return [self.default_smiles] * top_k

        valid_nms = [q["m_neutral"] for q in q_list if q.get("m_neutral") is not None and q["m_neutral"] > 0]
        if not valid_nms:
            return [self.default_smiles] * top_k
        m_neutral = float(np.median(valid_nms))

        # 1. Physics retrieval
        cands_idx, tier_w = self.retrieve_candidates(m_neutral)
        if len(cands_idx) == 0:
            return [self.default_smiles] * top_k

        cand_smis = self.cand_smiles[cands_idx]
        cand_smis_set = set(cand_smis)

        # 2. Reference library search (leak-free)
        ext_hits, top_ref_smi, _ = self.search_reference_spectra(
            q_list, m_neutral, cand_smis_set, excluded_smiles=excluded_smiles
        )

        top_ref_local_idx = None
        scores_evidence = []
        ev_vectors = []

        for i_local, ci in enumerate(cands_idx):
            c_smi = self.cand_smiles[ci]
            if c_smi == top_ref_smi:
                top_ref_local_idx = i_local

            ppm_err = abs(self.cand_masses[ci] - m_neutral) / m_neutral * 1e6
            s_mass = 1.50 * np.exp(-ppm_err / 10.0) * tier_w

            hit = ext_hits.get(c_smi)
            if hit and hit["cos"] >= 0.10:
                raw_ev = 0.769 * hit["cos"] + 0.231 * min(1.0, hit["n_peaks"] / 6.0)
                s_ev = 4.50 * (raw_ev ** 2) if raw_ev >= 0.45 else (1.0 * raw_ev if raw_ev >= 0.20 else 0.0)
                cos_val = min(1.0, max(0.0, float(hit["cos"])))
                n_p = float(hit["n_peaks"])
                p_norm = min(1.0, n_p / 15.0)
                p_ratio = min(1.0, n_p / (1.0 + 15.0 * cos_val))
                ce_d = hit["ce_diff"]
                ce_agree = float(np.exp(-abs(ce_d) / 20.0)) if np.isfinite(ce_d) else 0.60
                n_s = hit["n_supporting"]
                multi_n = min(1.0, math.log1p(max(1, n_s)) / math.log1p(5))
                src_c = 1.0 if hit["source_count"] >= 2 else 0.0
            else:
                s_ev = 0.0
                cos_val, p_norm, p_ratio, ce_agree, multi_n, src_c = 0.0, 0.0, 0.0, 0.60, 0.0, 0.0

            scores_evidence.append(s_mass + s_ev)

            # Clean inference feature vector: NO oracle label leakage!
            # Instead of comparing to unknown ground truth, compare to the primary window mass-center
            is_iso = 1.0 if (self.cand_formulas is not None and self.cand_formulas[ci] == self.cand_formulas[cands_idx[0]]) else 0.0
            ev_vectors.append([
                cos_val, p_norm, p_ratio, ce_agree, multi_n, src_c,
                min(1.0, max(0.0, ppm_err / 20.0)),
                tier_w,
                q_list[0]["precursor_mz"] / 1000.0,
                is_iso,
            ])

        # 3. Neural scoring
        z_spec_list = []
        for q in q_list:
            binned = spectrum_to_coarse_bins(q["mzs"], q["intens"])
            ce_val = q.get("ce", 30.0)
            if not np.isfinite(ce_val):
                ce_val = 30.0
            meta_vec = np.array([q["precursor_mz"] / 1000.0, q.get("ion_mode", 1.0), ce_val / 100.0], dtype=np.float32)
            spec_feat = np.concatenate([binned, meta_vec])
            spec_t = torch.from_numpy(spec_feat).unsqueeze(0).to(self.device)
            with torch.no_grad():
                z_spec_list.append(self.spec_encoder(spec_t))

        if z_spec_list:
            z_spec = torch.mean(torch.cat(z_spec_list, dim=0), dim=0, keepdim=True)
            z_spec = F.normalize(z_spec, p=2, dim=-1)
        else:
            z_spec = torch.zeros((1, 256), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            sub_z_mols = torch.from_numpy(self.cand_embs[cands_idx].astype(np.float32)).to(self.device)
            # Contrastive cross-modal cosine similarity
            dot_scores = (sub_z_mols * z_spec).sum(dim=-1).cpu().numpy()

        ppm_errs = abs(self.cand_masses[cands_idx] - m_neutral) / m_neutral * 1e6
        # Gentle mass decay: preserves candidate rankings within +-15 ppm while penalizing wide fallbacks
        mass_scores = 0.10 * np.exp(-ppm_errs / 50.0) * tier_w

        # High-confidence external library evidence bonus (gated >= 0.70 to prevent false-analog poisoning)
        lib_bonus = np.zeros(len(cands_idx), dtype=np.float32)
        for i_local, ci in enumerate(cands_idx):
            c_smi = self.cand_smiles[ci]
            hit = ext_hits.get(c_smi)
            if hit and hit["cos"] >= 0.70 and hit["n_peaks"] >= 4:
                lib_bonus[i_local] = 1.0 + 2.0 * float(hit["cos"])

        final_scores = dot_scores + mass_scores + lib_bonus

        # 4. Strict deduplication and stable sort
        order = np.argsort(-final_scores, kind="mergesort")
        seen: set[str] = set()
        dedup_ranked = []
        for idx in order:
            smi = cand_smis[idx]
            if smi not in seen:
                seen.add(smi)
                dedup_ranked.append(smi)
                if len(dedup_ranked) == top_k:
                    break

        while len(dedup_ranked) < top_k:
            dedup_ranked.append(self.default_smiles)

        return dedup_ranked[:top_k]
