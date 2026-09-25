"""Evaluate Clean Stage 5 v2 Retrained Model on Clean v3 Benchmark (500 queries) against Expanded Candidate Universe (776,699 structures).

Evaluates:
- Pure neural cross-modal alignment (z_spec, z_mol)
- Morgan structural similarity to top reference
- 10-D unprivileged evidence vector
- 0 Oracle Leaks (precursor_mz + adduct only, zero formula flag, zero target mass window)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.preprocessing_v3 import (
    MAX_PEAKS,
    encode_spectrum_feature,
    fast_mutual_cosine,
    neutral_mass,
    preprocess_spectrum,
    retrieve_candidates_progressive,
)
from src.data.clean_hard_negative_dataset import extract_unprivileged_evidence_vector
from src.data.mol_graph import ATOM_FDIM, BOND_FDIM, smiles_to_graph
from src.models.molecule_encoder import MoleculeGNN
from src.models.reranker_v2 import CrossModalRerankerV2
from src.models.spectrum_encoder import SpectrumEncoder


def make_fallback_graph() -> Data:
    x = torch.zeros((1, ATOM_FDIM), dtype=torch.float32)
    x[0, 2] = 1.0
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class CleanStage5Evaluator:
    def __init__(
        self,
        ref_library_path: Path,
        cand_df: pd.DataFrame,
        cand_fps: np.ndarray,
        spec_encoder: nn.Module,
        mol_encoder: nn.Module,
        reranker: nn.Module,
        device: torch.device | str = "cpu",
    ):
        self.device = torch.device(device)
        self.cand_df = cand_df
        self.cand_smiles = (
            cand_df["canonical_smiles"].to_numpy()
            if "canonical_smiles" in cand_df.columns
            else cand_df["normalized_smiles"].to_numpy()
        )
        self.cand_masses = cand_df["exact_mass"].to_numpy(dtype=np.float64)
        self.cand_fps = cand_fps
        self.smi_to_idx = {s: i for i, s in enumerate(self.cand_smiles)}

        self.spec_encoder = spec_encoder.to(self.device).eval()
        self.mol_encoder = mol_encoder.to(self.device).eval()
        self.reranker = reranker.to(self.device).eval()

        # Popcount table for fast bitwise Tanimoto
        self._popcount_lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

        print(f"Loading clean reference library from {ref_library_path}...", flush=True)
        df_lib = pd.read_parquet(ref_library_path)
        order = np.argsort(df_lib["neutral_mass"].to_numpy(dtype=np.float64))
        self.lib_smiles = df_lib["normalized_smiles"].to_numpy()[order]
        self.lib_neutral_masses = df_lib["neutral_mass"].to_numpy(dtype=np.float64)[order]
        self.lib_prec_mzs = df_lib["precursor_mz"].to_numpy(dtype=np.float64)[order]
        self.lib_ces = df_lib["collision_energy"].to_numpy(dtype=np.float32)[order]
        self.lib_mzs_list = [np.asarray(x, dtype=np.float32) for x in df_lib["peaks_mz"].iloc[order]]
        self.lib_intens_list = [np.asarray(x, dtype=np.float32) for x in df_lib["peaks_intensity"].iloc[order]]
        print(f"Loaded and indexed {len(df_lib):,} clean reference spectra.", flush=True)

        self.graph_cache: dict[str, Data] = {}

    def get_graph(self, smi: str) -> Data:
        if smi not in self.graph_cache:
            try:
                g = smiles_to_graph(smi)
            except Exception:
                g = make_fallback_graph()
            self.graph_cache[smi] = g
        return self.graph_cache[smi]

    def fast_tanimoto_single(self, query_fp: np.ndarray, target_fps: np.ndarray) -> np.ndarray:
        inter = np.bitwise_and(query_fp, target_fps)
        inter_counts = self._popcount_lut[inter].sum(axis=-1)
        q_count = self._popcount_lut[query_fp].sum()
        t_counts = self._popcount_lut[target_fps].sum(axis=-1)
        union_counts = q_count + t_counts - inter_counts
        return np.where(union_counts > 0, inter_counts / union_counts, 0.0).astype(np.float32)

    def search_library(
        self,
        m_neutral: float,
        q_mzs: np.ndarray,
        q_intens: np.ndarray,
        q_prec_mz: float,
        q_ce: float,
        ppm: float = 20.0,
    ) -> tuple[dict[str, dict[str, Any]], str]:
        tol = m_neutral * (ppm / 1e6)
        l_idx = int(np.searchsorted(self.lib_neutral_masses, m_neutral - tol, side="left"))
        r_idx = int(np.searchsorted(self.lib_neutral_masses, m_neutral + tol, side="right"))

        hits: dict[str, dict[str, Any]] = {}
        top_ref_smi = ""
        top_ref_cos = 0.0

        if r_idx > l_idx:
            for ri in range(l_idx, r_idx):
                ref_smi = self.lib_smiles[ri]
                ref_prec = self.lib_prec_mzs[ri]
                delta = q_prec_mz - ref_prec

                cos_sim, n_peaks = fast_mutual_cosine(
                    q_mzs,
                    q_intens,
                    self.lib_mzs_list[ri],
                    self.lib_intens_list[ri],
                    delta=delta,
                )
                if cos_sim >= 0.10:
                    r_ce = self.lib_ces[ri]
                    ce_diff = abs(q_ce - r_ce) if (np.isfinite(q_ce) and np.isfinite(r_ce)) else float("nan")
                    if ref_smi not in hits or cos_sim > hits[ref_smi]["cos"]:
                        hits[ref_smi] = {
                            "cos": cos_sim,
                            "n_peaks": n_peaks,
                            "ce_diff": ce_diff,
                            "n_supporting": 1,
                            "source_count": 1,
                        }
                    if cos_sim > top_ref_cos:
                        top_ref_cos = cos_sim
                        top_ref_smi = ref_smi

        return hits, top_ref_smi

    def rank_query(
        self,
        q_mzs: np.ndarray,
        q_intens: np.ndarray,
        precursor_mz: float,
        adduct: str,
        ce_val: float,
        top_k: int = 25,
    ) -> list[str]:
        # 1. Observed neutral mass ONLY (zero oracle leakage)
        m_neutral = neutral_mass(precursor_mz, adduct)
        if m_neutral is None or m_neutral <= 0 or not np.isfinite(m_neutral):
            m_neutral = precursor_mz - 1.007825

        # 2. Candidate retrieval on expanded catalog
        cands_idx, tier_weights = retrieve_candidates_progressive(m_neutral, self.cand_masses, min_cands=top_k)
        cand_smis = self.cand_smiles[cands_idx]
        n_cands = len(cands_idx)

        # 3. Spectral Library Search
        ext_hits, top_ref_smi = self.search_library(m_neutral, q_mzs, q_intens, precursor_mz, ce_val)

        # 4. Neural query embedding (SpectrumEncoder)
        spec_feat = encode_spectrum_feature(q_mzs, q_intens, precursor_mz, adduct, ce_val)
        spec_t = torch.from_numpy(spec_feat).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z_spec = self.spec_encoder(spec_t)  # (1, 256)

        # 5. Candidate graph representations & MoleculeGNN forward pass
        graph_list = [self.get_graph(s) for s in cand_smis]
        batch_graphs = Batch.from_data_list(graph_list).to(self.device)
        with torch.no_grad():
            z_mols = self.mol_encoder(batch_graphs)  # (K, 256)

        # 6. Morgan similarities to top spectral reference
        morgan_sims = np.zeros(n_cands, dtype=np.float32)
        if top_ref_smi and top_ref_smi in self.smi_to_idx:
            ref_idx = self.smi_to_idx[top_ref_smi]
            ref_fp = self.cand_fps[ref_idx]
            cand_fps_sub = self.cand_fps[cands_idx]
            morgan_sims = self.fast_tanimoto_single(ref_fp, cand_fps_sub)

        # 7. Unprivileged 10-D evidence vectors
        ev_matrix = np.zeros((n_cands, 10), dtype=np.float32)
        for i_local, ci in enumerate(cands_idx):
            c_smi = self.cand_smiles[ci]
            hit = ext_hits.get(c_smi)
            c_mass = self.cand_masses[ci]
            ppm_err = abs(c_mass - m_neutral) / m_neutral * 1e6
            tw = tier_weights[i_local]
            ev_matrix[i_local] = extract_unprivileged_evidence_vector(
                hit=hit,
                ppm_error=ppm_err,
                tier_weight=tw,
                prec_mz=precursor_mz,
                formula_match_center=1.0,
            )

        morgan_t = torch.from_numpy(morgan_sims).to(self.device)
        ev_t = torch.from_numpy(ev_matrix).to(self.device)

        # 8. CrossModalRerankerV2 forward pass
        with torch.no_grad():
            reranker_scores = self.reranker(z_spec, z_mols, morgan_t, ev_t).cpu().numpy()

        # 9. Mass decay prior (gentle 50 ppm decay)
        ppm_errs = abs(self.cand_masses[cands_idx] - m_neutral) / m_neutral * 1e6
        mass_scores = 0.10 * np.exp(-ppm_errs / 50.0) * tier_weights

        final_scores = reranker_scores + mass_scores
        sort_order = np.argsort(-final_scores)
        top_indices = sort_order[:top_k]
        return [cand_smis[idx] for idx in top_indices]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Clean Stage 5 v2 on Benchmark")
    parser.add_argument("--checkpoint", type=str, default="artifacts/v3_clean/checkpoints/v3_clean_stage5_v2_best.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    split_file = ROOT / "artifacts" / "v3_clean" / "clean_split.json"
    ref_file = ROOT / "artifacts" / "v3_clean" / "clean_reference_library.parquet"
    cand_file = ROOT / "artifacts" / "v3_clean" / "candidate_union.parquet"
    fps_file = ROOT / "artifacts" / "v3_clean" / "candidate_fps.npy"
    ckpt_path = ROOT / args.checkpoint

    print("=" * 85)
    print("  EVALUATING CLEAN STAGE 5 v2 ON EXPANDED 777K CANDIDATE UNIVERSE")
    print(f"  Checkpoint: {ckpt_path.name} | Device: {args.device}")
    print("=" * 85, flush=True)

    with open(split_file, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    benchmark_queries = split_data["benchmark_queries"]

    print(f"Loading expanded candidate library from {cand_file}...", flush=True)
    cand_df = pd.read_parquet(cand_file)
    print(f"Loading candidate fingerprints from {fps_file}...", flush=True)
    cand_fps = np.load(fps_file)

    # Load Models
    print("Loading models and checkpoint weights...", flush=True)
    spec_encoder = SpectrumEncoder(embed_dim=256)
    spec_ckpt = torch.load(ROOT / "kaggle_dataset" / "spec_encoder.pt", map_location=args.device, weights_only=True)
    spec_encoder.load_state_dict(spec_ckpt.get("model_state_dict", spec_ckpt))

    mol_encoder = MoleculeGNN(embed_dim=256)
    reranker = CrossModalRerankerV2(embed_dim=256, evidence_dim=10, hidden_dim=256, dropout=0.10)

    if ckpt_path.exists():
        ckpt_data = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        print(f"Loaded checkpoint saved from epoch {ckpt_data.get('epoch', '?')}")
        if "mol_encoder_state_dict" in ckpt_data:
            mol_encoder.load_state_dict(ckpt_data["mol_encoder_state_dict"])
        if "reranker_state_dict" in ckpt_data:
            reranker.load_state_dict(ckpt_data["reranker_state_dict"])
    else:
        print(f"WARNING: Checkpoint {ckpt_path} not found! Using initialized weights.")

    evaluator = CleanStage5Evaluator(
        ref_library_path=ref_file,
        cand_df=cand_df,
        cand_fps=cand_fps,
        spec_encoder=spec_encoder,
        mol_encoder=mol_encoder,
        reranker=reranker,
        device=args.device,
    )

    raw_pq = ROOT / "artifacts" / "v3_clean" / "benchmark_queries_raw.parquet"
    print(f"Loading raw query spectra from {raw_pq}...", flush=True)
    df_raw = pq.read_table(raw_pq).to_pandas()

    all_recips = []
    mode_a_recips = []
    mode_b_recips = []
    h1_list = []
    h5_list = []
    h25_list = []

    t_start = time.time()
    for qi, bq in enumerate(benchmark_queries):
        row = df_raw.iloc[qi]
        raw_mzs = np.asarray(row["ms2_mzs"], dtype=np.float32)
        raw_intens = np.asarray(row["ms2_normalized_intensities"], dtype=np.float32)

        cm, ci = preprocess_spectrum(raw_mzs, raw_intens, max_peaks=MAX_PEAKS, deisotope=True)
        prec_mz = float(row["precursor_mz"])
        adduct = str(row["adduct"])

        ce_val = 30.0
        ce_raw = row["collision_energy_ev"]
        if ce_raw is not None:
            try:
                if hasattr(ce_raw, "__iter__") and len(ce_raw) > 0:
                    ce_val = float(ce_raw[0])
                else:
                    ce_val = float(ce_raw)
            except Exception:
                ce_val = 30.0

        ranked_smis = evaluator.rank_query(
            cm, ci, prec_mz, adduct, ce_val, top_k=25
        )

        true_smi = bq["true_smiles"]
        r = ranked_smis.index(true_smi) + 1 if true_smi in ranked_smis else 0
        rr = 1.0 / r if 1 <= r <= 25 else 0.0

        all_recips.append(rr)
        h1_list.append(1.0 if r == 1 else 0.0)
        h5_list.append(1.0 if 1 <= r <= 5 else 0.0)
        h25_list.append(1.0 if 1 <= r <= 25 else 0.0)

        if "zero_reference" in bq["mode"]:
            mode_a_recips.append(rr)
        else:
            mode_b_recips.append(rr)

        if (qi + 1) % 50 == 0 or (qi + 1) == len(benchmark_queries):
            cur_mrr = np.mean(all_recips)
            cur_mrr_a = np.mean(mode_a_recips) if mode_a_recips else 0.0
            cur_mrr_b = np.mean(mode_b_recips) if mode_b_recips else 0.0
            elapsed = time.time() - t_start
            print(
                f"  [{qi+1:3d}/500] Elapsed: {elapsed:.1f}s | "
                f"Overall MRR: {cur_mrr:.4f} | Mode A: {cur_mrr_a:.4f} | Mode B: {cur_mrr_b:.4f}",
                flush=True,
            )

    eval_time = time.time() - t_start
    overall_mrr = float(np.mean(all_recips))
    mode_a_mrr = float(np.mean(mode_a_recips))
    mode_b_mrr = float(np.mean(mode_b_recips))
    h1 = float(np.mean(h1_list))
    h5 = float(np.mean(h5_list))
    h25 = float(np.mean(h25_list))

    print("\n" + "=" * 85)
    print("  CLEAN STAGE 5 v2 BENCHMARK EVALUATION RESULTS (776,699 CANDIDATES)")
    print("=" * 85)
    print(f"  Overall MRR@25:   {overall_mrr:.4f}   (Baseline unretrained: 0.2544, Legacy 277k: 0.2805)")
    print(f"  Mode A (Zero-Ref):{mode_a_mrr:.4f}   (Baseline unretrained: 0.0980, Legacy 277k: 0.1296)")
    print(f"  Mode B (Leave-1): {mode_b_mrr:.4f}   (Baseline unretrained: 0.4107, Legacy 277k: 0.4314)")
    print(f"  Hit@1:            {h1*100:.2f}%")
    print(f"  Hit@5:            {h5*100:.2f}%")
    print(f"  Hit@25:           {h25*100:.2f}%")
    print(f"  Evaluation Time:  {eval_time:.1f}s ({eval_time/500*1000:.1f} ms/query)")
    print("=" * 85)

    results = {
        "checkpoint": str(ckpt_path),
        "overall_mrr": overall_mrr,
        "mode_a_mrr": mode_a_mrr,
        "mode_b_mrr": mode_b_mrr,
        "hit1": h1,
        "hit5": h5,
        "hit25": h25,
        "eval_time_s": eval_time,
    }
    out_json = ROOT / "artifacts" / "v3_clean" / "benchmark_stage5_v2_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
