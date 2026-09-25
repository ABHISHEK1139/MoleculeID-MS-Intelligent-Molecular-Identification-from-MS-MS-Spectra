"""CASMI 2026 Submission Generator (Canonical Inference Pipeline).

Single authoritative submission generator using src.inference.pipeline.
Ensures 100% parity between local clean evaluation and Kaggle production.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch

from src.core.adducts import neutral_mass
from src.core.spectral_search import CompactSpectralLibrary
from src.models.reranker_v2 import CrossModalRerankerV2
from src.models.spectrum_encoder import SpectrumEncoder
from src.inference.pipeline import CanonicalInferencePipeline, clean_spectrum


def main():
    parser = argparse.ArgumentParser(description="CASMI 2026 Canonical Submission Generator")
    parser.add_argument("--test-path", type=str, default="dataset/test.parquet")
    parser.add_argument("--sample-sub", type=str, default="dataset/sample_submission.csv")
    parser.add_argument("--output", type=str, default="submission_canonical.csv")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 85)
    print("  CASMI 2026 CANONICAL INFERENCE SUBMISSION GENERATOR")
    print(f"  Target Device: {args.device} | Output: {args.output}")
    print("=" * 85)

    # 1. Load Candidate Universe
    print("\n[1/4] Loading Candidate Universe (276,940 candidates)...", flush=True)
    cand_df = pd.read_parquet(ROOT / "kaggle_dataset" / "candidate_library.parquet")
    cand_embs = np.load(ROOT / "kaggle_dataset" / "candidate_embeddings.npy")
    cand_fps = np.load(ROOT / "kaggle_dataset" / "candidate_fps.npy")
    print(f"Loaded {len(cand_df):,} candidates.")

    # 2. Load Models
    print("\n[2/4] Loading Pretrained Neural Models...", flush=True)
    spec_encoder = SpectrumEncoder(embed_dim=256)
    spec_ckpt = torch.load(ROOT / "kaggle_dataset" / "spec_encoder.pt", map_location=args.device, weights_only=False)
    spec_encoder.load_state_dict(spec_ckpt.get("model_state_dict", spec_ckpt))
    spec_encoder.eval()

    reranker = CrossModalRerankerV2(embed_dim=256, evidence_dim=10, hidden_dim=256, dropout=0.10)
    reranker_ckpt = torch.load(ROOT / "kaggle_dataset" / "reranker_v2.pt", map_location=args.device, weights_only=False)
    reranker.load_state_dict(reranker_ckpt.get("model_state_dict", reranker_ckpt))
    reranker.eval()

    # 3. Load Unified Reference Library
    print("\n[3/4] Ingesting Unified Reference Library (1.38M Spectra)...", flush=True)
    ref_paths = [
        ROOT / "kaggle_dataset" / "unified_reference_library.parquet",
        ROOT / "artifacts" / "baseline" / "reference_library_multice.parquet",
    ]
    ref_file = next((p for p in ref_paths if p.exists()), None)
    if ref_file is None:
        raise FileNotFoundError("Could not find unified reference library parquet file.")

    df_lib = pd.read_parquet(ref_file)
    smi_col = "canonical_smiles" if "canonical_smiles" in df_lib.columns else "normalized_smiles"
    mzs_col = "peaks_mz" if "peaks_mz" in df_lib.columns else "ms2_mzs"
    ints_col = "peaks_intensity" if "peaks_intensity" in df_lib.columns else "ms2_intensities"

    ref_library = CompactSpectralLibrary(
        smiles=df_lib[smi_col].to_numpy(),
        neutral_masses=df_lib["neutral_mass"].to_numpy(dtype=np.float64),
        precursor_mzs=df_lib["precursor_mz"].to_numpy(dtype=np.float64),
        collision_energies=df_lib["collision_energy"].to_numpy(dtype=np.float32),
        mzs_list=[np.asarray(x, dtype=np.float32) for x in df_lib[mzs_col]],
        intens_list=[np.asarray(x, dtype=np.float32) for x in df_lib[ints_col]],
        n_supporting=df_lib["n_supporting_spectra"].to_numpy(dtype=np.int16) if "n_supporting_spectra" in df_lib.columns else None,
        source_counts=df_lib["source_count"].to_numpy(dtype=np.int8) if "source_count" in df_lib.columns else None,
    )
    print(f"Loaded {len(df_lib):,} reference spectra.")

    # Canonical Pipeline Instance
    pipeline = CanonicalInferencePipeline(
        candidate_df=cand_df,
        candidate_embs=cand_embs,
        candidate_fps=cand_fps,
        ref_library=ref_library,
        spec_encoder=spec_encoder,
        reranker=reranker,
        device=args.device,
    )

    # 4. Ingest and Process Test Queries
    print(f"\n[4/4] Ingesting Test Queries ({args.test_path})...", flush=True)
    test_df = pd.read_parquet(ROOT / args.test_path)
    sample_sub = pd.read_csv(ROOT / args.sample_sub)
    sample_mol_ids = list(sample_sub["molecule_id"])

    mol_groups = defaultdict(list)
    for _, row in test_df.iterrows():
        raw_mzs = np.asarray(row["ms2_mzs"], dtype=np.float32)
        raw_intens = np.asarray(row["ms2_normalized_intensities"], dtype=np.float32)
        clean_mzs, clean_intens = clean_spectrum(raw_mzs, raw_intens, max_peaks=60)

        prec_mz = float(row["precursor_mz"])
        adduct = str(row["adduct"])
        m_neut = neutral_mass(prec_mz, adduct)
        if m_neut is None or m_neut <= 0:
            m_neut = prec_mz - 1.007825

        ce_raw = row.get("collision_energy_ev", 30.0)
        if isinstance(ce_raw, (list, tuple, np.ndarray)) and len(ce_raw) > 0:
            ce_val = float(ce_raw[0])
        elif isinstance(ce_raw, (int, float)):
            ce_val = float(ce_raw)
        else:
            ce_val = 30.0

        q_entry = {
            "adduct": adduct,
            "precursor_mz": prec_mz,
            "m_neutral": m_neut,
            "mzs": clean_mzs,
            "intens": clean_intens,
            "ce": ce_val,
            "ion_mode": 1.0 if not adduct.endswith("-") else 0.0,
        }
        mol_groups[row["molecule_id"]].append(q_entry)

    submission_rows = []
    t_inf_start = time.time()
    for mi, mol_id in enumerate(sample_mol_ids):
        q_list = mol_groups.get(mol_id, [])
        ranked_smis = pipeline.rank_query(q_list, top_k=25)
        submission_rows.append({
            "molecule_id": mol_id,
            "smiles": ";".join(ranked_smis),
        })
        if (mi + 1) % 100 == 0 or (mi + 1) == len(sample_mol_ids):
            print(f"  Processed {mi+1}/{len(sample_mol_ids)} queries in {time.time()-t_inf_start:.1f}s...", flush=True)

    df_sub = pd.DataFrame(submission_rows)

    # 5. Strict Verification & Output
    print("\n" + "=" * 85)
    print("  VERIFYING SUBMISSION INTEGRITY")
    print("=" * 85)
    assert len(df_sub) == len(sample_sub), f"Row count mismatch: {len(df_sub)} vs {len(sample_sub)}"
    assert list(df_sub["molecule_id"]) == sample_mol_ids, "Molecule ID order mismatch!"
    assert df_sub.isnull().sum().sum() == 0, "Null values found in submission!"
    assert all(len(s.split(";")) == 25 for s in df_sub["smiles"]), "Rows must have exactly 25 SMILES!"
    assert all(len(set(s.split(";"))) == 25 for s in df_sub["smiles"]), "Duplicate SMILES found in rows!"

    out_path = Path(args.output)
    df_sub.to_csv(out_path, index=False)
    print(f"  Total Rows:           {len(df_sub)} / {len(sample_sub)}")
    print(f"  Order Preserved:      PASSED")
    print(f"  Duplicate SMILES:     0 (ALL ROWS 100% UNIQUE)")
    print(f"  Inference Latency:    {time.time() - t_inf_start:.2f}s (~{(time.time()-t_inf_start)/len(sample_sub)*1000:.1f} ms/mol)")
    print(f"  Saved clean submission to: {out_path.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    main()
