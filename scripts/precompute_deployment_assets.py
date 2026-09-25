"""Precompute Deployment Assets for Stage 5 v2 + External Evidence Pipeline.

Generates standalone offline artifacts for Kaggle CPU environment:
1. kaggle_dataset/reranker_v2.pt               (CrossModalRerankerV2 state_dict)
2. kaggle_dataset/spec_encoder.pt               (SpectrumEncoder state_dict)
3. kaggle_dataset/candidate_embeddings.npy      (float16, shape [276940, 256] from fine-tuned MoleculeGNN)
4. kaggle_dataset/candidate_fps.npy             (uint8, shape [276940, 128] packed 1024-bit Morgan bitvectors)
5. kaggle_dataset/unified_reference_library.parquet (1.38M reference spectra sorted by neutral_mass)
"""
from __future__ import annotations

import concurrent.futures
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
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch, Data

from src.core.config import ARTIFACTS_DIR
from src.data.mol_graph import smiles_to_graph, ATOM_FDIM, BOND_FDIM
from src.models.molecule_encoder import MoleculeGNN
from src.models.reranker_v2 import CrossModalRerankerV2
from src.models.spectrum_encoder import SpectrumEncoder


def make_dummy_graph() -> Data:
    x = torch.zeros((1, ATOM_FDIM), dtype=torch.float32)
    x[0, 2] = 1.0  # carbon
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _compute_morgan_chunk(smiles_chunk: list[str]) -> np.ndarray:
    """Compute packed Morgan bit vectors (128 bytes = 1024 bits) for a chunk of SMILES."""
    chunk_size = len(smiles_chunk)
    packed_fps = np.zeros((chunk_size, 128), dtype=np.uint8)

    for i, smi in enumerate(smiles_chunk):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            # Convert BitVect to packed uint8 array (1024 bits = 128 bytes)
            # fp.ToBitString() returns '0101...' of length 1024
            bit_arr = np.array([int(b) for b in fp.ToBitString()], dtype=np.uint8)
            packed_fps[i] = np.packbits(bit_arr)
        # else remains all zeros

    return packed_fps


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_cpus = os.cpu_count() or 4
    n_workers = min(12, num_cpus)

    out_dir = ROOT / "kaggle_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("  PRECOMPUTING KAGGLE DEPLOYMENT ASSETS (STAGE 5 v2 + UNIFIED EVIDENCE)")
    print(f"  Device: {device} | Workers: {n_workers} | Output: {out_dir}")
    print("=" * 85, flush=True)

    # 1. Load Candidate Library
    cand_path = ROOT / "candidate_library.parquet"
    if not cand_path.exists():
        cand_path = out_dir / "candidate_library.parquet"
    print(f"\n[1/5] Loading Candidate Library from {cand_path}...", flush=True)
    df_cand = pd.read_parquet(cand_path)
    n_cands = len(df_cand)
    print(f"Loaded {n_cands:,} candidate structures. Neutral mass sorted: {df_cand['exact_mass'].is_monotonic_increasing}")
    assert df_cand["exact_mass"].is_monotonic_increasing, "Candidate library must be sorted by exact_mass!"

    # 2. Checkpoints: Stage 5 v2 & Stage 2 Spec Encoder
    print("\n[2/5] Exporting clean model checkpoints...", flush=True)
    s5_ckpt_path = ARTIFACTS_DIR / "stage05" / "exp5_v2" / "checkpoints" / "best.pt"
    s2_ckpt_path = ARTIFACTS_DIR / "stage02" / "exp2a" / "checkpoints" / "best.pt"

    assert s5_ckpt_path.exists(), f"Missing Stage 5 v2 checkpoint: {s5_ckpt_path}"
    assert s2_ckpt_path.exists(), f"Missing Stage 2 checkpoint: {s2_ckpt_path}"

    s5_data = torch.load(s5_ckpt_path, map_location=device, weights_only=False)
    s2_data = torch.load(s2_ckpt_path, map_location=device, weights_only=False)

    # Save reranker_v2 state_dict
    reranker_v2_path = out_dir / "reranker_v2.pt"
    torch.save(s5_data["reranker_state_dict"], reranker_v2_path)
    print(f"Saved CrossModalRerankerV2 weights to: {reranker_v2_path} ({reranker_v2_path.stat().st_size / 1e6:.2f} MB)")

    # Save spec_encoder state_dict
    spec_encoder_path = out_dir / "spec_encoder.pt"
    spec_state = s2_data.get("model_state_dict", s2_data)
    torch.save(spec_state, spec_encoder_path)
    print(f"Saved SpectrumEncoder weights to: {spec_encoder_path} ({spec_encoder_path.stat().st_size / 1e6:.2f} MB)")

    # 3. Precompute Candidate Morgan Fingerprints (1024-bit packed uint8)
    fps_out_path = out_dir / "candidate_fps.npy"
    print(f"\n[3/5] Precomputing Morgan fingerprints (1024 bits -> 128 bytes) for {n_cands:,} candidates...", flush=True)
    t_fp_start = time.time()

    smiles_list = df_cand["normalized_smiles"].tolist()
    chunk_size = max(500, n_cands // (n_workers * 4))
    chunks = [smiles_list[i:i + chunk_size] for i in range(0, n_cands, chunk_size)]

    packed_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for res in executor.map(_compute_morgan_chunk, chunks):
            packed_results.append(res)

    all_fps = np.vstack(packed_results)
    assert all_fps.shape == (n_cands, 128), f"Unexpected shape for all_fps: {all_fps.shape}"
    assert all_fps.dtype == np.uint8, f"Unexpected dtype: {all_fps.dtype}"
    np.save(fps_out_path, all_fps)
    print(f"Computed and saved Morgan fingerprints in {time.time() - t_fp_start:.1f}s -> {fps_out_path} ({fps_out_path.stat().st_size / 1e6:.2f} MB)")

    # 4. Precompute Candidate GNN Embeddings using Stage 5 v2 fine-tuned MoleculeGNN
    embs_out_path = out_dir / "candidate_embeddings.npy"
    print(f"\n[4/5] Precomputing fine-tuned MoleculeGNN embeddings (256-D float16) for {n_cands:,} candidates...", flush=True)
    t_gnn_start = time.time()

    mol_model = MoleculeGNN(embed_dim=256).to(device)
    mol_model.load_state_dict(s5_data["mol_encoder_state_dict"])
    mol_model.eval()

    embeddings = np.zeros((n_cands, 256), dtype=np.float16)
    dummy_graph = make_dummy_graph()
    batch_size = 512
    n_failed = 0

    for start_idx in range(0, n_cands, batch_size):
        end_idx = min(start_idx + batch_size, n_cands)
        batch_smis = smiles_list[start_idx:end_idx]

        graphs = []
        for smi in batch_smis:
            g = smiles_to_graph(smi, cache=False)
            if g is None or g.x is None or g.x.size(0) == 0:
                g = dummy_graph
                n_failed += 1
            graphs.append(g)

        bg = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            z = mol_model(bg)
        embeddings[start_idx:end_idx] = z.cpu().numpy().astype(np.float16)

        if (end_idx % 50000 < batch_size) or (end_idx == n_cands):
            elapsed = time.time() - t_gnn_start
            rate = end_idx / max(elapsed, 0.001)
            print(f"  [{end_idx:6d}/{n_cands:6d}] ({end_idx / n_cands * 100:.1f}%) | Rate: {rate:.1f} mols/s", flush=True)

    # Validate embeddings
    assert embeddings.shape == (n_cands, 256)
    assert not np.isnan(embeddings).any(), "NaN found in candidate embeddings!"
    assert not np.isinf(embeddings).any(), "Inf found in candidate embeddings!"
    np.save(embs_out_path, embeddings)
    print(f"Computed and saved GNN embeddings in {time.time() - t_gnn_start:.1f}s (failed graphs: {n_failed}) -> {embs_out_path} ({embs_out_path.stat().st_size / 1e6:.2f} MB)")

    # 5. Build Unified Reference Library Parquet (MoNA + GNPS + Baseline 839k)
    unified_ref_path = out_dir / "unified_reference_library.parquet"
    print(f"\n[5/5] Building Unified Reference Library ({unified_ref_path.name})...", flush=True)
    t_ref_start = time.time()

    ref_path = out_dir / "reference_library_multice.parquet"
    if not ref_path.exists():
        ref_path = ROOT / "kaggle_dataset" / "reference_library_multice.parquet"

    ext_path = ARTIFACTS_DIR / "external" / "external_spectra.parquet"

    print(f"Reading baseline reference library from: {ref_path}")
    df_ref = pd.read_parquet(ref_path, columns=["normalized_smiles", "neutral_mass", "precursor_mz", "collision_energy", "ms2_mzs", "ms2_intensities"])

    print(f"Reading external spectra library from: {ext_path}")
    df_ext = pd.read_parquet(ext_path, columns=[
        "canonical_smiles", "neutral_mass", "precursor_mz", "collision_energy",
        "peaks_mz", "peaks_intensity", "n_supporting_spectra", "source_count"
    ])

    # Unify columns
    df_ref = df_ref.rename(columns={"normalized_smiles": "canonical_smiles", "ms2_mzs": "peaks_mz", "ms2_intensities": "peaks_intensity"})
    df_ref["n_supporting_spectra"] = np.ones(len(df_ref), dtype=np.int16)
    df_ref["source_count"] = np.ones(len(df_ref), dtype=np.int8)

    # Cast datatypes for efficiency
    df_unified = pd.concat([df_ref, df_ext], ignore_index=True)
    n_total_ref = len(df_unified)
    print(f"Combined total spectra: {n_total_ref:,}")

    # Ensure strictly sorted by neutral_mass for binary search
    print("Sorting unified reference library by neutral_mass...")
    df_unified = df_unified.sort_values("neutral_mass").reset_index(drop=True)

    print(f"Writing unified parquet to: {unified_ref_path}...")
    df_unified.to_parquet(unified_ref_path, index=False, engine="pyarrow", compression="snappy")
    ref_size_mb = unified_ref_path.stat().st_size / (1024 * 1024)
    print(f"Unified reference library written successfully in {time.time() - t_ref_start:.1f}s ({ref_size_mb:.2f} MB).")

    print("\n" + "=" * 85)
    print("  ALL DEPLOYMENT ASSETS SUCCESSFULLY GENERATED & PACKAGED")
    print(f"  Total elapsed time: {time.time() - t0:.1f}s")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
