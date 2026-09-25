"""Precompute 256-D molecular GNN embeddings for all candidates in candidate_library.parquet.

Exports:
1. kaggle_dataset/candidate_embeddings.npy  (float16, shape [276940, 256])
2. kaggle_dataset/spec_encoder.pt          (clean PyTorch state_dict for SpectrumEncoder)
3. kaggle_dataset/reranker.pt              (clean PyTorch state_dict for CrossModalReranker)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from src.models.molecule_encoder import MoleculeGNN
from src.models.spectrum_encoder import SpectrumEncoder
from src.models.reranker import CrossModalReranker
from src.data.mol_graph import smiles_to_graph, ATOM_FDIM, BOND_FDIM


def make_dummy_graph() -> Data:
    """Fallback single-atom graph for unparseable SMILES."""
    x = torch.zeros((1, ATOM_FDIM), dtype=torch.float32)
    x[0, 2] = 1.0  # carbon
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def main() -> None:
    print("=" * 80)
    print("  PRECOMPUTING CANDIDATE MOLECULAR EMBEDDINGS (STAGE 3/5 GNN)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}", flush=True)

    # 1. Load candidate library
    cand_path = ROOT / "candidate_library.parquet"
    if not cand_path.exists():
        cand_path = ROOT / "kaggle_dataset" / "candidate_library.parquet"
    print(f"Reading candidate library from: {cand_path}")
    df_cand = pd.read_parquet(cand_path)
    n_cands = len(df_cand)
    print(f"Total candidate structures: {n_cands:,}")

    # 2. Load trained models
    s5_ckpt_path = ROOT / "artifacts/stage05/exp5a/checkpoints/best.pt"
    s2_ckpt_path = ROOT / "artifacts/stage02/exp2a/checkpoints/best.pt"

    print(f"Loading Stage 5 checkpoint from: {s5_ckpt_path}")
    s5_data = torch.load(s5_ckpt_path, map_location=device, weights_only=False)
    mol_model = MoleculeGNN(embed_dim=256).to(device)
    mol_model.load_state_dict(s5_data["mol_encoder_state_dict"])
    mol_model.eval()

    reranker_state = s5_data["reranker_state_dict"]

    print(f"Loading Stage 2 checkpoint from: {s2_ckpt_path}")
    s2_data = torch.load(s2_ckpt_path, map_location=device, weights_only=False)
    spec_state = s2_data.get("model_state_dict", s2_data)

    # 3. Export clean standalone checkpoints
    out_dir = ROOT / "kaggle_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_out_path = out_dir / "spec_encoder.pt"
    rerank_out_path = out_dir / "reranker.pt"
    print(f"Saving standalone SpectrumEncoder state_dict to: {spec_out_path}")
    torch.save(spec_state, spec_out_path)
    print(f"Saving standalone CrossModalReranker state_dict to: {rerank_out_path}")
    torch.save(reranker_state, rerank_out_path)

    # 4. Precompute candidate embeddings
    embeddings = np.zeros((n_cands, 256), dtype=np.float16)
    dummy_graph = make_dummy_graph()

    batch_size = 512
    t_start = time.time()
    n_failed = 0

    print(f"\nEncoding {n_cands:,} molecules in batches of {batch_size}...")

    smiles_list = df_cand["normalized_smiles"].tolist()

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

        if (end_idx % 25000 < batch_size) or (end_idx == n_cands):
            elapsed = time.time() - t_start
            rate = end_idx / max(elapsed, 0.001)
            remaining = (n_cands - end_idx) / max(rate, 0.001) / 60.0
            print(f"  [{end_idx:6d}/{n_cands:6d}] ({end_idx/n_cands*100:.1f}%) "
                  f"Elapsed: {elapsed:.1f}s | Rate: {rate:.1f} mols/s | ETA: {remaining:.1f} min",
                  flush=True)

    total_time = time.time() - t_start
    print(f"\nEncoding complete in {total_time:.1f} seconds ({n_cands/total_time:.1f} mols/s).")
    print(f"Fallback dummy graphs used for unparseable SMILES: {n_failed}")

    # 5. Validation Checks
    print("\nRunning embedding verification checks...")
    assert embeddings.shape == (n_cands, 256), f"Shape mismatch: {embeddings.shape}"
    assert not np.isnan(embeddings).any(), "Found NaN in candidate embeddings!"
    assert not np.isinf(embeddings).any(), "Found Inf in candidate embeddings!"

    # Sample L2 norms
    norms = np.linalg.norm(embeddings[:1000].astype(np.float32), axis=1)
    mean_norm = float(np.mean(norms))
    print(f"Mean L2 norm of sample embeddings: {mean_norm:.4f} (target: ~1.000)")
    assert 0.95 <= mean_norm <= 1.05, f"Unexpected mean norm: {mean_norm}"

    # 6. Save embeddings
    embs_path = out_dir / "candidate_embeddings.npy"
    print(f"Saving candidate embeddings to: {embs_path} ({embs_path.stat().st_size / 1e6 if embs_path.exists() else 0:.1f} MB)...")
    np.save(embs_path, embeddings)

    file_size_mb = embs_path.stat().st_size / (1024 * 1024)
    print(f"File saved successfully! Size: {file_size_mb:.2f} MB")
    print("=" * 80)
    print("  SUCCESSFULLY PRECOMPUTED & PACKAGED ALL CANDIDATE EMBEDDINGS")
    print("=" * 80)


if __name__ == "__main__":
    main()
