"""Stage 5 v2 Training: Hard-Negative Isomer & False-Analog Reranker (Memory-Optimized).

High-Performance & Low-Memory Training Pipeline:
1. Samples hard negatives across the full 276,940 candidate universe:
   - Exact constitutional isomers (50% target)
   - High-cosine / low-peak false analogs (25% target)
   - Mass isobars <= 20 ppm (25% target)
2. Targeted spectral evidence lookup: only evaluates reference spectra for the
   positive and negative candidates (O(1) lookups instead of O(N_lib)), keeping
   RAM strictly < 4 GB.
3. 1,035-D CrossModalRerankerV2 combining neural representations, Morgan similarities,
   and 10-D physical/spectral evidence features (including peak-to-cosine ratio).

Usage:
    python scripts/train_stage5_v2.py --epochs 8 --batch-size 64 --exp-name exp5_v2
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

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
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, DataStructs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from src.core.config import ARTIFACTS_DIR, TRAIN_PATH
from src.core.canonical_benchmark import CanonicalBenchmark
from src.data.mol_graph import smiles_to_graph
from src.data.hard_negative_dataset_v2 import (
    Catalog277KIndex,
    extract_candidate_evidence_vector,
    triplet_collate_fn_v2,
)
from src.models.molecule_encoder import MoleculeGNN
from src.models.reranker_v2 import CrossModalRerankerV2
from src.models.spectrum_encoder import SpectrumEncoder
from scripts.evaluate_external_library_diagnostic import (
    CompactSpectralLibrary,
    fast_mutual_cosine,
)


class PrecomputedTripletDataset(Dataset):
    """Memory-efficient precomputed triplet dataset."""

    def __init__(self, triplets: list[dict[str, Any]], graph_cache: dict[str, Data]):
        self.triplets = triplets
        self.graph_cache = graph_cache

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int):
        t = self.triplets[idx]
        pos_smi = t["pos_smi"]
        neg_smi = t["neg_smi"]

        if pos_smi not in self.graph_cache:
            self.graph_cache[pos_smi] = smiles_to_graph(pos_smi)
        if neg_smi not in self.graph_cache:
            self.graph_cache[neg_smi] = smiles_to_graph(neg_smi)

        return (
            t["spec_tensor"],
            self.graph_cache[pos_smi],
            self.graph_cache[neg_smi],
            t["pos_morgan"],
            t["neg_morgan"],
            t["pos_ev"],
            t["neg_ev"],
            t["tier"],
        )


def train_one_epoch_v2(
    spec_encoder: nn.Module,
    mol_encoder: nn.Module,
    reranker: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float = 0.20,
    fine_tune_gnn: bool = True,
) -> dict[str, float]:
    """Train reranker for one epoch on Stage 5 v2 hard-negative triplets."""
    spec_encoder.eval()
    if fine_tune_gnn:
        mol_encoder.train()
    else:
        mol_encoder.eval()
    reranker.train()

    total_loss = 0.0
    total_pairs = 0
    correct_pairs = 0
    tier_counts: dict[str, int] = {}
    tier_correct: dict[str, int] = {}

    for batch in loader:
        (
            specs,
            pos_graphs,
            neg_graphs,
            pos_morgans,
            neg_morgans,
            pos_ev,
            neg_ev,
            tiers,
        ) = batch

        specs = specs.to(device)
        pos_graphs = pos_graphs.to(device)
        neg_graphs = neg_graphs.to(device)
        pos_morgans = pos_morgans.to(device)
        neg_morgans = neg_morgans.to(device)
        pos_ev = pos_ev.to(device)
        neg_ev = neg_ev.to(device)

        with torch.no_grad():
            z_spec = spec_encoder(specs)

        if fine_tune_gnn:
            z_pos = mol_encoder(pos_graphs)
            z_neg = mol_encoder(neg_graphs)
        else:
            with torch.no_grad():
                z_pos = mol_encoder(pos_graphs)
                z_neg = mol_encoder(neg_graphs)

        s_pos = reranker(z_spec, z_pos, pos_morgans, pos_ev)
        s_neg = reranker(z_spec, z_neg, neg_morgans, neg_ev)

        loss = CrossModalRerankerV2.margin_loss(s_pos, s_neg, margin=margin)

        optimizer.zero_grad()
        loss.backward()
        if fine_tune_gnn:
            torch.nn.utils.clip_grad_norm_(mol_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(reranker.parameters(), max_norm=1.0)
        optimizer.step()

        b_size = specs.size(0)
        total_loss += float(loss.item()) * b_size
        total_pairs += b_size

        is_correct = (s_pos > s_neg).cpu().numpy()
        correct_pairs += int(is_correct.sum())

        for c, t in zip(is_correct, tiers):
            tier_counts[t] = tier_counts.get(t, 0) + 1
            if c:
                tier_correct[t] = tier_correct.get(t, 0) + 1

    metrics = {
        "train_loss": total_loss / max(1, total_pairs),
        "train_acc": correct_pairs / max(1, total_pairs),
    }
    for t in tier_counts:
        metrics[f"acc_{t}"] = tier_correct.get(t, 0) / tier_counts[t]

    return metrics


def evaluate_val_discrimination(
    spec_encoder: nn.Module,
    mol_encoder: nn.Module,
    reranker: nn.Module,
    loader: DataLoader,
    device: torch.device,
    margin: float = 0.20,
) -> dict[str, float]:
    """Evaluate pairwise discrimination accuracy and margin loss on validation triplets."""
    spec_encoder.eval()
    mol_encoder.eval()
    reranker.eval()

    total_loss = 0.0
    total_pairs = 0
    correct_pairs = 0
    tier_counts: dict[str, int] = {}
    tier_correct: dict[str, int] = {}

    with torch.no_grad():
        for batch in loader:
            (
                specs,
                pos_graphs,
                neg_graphs,
                pos_morgans,
                neg_morgans,
                pos_ev,
                neg_ev,
                tiers,
            ) = batch

            specs = specs.to(device)
            pos_graphs = pos_graphs.to(device)
            neg_graphs = neg_graphs.to(device)
            pos_morgans = pos_morgans.to(device)
            neg_morgans = neg_morgans.to(device)
            pos_ev = pos_ev.to(device)
            neg_ev = neg_ev.to(device)

            z_spec = spec_encoder(specs)
            z_pos = mol_encoder(pos_graphs)
            z_neg = mol_encoder(neg_graphs)

            s_pos = reranker(z_spec, z_pos, pos_morgans, pos_ev)
            s_neg = reranker(z_spec, z_neg, neg_morgans, neg_ev)

            loss = CrossModalRerankerV2.margin_loss(s_pos, s_neg, margin=margin)

            b_size = specs.size(0)
            total_loss += float(loss.item()) * b_size
            total_pairs += b_size

            is_correct = (s_pos > s_neg).cpu().numpy()
            correct_pairs += int(is_correct.sum())

            for c, t in zip(is_correct, tiers):
                tier_counts[t] = tier_counts.get(t, 0) + 1
                if c:
                    tier_correct[t] = tier_correct.get(t, 0) + 1

    metrics = {
        "val_loss": total_loss / max(1, total_pairs),
        "val_acc": correct_pairs / max(1, total_pairs),
    }
    for t in tier_counts:
        metrics[f"val_acc_{t}"] = tier_correct.get(t, 0) / tier_counts[t]

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Stage 5 v2 Hard-Negative Training")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-gnn", type=float, default=5e-5)
    parser.add_argument("--lr-reranker", type=float, default=5e-4)
    parser.add_argument("--n-train", type=int, default=3000, help="Number of training triplets")
    parser.add_argument("--n-val", type=int, default=600, help="Number of validation triplets")
    parser.add_argument("--exp-name", type=str, default="exp5_v2")
    args = parser.parse_args()

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 85)
    print("  STAGE 5 v2 TRAINING: HARD-NEGATIVE ISOMER & FALSE-ANALOG RERANKER")
    print(f"  Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | Triplets: {args.n_train}")
    print("=" * 85, flush=True)

    out_dir = ARTIFACTS_DIR / "stage05" / args.exp_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load 277K Candidate Catalog
    print("\n[1/5] Ingesting 277K Candidate Catalog & building chemical index...", flush=True)
    t0 = time.time()
    df_cand = pd.read_parquet(ROOT / "kaggle_dataset" / "candidate_library.parquet")
    catalog_index = Catalog277KIndex(df_cand)
    print(f"Loaded {len(df_cand):,} candidates and built index in {time.time()-t0:.2f}s.")

    # 2. Ingest Unified External Library
    ref_paths = [
        ROOT / "kaggle_dataset" / "reference_library_multice.parquet",
        ROOT / "artifacts" / "baseline" / "reference_library_multice.parquet",
    ]
    ref_path = next((p for p in ref_paths if p.exists()), ref_paths[0])
    df_ref = pd.read_parquet(ref_path, columns=["normalized_smiles", "neutral_mass", "precursor_mz", "collision_energy", "ms2_mzs", "ms2_intensities"])

    ext_path = ROOT / "artifacts" / "external" / "external_spectra.parquet"
    df_ext = pd.read_parquet(ext_path, columns=[
        "canonical_smiles", "neutral_mass", "precursor_mz", "collision_energy",
        "peaks_mz", "peaks_intensity", "source_library", "n_supporting_spectra", "source_count"
    ])

    d_smiles = np.concatenate([df_ref["normalized_smiles"].to_numpy(), df_ext["canonical_smiles"].to_numpy()])
    d_masses = np.concatenate([df_ref["neutral_mass"].to_numpy(dtype=np.float64), df_ext["neutral_mass"].to_numpy(dtype=np.float64)])
    d_precs = np.concatenate([df_ref["precursor_mz"].to_numpy(dtype=np.float64), df_ext["precursor_mz"].to_numpy(dtype=np.float64)])
    d_ces = np.concatenate([df_ref["collision_energy"].to_numpy(dtype=np.float32), df_ext["collision_energy"].to_numpy(dtype=np.float32)])
    d_mzs = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_mzs"]] + [np.asarray(x, dtype=np.float32) for x in df_ext["peaks_mz"]]
    d_ints = [np.asarray(x, dtype=np.float32) for x in df_ref["ms2_intensities"]] + [np.asarray(x, dtype=np.float32) for x in df_ext["peaks_intensity"]]
    d_sup = np.concatenate([np.ones(len(df_ref), dtype=np.int16), df_ext["n_supporting_spectra"].to_numpy(dtype=np.int16)])
    d_src = np.concatenate([np.ones(len(df_ref), dtype=np.int8), df_ext["source_count"].to_numpy(dtype=np.int8)])

    lib_ext = CompactSpectralLibrary(d_smiles, d_masses, d_precs, d_ces, d_mzs, d_ints, d_sup, d_src)
    print(f"Unified External Library indexed: {len(lib_ext.neutral_masses):,} spectra.")

    # 3. Load Cross-Modal Benchmark for molecule pool
    print("\n[3/5] Loading Cross-Modal Training pool & building fast triplets...", flush=True)
    bm = CanonicalBenchmark(subset_size=10000, n_benchmark_queries=200, split_seed=42, benchmark_seed=123)

    train_mols = sorted(list(bm.train_ds.selected_mols))
    val_mols = sorted(list(bm.val_ds.selected_mols))

    # Fast targeted triplet builder
    def build_fast_triplets(ds, mols, target_count: int, seed: int = 42, is_val: bool = False) -> list[dict]:
        rng = np.random.default_rng(seed)
        mol_samples = defaultdict(list)
        for idx, (s_info, mol_id) in enumerate(ds.samples):
            if mol_id in ds.graph_cache:
                mol_samples[mol_id].append((idx, s_info))

        available_mols = [m for m in mols if m in mol_samples and mol_samples[m]]
        selected_mols = rng.choice(available_mols, size=min(target_count, len(available_mols)), replace=False)

        tier_weights = ["isomer", "isomer", "false_analog", "isobar"]
        triplets = []

        for m in selected_mols:
            s_idx, spec_info = mol_samples[m][0]
            prec_mz = spec_info.get("precursor_mz", 0.0)
            q_ce = spec_info.get("ce", float("nan"))
            q_mzs = np.asarray(spec_info["mz"], dtype=np.float32)
            q_ints = np.asarray(spec_info["intensity"], dtype=np.float32)
            pos_smi = ds.mol_smiles[m]

            pos_mol = Chem.MolFromSmiles(pos_smi) if pos_smi else None
            if pos_mol is None:
                continue
            pos_formula = rdMolDescriptors.CalcMolFormula(pos_mol)
            pos_mass = float(rdMolDescriptors.CalcExactMolWt(pos_mol))

            # Sample target tier
            t_tier = rng.choice(tier_weights)
            neg_idx, actual_tier = catalog_index.sample_negative(
                pos_smi=pos_smi,
                pos_formula=pos_formula,
                pos_mass=pos_mass,
                target_tier=t_tier,
                rng=rng,
            )
            neg_smi = catalog_index.smiles_list[neg_idx]
            neg_formula = catalog_index.formula_list[neg_idx]
            neg_mass = catalog_index.mass_array[neg_idx]

            # Targeted spectral lookup: ONLY check pos_smi and neg_smi!
            target_smis = {pos_smi, neg_smi}
            l_idx, r_idx = lib_ext.query_window(pos_mass, ppm=20.0)

            pos_hit = None
            neg_hit = None
            top_ref_smi = ""
            top_ref_cos = 0.0

            if r_idx > l_idx:
                for ri in range(l_idx, r_idx):
                    ref_smi = lib_ext.smiles[ri]
                    if is_val and ref_smi == pos_smi:
                        continue  # Prevent validation query-vs-itself evidence leakage (C1)
                    if ref_smi in target_smis:
                        delta = prec_mz - lib_ext.precursor_mzs[ri]
                        cos_sim, n_peaks = fast_mutual_cosine(
                            q_mzs, q_ints, lib_ext.mzs_list[ri], lib_ext.intens_list[ri], delta=delta
                        )
                        if cos_sim > 0.10:
                            r_ce = lib_ext.collision_energies[ri]
                            ce_diff = abs(q_ce - r_ce) if (np.isfinite(q_ce) and np.isfinite(r_ce)) else float("nan")
                            hit_data = {
                                "cos": cos_sim,
                                "n_peaks": n_peaks,
                                "ce_diff": ce_diff,
                                "n_supporting": int(lib_ext.n_supporting[ri]),
                                "source_count": int(lib_ext.source_counts[ri]),
                            }
                            if ref_smi == pos_smi:
                                if pos_hit is None or cos_sim > pos_hit["cos"]:
                                    pos_hit = hit_data
                            elif ref_smi == neg_smi:
                                if neg_hit is None or cos_sim > neg_hit["cos"]:
                                    neg_hit = hit_data

                            if cos_sim > top_ref_cos:
                                top_ref_cos = cos_sim
                                top_ref_smi = ref_smi

            # Morgan similarity to top ref
            pos_morgan = 1.0
            neg_morgan = 0.0
            if top_ref_smi:
                ref_mol = Chem.MolFromSmiles(top_ref_smi)
                if ref_mol:
                    ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                    neg_mol = Chem.MolFromSmiles(neg_smi)
                    if pos_mol and ref_fp:
                        p_fp = AllChem.GetMorganFingerprintAsBitVect(pos_mol, 2, nBits=1024)
                        pos_morgan = float(DataStructs.TanimotoSimilarity(ref_fp, p_fp))
                    if neg_mol and ref_fp:
                        n_fp = AllChem.GetMorganFingerprintAsBitVect(neg_mol, 2, nBits=1024)
                        neg_morgan = float(DataStructs.TanimotoSimilarity(ref_fp, n_fp))

            pos_ppm = abs(prec_mz - (pos_mass + 1.0078)) / (pos_mass + 1.0078) * 1e6
            neg_ppm = abs(prec_mz - (neg_mass + 1.0078)) / (neg_mass + 1.0078) * 1e6

            pos_ev = extract_candidate_evidence_vector(
                hit=pos_hit,
                ppm_error=pos_ppm,
                tier=1 if pos_ppm <= 20.0 else 2,
                prec_mz=prec_mz,
                is_isomer=True,
            )
            neg_ev = extract_candidate_evidence_vector(
                hit=neg_hit,
                ppm_error=neg_ppm,
                tier=1 if neg_ppm <= 20.0 else 2,
                prec_mz=prec_mz,
                is_isomer=(neg_formula == pos_formula),
            )

            spec_tensor = ds[s_idx][0]
            triplets.append({
                "spec_tensor": spec_tensor,
                "pos_smi": pos_smi,
                "neg_smi": neg_smi,
                "pos_morgan": pos_morgan,
                "neg_morgan": neg_morgan,
                "pos_ev": torch.from_numpy(pos_ev),
                "neg_ev": torch.from_numpy(neg_ev),
                "tier": actual_tier,
            })

        return triplets

    t_trip = time.time()
    train_triplets = build_fast_triplets(bm.train_ds, train_mols, target_count=args.n_train, seed=42, is_val=False)
    val_triplets = build_fast_triplets(bm.val_ds, val_mols, target_count=args.n_val, seed=123, is_val=True)
    print(f"Built {len(train_triplets):,} training triplets and {len(val_triplets):,} validation triplets in {time.time()-t_trip:.1f}s!")

    # Free memory
    del df_ref, df_ext, d_smiles, d_masses, d_precs, d_ces, d_mzs, d_ints
    gc.collect()

    graph_cache = {**bm.train_ds.graph_cache, **bm.val_ds.graph_cache}
    train_dataset = PrecomputedTripletDataset(train_triplets, graph_cache)
    val_dataset = PrecomputedTripletDataset(val_triplets, graph_cache)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=triplet_collate_fn_v2,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=triplet_collate_fn_v2,
        num_workers=0,
    )

    # 4. Initialize Models
    print("\n[4/5] Initializing Stage 5 v2 Models...", flush=True)

    spec_encoder = SpectrumEncoder(embed_dim=256).to(device)
    spec_ckpt = ARTIFACTS_DIR / "stage02/exp2a/checkpoints/best.pt"
    spec_data = torch.load(spec_ckpt, map_location=device, weights_only=False)
    spec_encoder.load_state_dict(spec_data.get("model_state_dict", spec_data))
    spec_encoder.eval()

    mol_encoder = MoleculeGNN(embed_dim=256).to(device)
    gnn_ckpt = ARTIFACTS_DIR / "stage03/exp3a/checkpoints/best.pt"
    gnn_data = torch.load(gnn_ckpt, map_location=device, weights_only=False)
    mol_encoder.load_state_dict(gnn_data.get("model_state_dict", gnn_data))

    reranker = CrossModalRerankerV2(embed_dim=256, evidence_dim=10, hidden_dim=256, dropout=0.10).to(device)

    optimizer = torch.optim.AdamW([
        {"params": mol_encoder.parameters(), "lr": args.lr_gnn, "weight_decay": 0.01},
        {"params": reranker.parameters(), "lr": args.lr_reranker, "weight_decay": 0.01},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Training Loop
    print("\n[5/5] Training Stage 5 v2 across hard-negative tiers...", flush=True)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Train Acc':<11} | {'Val Loss':<10} | {'Val Acc':<9} | {'Isomer Acc':<12} | {'Time':<6}")
    print("-" * 75)

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        train_m = train_one_epoch_v2(
            spec_encoder=spec_encoder,
            mol_encoder=mol_encoder,
            reranker=reranker,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            margin=0.20,
            fine_tune_gnn=True,
        )
        scheduler.step()

        val_m = evaluate_val_discrimination(
            spec_encoder=spec_encoder,
            mol_encoder=mol_encoder,
            reranker=reranker,
            loader=val_loader,
            device=device,
            margin=0.20,
        )

        ep_time = time.time() - t_ep
        iso_acc = val_m.get("val_acc_isomer", 0.0) * 100.0

        print(
            f"{epoch:<6d} | {train_m['train_loss']:<12.4f} | {train_m['train_acc']*100:<10.2f}% | "
            f"{val_m['val_loss']:<10.4f} | {val_m['val_acc']*100:<8.2f}% | "
            f"{iso_acc:<11.2f}% | {ep_time:<5.1f}s",
            flush=True,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_m["train_loss"],
            "train_acc": train_m["train_acc"],
            "val_loss": val_m["val_loss"],
            "val_acc": val_m["val_acc"],
            "val_acc_isomer": val_m.get("val_acc_isomer", 0.0),
        })

        if val_m["val_acc"] > best_val_acc:
            best_val_acc = val_m["val_acc"]
            best_ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "mol_encoder_state_dict": mol_encoder.state_dict(),
                "reranker_state_dict": reranker.state_dict(),
                "val_acc": best_val_acc,
                "metrics": val_m,
            }, best_ckpt_path)

    # Save last checkpoint and training history
    last_ckpt_path = ckpt_dir / "last.pt"
    torch.save({
        "epoch": args.epochs,
        "mol_encoder_state_dict": mol_encoder.state_dict(),
        "reranker_state_dict": reranker.state_dict(),
        "val_acc": val_m["val_acc"],
    }, last_ckpt_path)

    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 85)
    print(f"  Stage 5 v2 Training Complete in {time.time()-t_start:.1f}s!")
    print(f"  Best Validation Discrimination Accuracy: {best_val_acc*100:.2f}%")
    print(f"  Checkpoint saved to: {best_ckpt_path}")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
