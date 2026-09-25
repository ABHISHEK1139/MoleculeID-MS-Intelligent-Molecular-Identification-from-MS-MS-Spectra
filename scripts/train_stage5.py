"""Stage 5 Training: Hard-Negative Isomer Reranker.

Usage:
    python scripts/train_stage5.py --config configs/stage05/exp5a.yaml --exp-name exp5a

The training pipeline:
1. Loads frozen Stage 2 SpectrumEncoder (artifacts/stage02/exp2a/checkpoints/best.pt).
2. Loads pretrained Stage 3 MoleculeGNN (artifacts/stage03/exp3a/checkpoints/best.pt).
3. Trains CrossModalReranker and fine-tunes MoleculeGNN with 4-tier curriculum
   (random -> isobar -> isomer -> scaffold-isomer).
4. Evaluates online on held-out isomer discrimination pairs and validation margin loss.
5. Saves best checkpoint to artifacts/stage05/exp5a/checkpoints/best.pt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ARTIFACTS_DIR, TRAIN_PATH
from src.models.spectrum_encoder import SpectrumEncoder
from src.models.molecule_encoder import MoleculeGNN
from src.models.reranker import CrossModalReranker
from src.data.cross_modal_dataset import create_cross_modal_datasets
from src.data.hard_negative_dataset import (
    HardNegativeIndex,
    HardNegativeDataset,
    triplet_collate_fn,
)


def train_one_epoch(
    spec_encoder: nn.Module,
    mol_encoder: nn.Module,
    reranker: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float = 0.2,
    fine_tune_gnn: bool = True,
) -> dict[str, float]:
    """Train reranker for one epoch on hard-negative triplets."""
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
        specs, pos_graphs, neg_graphs, pos_phys, neg_phys, mol_pos, mol_neg, tiers = batch

        specs = specs.to(device)
        pos_graphs = pos_graphs.to(device)
        neg_graphs = neg_graphs.to(device)
        pos_phys = pos_phys.to(device)
        neg_phys = neg_phys.to(device)

        with torch.no_grad():
            z_spec = spec_encoder(specs)

        if fine_tune_gnn:
            z_pos = mol_encoder(pos_graphs)
            z_neg = mol_encoder(neg_graphs)
        else:
            with torch.no_grad():
                z_pos = mol_encoder(pos_graphs)
                z_neg = mol_encoder(neg_graphs)

        s_pos = reranker(z_spec, z_pos, pos_phys)
        s_neg = reranker(z_spec, z_neg, neg_phys)

        loss = CrossModalReranker.margin_loss(s_pos, s_neg, margin=margin)

        optimizer.zero_grad()
        loss.backward()
        if fine_tune_gnn:
            torch.nn.utils.clip_grad_norm_(mol_encoder.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(reranker.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = s_pos.size(0)
        total_loss += loss.item() * batch_size
        total_pairs += batch_size

        is_correct = (s_pos > s_neg).cpu().numpy()
        correct_pairs += int(np.sum(is_correct))

        for t, c in zip(tiers, is_correct):
            t_clean = "isomer" if "isomer" in t else t
            tier_counts[t_clean] = tier_counts.get(t_clean, 0) + 1
            if c:
                tier_correct[t_clean] = tier_correct.get(t_clean, 0) + 1

        batch_idx = total_pairs // batch_size
        if batch_idx % 200 == 0:
            print(f"  [step {batch_idx:04d}/{len(loader)}] "
                  f"running_loss: {total_loss / total_pairs:.4f} | "
                  f"running_acc: {correct_pairs / total_pairs * 100:.1f}%", flush=True)

    metrics = {
        "train_loss": total_loss / max(total_pairs, 1),
        "train_accuracy": correct_pairs / max(total_pairs, 1),
    }
    for t in tier_counts:
        metrics[f"train_acc_{t}"] = tier_correct.get(t, 0) / max(tier_counts[t], 1)

    return metrics


@torch.no_grad()
def evaluate_validation(
    spec_encoder: nn.Module,
    mol_encoder: nn.Module,
    reranker: nn.Module,
    val_dataset: HardNegativeDataset,
    device: torch.device,
    margin: float = 0.2,
    n_val_samples: int = 1000,
) -> dict[str, float]:
    """Evaluate on held-out validation triplets and specifically on exact isomers."""
    spec_encoder.eval()
    mol_encoder.eval()
    reranker.eval()

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=triplet_collate_fn,
    )

    total_loss = 0.0
    total_pairs = 0
    correct_pairs = 0

    isomer_pairs = 0
    isomer_correct = 0

    processed = 0
    for batch in val_loader:
        if processed >= n_val_samples:
            break

        specs, pos_graphs, neg_graphs, pos_phys, neg_phys, mol_pos, mol_neg, tiers = batch
        specs = specs.to(device)
        pos_graphs = pos_graphs.to(device)
        neg_graphs = neg_graphs.to(device)
        pos_phys = pos_phys.to(device)
        neg_phys = neg_phys.to(device)

        z_spec = spec_encoder(specs)
        z_pos = mol_encoder(pos_graphs)
        z_neg = mol_encoder(neg_graphs)

        s_pos = reranker(z_spec, z_pos, pos_phys)
        s_neg = reranker(z_spec, z_neg, neg_phys)

        loss = CrossModalReranker.margin_loss(s_pos, s_neg, margin=margin)

        b_size = s_pos.size(0)
        total_loss += loss.item() * b_size
        total_pairs += b_size

        is_correct = (s_pos > s_neg).cpu().numpy()
        correct_pairs += int(np.sum(is_correct))

        for t, c in zip(tiers, is_correct):
            if "isomer" in t:
                isomer_pairs += 1
                if c:
                    isomer_correct += 1

        processed += b_size

    return {
        "val_loss": total_loss / max(total_pairs, 1),
        "val_accuracy": correct_pairs / max(total_pairs, 1),
        "val_isomer_acc": isomer_correct / max(isomer_pairs, 1) if isomer_pairs > 0 else 0.50,
        "val_isomer_count": isomer_pairs,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 5: Hard-Negative Isomer Reranking")
    parser.add_argument("--config", type=str, default="configs/stage05/exp5a.yaml")
    parser.add_argument("--exp-name", type=str, default="exp5a")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_dir = ARTIFACTS_DIR / "stage05" / args.exp_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 5 Training [{args.exp_name}] on {device} ===", flush=True)

    # 1. Load Datasets
    data_cfg = cfg["data"]
    print(f"Loading datasets (subset_size={data_cfg.get('subset_size', 10000)})...", flush=True)
    base_train_ds, base_val_ds = create_cross_modal_datasets(
        train_path=data_cfg.get("train_path", TRAIN_PATH),
        subset_size=data_cfg.get("subset_size", 10000),
        val_frac=data_cfg.get("val_frac", 0.10),
        seed=data_cfg.get("seed", 42),
    )

    print("Building chemical negative indices...", flush=True)
    train_index = HardNegativeIndex(base_train_ds.selected_mols, base_train_ds.mol_smiles)
    val_index = HardNegativeIndex(base_val_ds.selected_mols, base_val_ds.mol_smiles)

    train_ds = HardNegativeDataset(base_train_ds, index=train_index, seed=data_cfg.get("seed", 42))
    val_ds = HardNegativeDataset(base_val_ds, index=val_index, seed=123)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=triplet_collate_fn,
    )

    # 2. Build / Load Models
    m_cfg = cfg["models"]

    # Spectrum Encoder (Frozen)
    spec_encoder = SpectrumEncoder(embed_dim=m_cfg.get("embed_dim", 256)).to(device)
    spec_ckpt = m_cfg.get("spectrum_checkpoint", "artifacts/stage02/exp2a/checkpoints/best.pt")
    print(f"Loading Stage 2 SpectrumEncoder from: {spec_ckpt}")
    ckpt_data = torch.load(spec_ckpt, map_location=device, weights_only=False)
    spec_state = ckpt_data.get("model_state_dict", ckpt_data)
    spec_encoder.load_state_dict(spec_state)
    spec_encoder.eval()
    for p in spec_encoder.parameters():
        p.requires_grad = False

    # Molecule GNN (Pretrained)
    mol_encoder = MoleculeGNN(embed_dim=m_cfg.get("embed_dim", 256)).to(device)
    gnn_ckpt = m_cfg.get("gnn_checkpoint", "artifacts/stage03/exp3a/checkpoints/best.pt")
    print(f"Loading Stage 3 MoleculeGNN from: {gnn_ckpt}")
    gnn_data = torch.load(gnn_ckpt, map_location=device, weights_only=False)
    gnn_state = gnn_data.get("model_state_dict", gnn_data)
    mol_encoder.load_state_dict(gnn_state)

    fine_tune_gnn = m_cfg.get("fine_tune_gnn", True)
    for p in mol_encoder.parameters():
        p.requires_grad = fine_tune_gnn

    # CrossModalReranker
    reranker = CrossModalReranker(
        embed_dim=m_cfg.get("embed_dim", 256),
        physics_dim=m_cfg.get("physics_dim", 4),
        hidden_dim=m_cfg.get("reranker_hidden_dim", 256),
        dropout=m_cfg.get("dropout", 0.10),
    ).to(device)

    print(f"SpectrumEncoder: {spec_encoder.num_parameters:,} params (FROZEN)", flush=True)
    print(f"MoleculeGNN:     {mol_encoder.num_parameters:,} params ({'FINE-TUNED' if fine_tune_gnn else 'FROZEN'})", flush=True)
    print(f"Reranker:        {reranker.num_parameters:,} params (TRAINABLE)", flush=True)

    # 3. Optimizer
    params = [{"params": reranker.parameters(), "lr": float(cfg["training"]["lr_reranker"])}]
    if fine_tune_gnn:
        params.append({"params": mol_encoder.parameters(), "lr": float(cfg["training"]["lr_gnn"])})

    optimizer = torch.optim.AdamW(params, weight_decay=float(cfg["training"]["weight_decay"]))
    epochs = cfg["training"]["epochs"]
    margin = float(cfg["training"]["margin"])
    curriculum = cfg["training"].get("curriculum", None)

    best_isomer_acc = 0.0
    history = []

    print(f"\nStarting training for {epochs} epochs...", flush=True)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        # Set curriculum probabilities
        train_ds.set_curriculum(epoch, curriculum)
        val_ds.set_curriculum(epoch, curriculum)

        train_metrics = train_one_epoch(
            spec_encoder=spec_encoder,
            mol_encoder=mol_encoder,
            reranker=reranker,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            margin=margin,
            fine_tune_gnn=fine_tune_gnn,
        )

        val_metrics = evaluate_validation(
            spec_encoder=spec_encoder,
            mol_encoder=mol_encoder,
            reranker=reranker,
            val_dataset=val_ds,
            device=device,
            margin=margin,
        )

        ep_duration = time.time() - ep_start
        iso_acc = val_metrics["val_isomer_acc"]
        is_best = iso_acc > best_isomer_acc
        if is_best:
            best_isomer_acc = iso_acc
            torch.save({
                "epoch": epoch,
                "reranker_state_dict": reranker.state_dict(),
                "mol_encoder_state_dict": mol_encoder.state_dict(),
                "val_isomer_acc": iso_acc,
                "val_metrics": val_metrics,
                "config": cfg,
            }, ckpt_dir / "best.pt")

        log_entry = {
            "epoch": epoch,
            "duration_s": round(ep_duration, 1),
            "tier_probs": train_ds.tier_probs,
            **train_metrics,
            **val_metrics,
            "is_best": is_best,
        }
        history.append(log_entry)

        print(f"Epoch {epoch:02d}/{epochs:02d} [{ep_duration:.1f}s] "
              f"TrainLoss: {train_metrics['train_loss']:.4f} | "
              f"TrainAcc: {train_metrics['train_accuracy']*100:.1f}% "
              f"(iso: {train_metrics.get('train_acc_isomer', 0.0)*100:.1f}%) | "
              f"ValLoss: {val_metrics['val_loss']:.4f} | "
              f"ValIsomerAcc: {iso_acc*100:.2f}% "
              f"{'[BEST]' if is_best else ''}", flush=True)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.1f} minutes. Best Val Isomer Acc: {best_isomer_acc*100:.2f}%", flush=True)

    with open(exp_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Save last checkpoint
    torch.save({
        "epoch": epochs,
        "reranker_state_dict": reranker.state_dict(),
        "mol_encoder_state_dict": mol_encoder.state_dict(),
        "history": history,
        "config": cfg,
    }, ckpt_dir / "last.pt")


if __name__ == "__main__":
    main()
