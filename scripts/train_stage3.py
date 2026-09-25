"""Stage 3 training: Cross-Modal Spectrum ↔ Molecule GNN.

Usage:
    python scripts/train_stage3.py --config configs/stage03/exp3a.yaml --exp-name exp3a

The training loop:
1. Loads frozen Stage 2 SpectrumEncoder checkpoint (exp2a best.pt)
2. Trains MoleculeGNN via symmetric cross-modal InfoNCE loss
3. Periodic evaluation:
   - In-batch cross-modal accuracy (s->m and m->s)
   - Protocol C Zero-Reference Retrieval benchmark on held-out validation molecules
4. Saves best checkpoint by Protocol C MRR@25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ARTIFACTS_DIR, TRAIN_PATH
from src.core.evaluation import summarize_ranks
from src.models.spectrum_encoder import SpectrumEncoder
from src.models.molecule_encoder import MoleculeGNN
from src.models.cross_modal_loss import cross_modal_infonce_loss, cross_modal_accuracy
from src.data.cross_modal_dataset import CrossModalDataset, create_cross_modal_datasets, cross_modal_collate_fn


def train_one_epoch(
    spec_encoder: nn.Module,
    mol_encoder: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 0.07,
) -> dict[str, float]:
    """Train GNN for one epoch; return loss and accuracy metrics."""
    mol_encoder.train()
    spec_encoder.eval()

    total_loss = 0.0
    total_acc_s2m = 0.0
    total_acc_m2s = 0.0
    n_batches = 0

    for specs, graphs, _ in loader:
        specs = specs.to(device)
        graphs = graphs.to(device)

        with torch.no_grad():
            z_spec = spec_encoder(specs)

        z_mol = mol_encoder(graphs)

        loss = cross_modal_infonce_loss(z_spec, z_mol, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mol_encoder.parameters(), max_norm=1.0)
        optimizer.step()

        acc_s2m, acc_m2s = cross_modal_accuracy(z_spec.detach(), z_mol.detach())

        total_loss += loss.item()
        total_acc_s2m += acc_s2m
        total_acc_m2s += acc_m2s
        n_batches += 1

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "train_acc_s2m": total_acc_s2m / max(n_batches, 1),
        "train_acc_m2s": total_acc_m2s / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    spec_encoder: nn.Module,
    mol_encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 0.07,
) -> dict[str, float]:
    """Validate on held-out spectrum-molecule pairs."""
    mol_encoder.eval()
    spec_encoder.eval()

    total_loss = 0.0
    total_acc_s2m = 0.0
    total_acc_m2s = 0.0
    n_batches = 0

    for specs, graphs, _ in loader:
        specs = specs.to(device)
        graphs = graphs.to(device)

        z_spec = spec_encoder(specs)
        z_mol = mol_encoder(graphs)

        loss = cross_modal_infonce_loss(z_spec, z_mol, temperature=temperature)
        acc_s2m, acc_m2s = cross_modal_accuracy(z_spec, z_mol)

        total_loss += loss.item()
        total_acc_s2m += acc_s2m
        total_acc_m2s += acc_m2s
        n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_acc_s2m": total_acc_s2m / max(n_batches, 1),
        "val_acc_m2s": total_acc_m2s / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_protocol_c(
    spec_encoder: nn.Module,
    mol_encoder: nn.Module,
    val_dataset: CrossModalDataset,
    device: torch.device,
    n_queries: int = 200,
    top_k: int = 25,
) -> dict[str, float]:
    """Protocol C Zero-Reference Benchmark:

    Given query spectra of held-out validation molecules (whose reference spectra
    were NEVER seen during training), rank candidate molecular graphs using cross-modal
    similarity: cos(z_spec, z_mol).

    Returns:
        MRR@25, Hit@1, Hit@5, Hit@25
    """
    mol_encoder.eval()
    spec_encoder.eval()

    # 1. Encode all unique validation candidate molecular graphs
    candidate_mols = [m for m in val_dataset.selected_mols if m in val_dataset.graph_cache]
    if not candidate_mols:
        return {"mrr": 0.0, "hit@1": 0.0, "hit@5": 0.0, "hit@25": 0.0}
    candidate_graphs = [val_dataset.graph_cache[m] for m in candidate_mols]

    mol_embs_list = []
    batch_size = 64
    for i in range(0, len(candidate_graphs), batch_size):
        batch_g = Batch.from_data_list(candidate_graphs[i:i + batch_size]).to(device)
        z_m = mol_encoder(batch_g)
        mol_embs_list.append(z_m.cpu())
    cand_embs = torch.cat(mol_embs_list, dim=0).numpy()  # (M_cand, D)

    # 2. Select query spectra from validation set
    rng = np.random.default_rng(123)
    n_q = min(n_queries, len(val_dataset))
    query_indices = rng.permutation(len(val_dataset))[:n_q]

    query_specs_list = []
    query_true_mols = []
    for q_idx in query_indices:
        spec_tensor, _, true_mol = val_dataset[q_idx]
        query_specs_list.append(spec_tensor)
        query_true_mols.append(true_mol)

    query_specs = torch.stack(query_specs_list, dim=0).to(device)
    z_queries = spec_encoder(query_specs).cpu().numpy()  # (n_queries, D)

    # 3. Vectorized Ranking
    # sims: (n_queries, M_cand)
    sims = z_queries @ cand_embs.T

    ranks = []
    mol_to_cand_idx = {m: i for i, m in enumerate(candidate_mols)}

    k = min(top_k, len(candidate_mols))
    row_idx = np.arange(len(query_true_mols))[:, None]
    if k < sims.shape[1]:
        top_k_partition = np.argpartition(-sims, k, axis=1)[:, :k]
        top_k_sorted = top_k_partition[row_idx, np.argsort(-sims[row_idx, top_k_partition], axis=1)]
    else:
        top_k_sorted = np.argsort(-sims, axis=1)[:, :k]

    for i, true_mol in enumerate(query_true_mols):
        target_idx = mol_to_cand_idx[true_mol]
        ranked_indices = top_k_sorted[i]
        matches = np.where(ranked_indices == target_idx)[0]
        rank = int(matches[0] + 1) if len(matches) > 0 else 0
        ranks.append(rank)

    metrics = summarize_ranks(ranks, k=k)
    return metrics


def main():
    # First pass: parse --config if provided
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default="configs/stage03/exp3a.yaml", help="Path to config YAML")
    conf_args, remaining_argv = conf_parser.parse_known_args()

    defaults = {
        "exp_name": "exp3a",
        "subset": 10000,
        "epochs": 20,
        "batch_size": 128,
        "lr": 1e-4,
        "temperature": 0.07,
        "gnn_hidden": 128,
        "gnn_layers": 4,
        "embed_dim": 256,
        "dropout": 0.10,
        "eval_every": 5,
        "n_protocol_c_queries": 200,
        "spectrum_checkpoint": "artifacts/stage02/exp2a/checkpoints/best.pt",
        "seed": 42,
        "max_row_groups": None,
    }

    # Load YAML defaults if available
    if conf_args.config and Path(conf_args.config).exists():
        import yaml
        with open(conf_args.config) as f:
            cfg = yaml.safe_load(f)
        if "data" in cfg:
            defaults["subset"] = cfg["data"].get("subset_size", defaults["subset"])
            defaults["seed"] = cfg["data"].get("seed", defaults["seed"])
        if "training" in cfg:
            defaults["epochs"] = cfg["training"].get("epochs", defaults["epochs"])
            defaults["batch_size"] = cfg["training"].get("batch_size", defaults["batch_size"])
            defaults["lr"] = float(cfg["training"].get("lr", defaults["lr"]))
            defaults["temperature"] = float(cfg["training"].get("temperature", defaults["temperature"]))
        if "models" in cfg:
            defaults["gnn_hidden"] = cfg["models"].get("gnn_hidden_dim", defaults["gnn_hidden"])
            defaults["gnn_layers"] = cfg["models"].get("gnn_n_layers", defaults["gnn_layers"])
            defaults["embed_dim"] = cfg["models"].get("embed_dim", defaults["embed_dim"])
            defaults["dropout"] = float(cfg["models"].get("dropout", defaults["dropout"]))
            defaults["spectrum_checkpoint"] = cfg["models"].get("spectrum_checkpoint", defaults["spectrum_checkpoint"])
        if "evaluation" in cfg:
            defaults["eval_every"] = cfg["evaluation"].get("eval_every", defaults["eval_every"])
            defaults["n_protocol_c_queries"] = cfg["evaluation"].get("n_protocol_c_queries", defaults["n_protocol_c_queries"])
        print(f"[stage3] loaded config defaults from {conf_args.config}")

    # Second pass: CLI arguments override defaults
    parser = argparse.ArgumentParser(parents=[conf_parser], description="Stage 3: Cross-Modal Spectrum ↔ Molecule GNN")
    parser.set_defaults(**defaults)
    parser.add_argument("--exp-name", type=str, help="Experiment name")
    parser.add_argument("--subset", type=int, help="Number of molecules to use")
    parser.add_argument("--epochs", type=int, help="Training epochs")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--temperature", type=float, help="InfoNCE temperature")
    parser.add_argument("--gnn-hidden", type=int, help="GNN hidden channels")
    parser.add_argument("--gnn-layers", type=int, help="GNN layers")
    parser.add_argument("--embed-dim", type=int, help="Shared embedding dimension")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--eval-every", type=int, help="Eval every N epochs")
    parser.add_argument("--n-protocol-c-queries", type=int, help="Number of Protocol C query spectra")
    parser.add_argument("--spectrum-checkpoint", type=str, help="Path to frozen SpectrumEncoder checkpoint")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--max-row-groups", type=int, help="Row groups limit")
    parser.add_argument("--eval-only", action="store_true", help="Only run Protocol C evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Stage 3 checkpoint to evaluate")
    args = parser.parse_args(remaining_argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage3] device={device}, subset={args.subset}, epochs={args.epochs}, batch={args.batch_size}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_dir = ARTIFACTS_DIR / "stage03" / args.exp_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Data ────────────────────────────────────────────────────────────
    print("[stage3] loading train and validation datasets in a single parquet pass...")
    train_dataset, val_dataset = create_cross_modal_datasets(
        train_path=TRAIN_PATH,
        subset_size=args.subset,
        val_frac=0.10,
        seed=args.seed,
        max_row_groups=args.max_row_groups,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=cross_modal_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=cross_modal_collate_fn,
        drop_last=False,
    )

    # ── 2. Models ──────────────────────────────────────────────────────────
    print("[stage3] loading frozen SpectrumEncoder...")
    spec_encoder = SpectrumEncoder(
        input_dim=1483,
        embed_dim=args.embed_dim,
        hidden_channels=128,
        n_blocks=4,
        dropout=0.1,
    ).to(device)

    ckpt_path = Path(args.spectrum_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SpectrumEncoder checkpoint not found at {ckpt_path}")

    spec_encoder.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    for p in spec_encoder.parameters():
        p.requires_grad = False
    spec_encoder.eval()
    print(f"  [*] loaded SpectrumEncoder from {ckpt_path} (FROZEN)")

    print("[stage3] initializing MoleculeGNN...")
    mol_encoder = MoleculeGNN(
        hidden_dim=args.gnn_hidden,
        embed_dim=args.embed_dim,
        n_layers=args.gnn_layers,
        dropout=args.dropout,
    ).to(device)
    print(f"  [*] MoleculeGNN: {mol_encoder.num_parameters:,} params ({mol_encoder.num_trainable:,} trainable)")

    if args.checkpoint:
        mol_encoder.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"  [*] loaded GNN checkpoint from {args.checkpoint}")

    if args.eval_only:
        print("[stage3] running Protocol C zero-reference evaluation...")
        proto_c = evaluate_protocol_c(spec_encoder, mol_encoder, val_dataset, device, n_queries=args.n_protocol_c_queries)
        print(f"[stage3] Protocol C: MRR@25={proto_c['mrr']:.4f} hit@1={proto_c['hit@1']:.3f} "
              f"hit@5={proto_c.get('hit@5', 0):.3f} hit@25={proto_c.get('hit@25', 0):.3f}")
        return

    # ── 3. Optimizer & Training ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(mol_encoder.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_proto_c_mrr = 0.0
    history: list[dict] = []

    print(f"[stage3] training GNN for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(spec_encoder, mol_encoder, train_loader, optimizer, device, args.temperature)
        val_metrics = validate(spec_encoder, mol_encoder, val_loader, device, args.temperature)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]

        epoch_metrics = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            "lr": lr_now,
            "elapsed_s": elapsed,
        }

        # Periodic Protocol C evaluation
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            proto_c = evaluate_protocol_c(spec_encoder, mol_encoder, val_dataset, device, n_queries=args.n_protocol_c_queries)
            epoch_metrics["protocol_c_mrr"] = proto_c["mrr"]
            epoch_metrics["protocol_c_hit1"] = proto_c["hit@1"]
            epoch_metrics["protocol_c_hit5"] = proto_c.get("hit@5", 0.0)
            epoch_metrics["protocol_c_hit25"] = proto_c.get("hit@25", 0.0)

            print(f"  epoch {epoch:3d} | loss={train_metrics['train_loss']:.4f} "
                  f"val_loss={val_metrics['val_loss']:.4f} "
                  f"train_acc={train_metrics['train_acc_s2m']:.3f} "
                  f"val_acc={val_metrics['val_acc_s2m']:.3f} "
                  f"ProtoC_MRR@25={proto_c['mrr']:.4f} ProtoC_Hit@1={proto_c['hit@1']:.3f} "
                  f"| {elapsed:.1f}s")

            if proto_c["mrr"] > best_proto_c_mrr:
                best_proto_c_mrr = proto_c["mrr"]
                torch.save(mol_encoder.state_dict(), ckpt_dir / "best.pt")
                print(f"  [*] new best Protocol C MRR@25={best_proto_c_mrr:.4f} -> saved best.pt")
        else:
            print(f"  epoch {epoch:3d} | loss={train_metrics['train_loss']:.4f} "
                  f"val_loss={val_metrics['val_loss']:.4f} "
                  f"train_acc={train_metrics['train_acc_s2m']:.3f} "
                  f"val_acc={val_metrics['val_acc_s2m']:.3f} "
                  f"| {elapsed:.1f}s")

        if epoch % 5 == 0:
            torch.save(mol_encoder.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

        history.append(epoch_metrics)

    results = {
        "config": {
            "subset_size": args.subset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "gnn_hidden": args.gnn_hidden,
            "gnn_layers": args.gnn_layers,
            "embed_dim": args.embed_dim,
            "spectrum_checkpoint": str(args.spectrum_checkpoint),
            "seed": args.seed,
            "gnn_parameters": mol_encoder.num_parameters,
        },
        "best_protocol_c_mrr": best_proto_c_mrr,
        "history": history,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(mol_encoder.state_dict(), ckpt_dir / "final.pt")
    print(f"\n[stage3] done. best Protocol C MRR@25={best_proto_c_mrr:.4f}")
    print(f"[stage3] artifacts saved to {exp_dir}")


if __name__ == "__main__":
    main()
