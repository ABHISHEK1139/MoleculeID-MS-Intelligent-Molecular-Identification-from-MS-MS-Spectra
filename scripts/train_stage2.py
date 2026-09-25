"""Stage 2 training: contrastive spectrum encoder.

Usage:
    python scripts/train_stage2.py                          # defaults (10K molecules)
    python scripts/train_stage2.py --subset 50000 --epochs 30
    python scripts/train_stage2.py --eval-only --checkpoint artifacts/stage02/exp2a/checkpoints/best.pt

The training loop:
1. Load spectra, group by molecule, form contrastive pairs
2. Train SpectrumEncoder with symmetric InfoNCE loss
3. Every N epochs: encode all training spectra → FAISS index → kNN retrieval → MRR@25
4. Save best checkpoint by val MRR@25

Hardware note: designed for RTX 3050 (4GB VRAM).
  - 10K molecules ≈ 57K spectra → ~3 min/epoch at batch 256
  - 50K molecules ≈ 285K spectra → ~15 min/epoch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import TRAIN_PATH, ARTIFACTS_DIR
from src.core.evaluation import summarize_ranks
from src.data.spectrum_dataset import SpectrumContrastiveDataset, spectrum_to_coarse_bins, COARSE_N_BINS
from src.models.spectrum_encoder import SpectrumEncoder
from src.models.losses import info_nce_loss_symmetric, embedding_accuracy


def create_dataloaders(
    subset_size: int,
    batch_size: int,
    val_frac: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    max_row_groups: int | None = None,
    augment_params: dict | None = None,
) -> tuple[DataLoader, DataLoader, SpectrumContrastiveDataset]:
    """Create train and val dataloaders."""
    dataset = SpectrumContrastiveDataset(
        train_path=TRAIN_PATH,
        subset_size=subset_size,
        augment=True,
        augment_params=augment_params,
        min_spectra_per_mol=2,
        seed=seed,
        max_row_groups=max_row_groups,
    )

    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    # Val dataset: disable augmentation for consistent evaluation
    # (We can't easily toggle augment on a Subset, so val still uses augmented
    #  pairs — but we evaluate retrieval on non-augmented embeddings separately)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader, dataset


def train_one_epoch(
    model: SpectrumEncoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 0.07,
) -> dict[str, float]:
    """Train for one epoch; return metrics dict."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for anchors, positives in loader:
        anchors = anchors.to(device)
        positives = positives.to(device)

        emb_a = model(anchors)
        emb_p = model(positives)

        loss = info_nce_loss_symmetric(emb_a, emb_p, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += embedding_accuracy(emb_a.detach(), emb_p.detach())
        n_batches += 1

    return {
        "train_loss": total_loss / max(n_batches, 1),
        "train_acc": total_acc / max(n_batches, 1),
    }


@torch.no_grad()
def validate(
    model: SpectrumEncoder,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 0.07,
) -> dict[str, float]:
    """Validate; return metrics dict."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for anchors, positives in loader:
        anchors = anchors.to(device)
        positives = positives.to(device)

        emb_a = model(anchors)
        emb_p = model(positives)

        loss = info_nce_loss_symmetric(emb_a, emb_p, temperature=temperature)
        total_loss += loss.item()
        total_acc += embedding_accuracy(emb_a, emb_p)
        n_batches += 1

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_acc": total_acc / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate_retrieval(
    model: SpectrumEncoder,
    dataset: SpectrumContrastiveDataset,
    device: torch.device,
    query_mols: list[str] | None = None,
    n_queries: int = 500,
    top_k: int = 25,
) -> dict[str, float]:
    """Embed all spectra → vectorized kNN retrieval → compute MRR@25.

    Evaluates same-molecule retrieval across the entire spectrum library.
    If query_mols is provided (e.g. from held-out validation set),
    evaluates out-of-distribution representation quality on unseen molecules.
    """
    model.eval()

    # Precomputed / cached feature vectors (N, 1483)
    features_np = dataset.get_all_features()
    all_mols = np.array([spec["mol"] for spec in dataset.all_spectra])
    N = len(all_mols)

    # Batch encode on device (stream CPU -> GPU in 512-chunks to conserve VRAM)
    batch_size = 512
    embeddings_list = []
    for i in range(0, N, batch_size):
        batch = torch.from_numpy(features_np[i:i + batch_size]).to(device)
        emb = model(batch)
        embeddings_list.append(emb.cpu())
    embeddings = torch.cat(embeddings_list, dim=0).numpy()  # (N, D)

    # Map mol -> indices
    mol_to_indices: dict[str, list[int]] = {}
    for i, m in enumerate(all_mols):
        if m not in mol_to_indices:
            mol_to_indices[m] = []
        mol_to_indices[m].append(i)

    # Filter query candidates to molecules with >= 2 spectra
    if query_mols is not None:
        eligible = [m for m in query_mols if len(mol_to_indices.get(m, [])) >= 2]
    else:
        eligible = [m for m, idxs in mol_to_indices.items() if len(idxs) >= 2]

    rng = np.random.default_rng(123)
    rng.shuffle(eligible)
    queries = eligible[:n_queries]

    if not queries:
        return {"mrr": 0.0, "hit@1": 0.0, "hit@5": 0.0, "hit@25": 0.0}

    q_idxs = np.array([mol_to_indices[m][0] for m in queries])
    Q = embeddings[q_idxs]  # (n_q, D)

    # Cosine similarity matrix: Q @ embeddings.T -> (n_q, N)
    all_sims = Q @ embeddings.T
    # Mask out self-match
    all_sims[np.arange(len(queries)), q_idxs] = -1e9

    # Vectorized Top-K
    k = min(top_k, N - 1)
    top_k_partition = np.argpartition(-all_sims, k, axis=1)[:, :k]
    row_idx = np.arange(len(queries))[:, None]
    top_k_sorted = top_k_partition[row_idx, np.argsort(-all_sims[row_idx, top_k_partition], axis=1)]

    ranks = []
    for i, mol in enumerate(queries):
        retrieved_mols = all_mols[top_k_sorted[i]]
        matches = np.where(retrieved_mols == mol)[0]
        ranks.append(int(matches[0] + 1) if len(matches) > 0 else 0)

    metrics = summarize_ranks(ranks, k=k)

    # Cross-CE similarity for condition robustness
    cross_ce_sims = []
    for mol in queries:
        idxs = mol_to_indices[mol]
        if len(idxs) >= 2:
            ce0 = dataset.all_spectra[idxs[0]]["ce"]
            ce1 = dataset.all_spectra[idxs[1]]["ce"]
            if abs(ce0 - ce1) > 1.0:
                sim = float(np.dot(embeddings[idxs[0]], embeddings[idxs[1]]))
                cross_ce_sims.append(sim)
    if cross_ce_sims:
        metrics["cross_ce_sim"] = float(np.mean(cross_ce_sims))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Contrastive spectrum encoder")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--subset", type=int, default=10000, help="Number of molecules to use")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden-channels", type=int, default=128, help="Conv channels")
    parser.add_argument("--n-blocks", type=int, default=4, help="Residual blocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--eval-every", type=int, default=5, help="Retrieval eval every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-row-groups", type=int, default=None, help="Limit row groups to read")
    parser.add_argument("--exp-name", type=str, default="exp2a", help="Experiment name")
    parser.add_argument("--eval-only", action="store_true", help="Only run retrieval evaluation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to load")
    args = parser.parse_args()

    # Load YAML config if provided
    if args.config:
        cfg_file = Path(args.config)
        if cfg_file.exists():
            import yaml
            with open(cfg_file) as f:
                cfg = yaml.safe_load(f)
            if "data" in cfg:
                args.subset = cfg["data"].get("subset_size", args.subset) or args.subset
            if "training" in cfg:
                args.epochs = cfg["training"].get("epochs", args.epochs)
                args.batch_size = cfg["training"].get("batch_size", args.batch_size)
                args.lr = float(cfg["training"].get("lr", args.lr))
                args.temperature = float(cfg["training"].get("temperature", args.temperature))
            if "model" in cfg:
                args.embed_dim = cfg["model"].get("embed_dim", args.embed_dim)
                args.hidden_channels = cfg["model"].get("hidden_channels", args.hidden_channels)
                args.n_blocks = cfg["model"].get("n_blocks", args.n_blocks)
                args.dropout = float(cfg["model"].get("dropout", args.dropout))
            if "evaluation" in cfg:
                args.eval_every = cfg["evaluation"].get("eval_every", args.eval_every)
            print(f"[stage2] loaded config from {cfg_file}")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[stage2] device={device}, subset={args.subset}, epochs={args.epochs}")
    print(f"[stage2] batch_size={args.batch_size}, lr={args.lr}, temp={args.temperature}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    exp_dir = ARTIFACTS_DIR / "stage02" / args.exp_name
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Data
    print("[stage2] loading data...")
    train_loader, val_loader, dataset = create_dataloaders(
        subset_size=args.subset,
        batch_size=args.batch_size,
        seed=args.seed,
        max_row_groups=args.max_row_groups,
    )
    # Extract held-out validation molecules for unbiased retrieval testing
    val_indices = getattr(val_loader.dataset, "indices", list(range(len(dataset))))
    val_mols = [dataset.molecules[i] for i in val_indices]

    # Model
    model = SpectrumEncoder(
        input_dim=dataset.feature_dim,
        embed_dim=args.embed_dim,
        hidden_channels=args.hidden_channels,
        n_blocks=args.n_blocks,
        dropout=args.dropout,
    ).to(device)
    print(f"[stage2] model: {model.num_parameters:,} params ({model.num_trainable:,} trainable)")

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        print(f"[stage2] loaded checkpoint: {args.checkpoint}")

    if args.eval_only:
        print("[stage2] running retrieval evaluation...")
        metrics = evaluate_retrieval(model, dataset, device, query_mols=val_mols)
        ce_str = f" cross_ce={metrics['cross_ce_sim']:.3f}" if "cross_ce_sim" in metrics else ""
        print(f"[stage2] retrieval: MRR@25={metrics['mrr']:.4f} hit@1={metrics['hit@1']:.3f} "
              f"hit@25={metrics.get('hit@25', 0):.3f}{ce_str}")
        return

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")
    best_mrr = 0.0
    history: list[dict] = []

    print(f"[stage2] training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.temperature)
        val_metrics = validate(model, val_loader, device, args.temperature)
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

        # Retrieval evaluation periodically
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            retrieval = evaluate_retrieval(model, dataset, device, query_mols=val_mols, n_queries=500)
            epoch_metrics["retrieval_mrr"] = retrieval["mrr"]
            epoch_metrics["retrieval_hit1"] = retrieval["hit@1"]
            if "cross_ce_sim" in retrieval:
                epoch_metrics["cross_ce_sim"] = retrieval["cross_ce_sim"]
            ce_str = f" cross_ce={retrieval['cross_ce_sim']:.3f}" if "cross_ce_sim" in retrieval else ""
            print(f"  epoch {epoch:3d} | loss={train_metrics['train_loss']:.4f} "
                  f"val_loss={val_metrics['val_loss']:.4f} "
                  f"acc={train_metrics['train_acc']:.3f} "
                  f"MRR@25={retrieval['mrr']:.4f} hit@1={retrieval['hit@1']:.3f}{ce_str} "
                  f"| {elapsed:.1f}s")

            if retrieval["mrr"] > best_mrr:
                best_mrr = retrieval["mrr"]
                torch.save(model.state_dict(), ckpt_dir / "best.pt")
                print(f"  [*] new best MRR@25={best_mrr:.4f} -> saved best.pt")
        else:
            print(f"  epoch {epoch:3d} | loss={train_metrics['train_loss']:.4f} "
                  f"val_loss={val_metrics['val_loss']:.4f} "
                  f"acc={train_metrics['train_acc']:.3f} "
                  f"| {elapsed:.1f}s")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:03d}.pt")

        # Track best val loss
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]

        history.append(epoch_metrics)

    # Save final results
    results = {
        "config": {
            "subset_size": args.subset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "embed_dim": args.embed_dim,
            "hidden_channels": args.hidden_channels,
            "n_blocks": args.n_blocks,
            "dropout": args.dropout,
            "seed": args.seed,
            "model_params": model.num_parameters,
        },
        "best_val_loss": best_val_loss,
        "best_retrieval_mrr": best_mrr,
        "history": history,
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save(model.state_dict(), ckpt_dir / "final.pt")
    print(f"\n[stage2] done. best MRR@25={best_mrr:.4f}")
    print(f"[stage2] artifacts saved to {exp_dir}")


if __name__ == "__main__":
    main()
