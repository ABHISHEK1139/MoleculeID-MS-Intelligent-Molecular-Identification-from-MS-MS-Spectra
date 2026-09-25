"""Contrastive losses for spectrum representation learning.

InfoNCE (Noise Contrastive Estimation):
  Given a batch of (anchor, positive) pairs, InfoNCE treats all other
  positives in the batch as negatives. With batch size B, each anchor
  has 1 positive and (B-1) negatives → effective negative ratio = B-1.

  This is why batch size matters: B=256 gives 255 negatives per query,
  which is enough for good contrastive learning.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE loss with in-batch negatives.

    Args:
        anchors: (B, D) L2-normalized anchor embeddings.
        positives: (B, D) L2-normalized positive embeddings.
        temperature: Softmax temperature. Lower = sharper distribution.
            0.07 is a good default from SimCLR/MoCo papers.

    Returns:
        Scalar loss (mean over batch).
    """
    # Similarity matrix: (B, B). sim[i,j] = anchor_i · positive_j
    logits = torch.mm(anchors, positives.t()) / temperature  # (B, B)

    # Labels: diagonal entries are the correct pairs
    labels = torch.arange(anchors.size(0), device=anchors.device)

    # Cross-entropy: each row should have max at the diagonal
    loss = F.cross_entropy(logits, labels)
    return loss


def info_nce_loss_symmetric(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE: average of anchor→positive and positive→anchor.

    More stable training than one-directional InfoNCE.
    """
    loss_ap = info_nce_loss(anchors, positives, temperature)
    loss_pa = info_nce_loss(positives, anchors, temperature)
    return (loss_ap + loss_pa) / 2.0


def embedding_accuracy(
    anchors: torch.Tensor,
    positives: torch.Tensor,
) -> float:
    """Fraction of anchors whose nearest neighbor is the correct positive.

    Useful as a training diagnostic (should increase from ~1/B to ~1.0).
    """
    sim = torch.mm(anchors, positives.t())  # (B, B)
    preds = sim.argmax(dim=1)
    labels = torch.arange(anchors.size(0), device=anchors.device)
    return (preds == labels).float().mean().item()
