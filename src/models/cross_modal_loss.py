"""Cross-modal contrastive losses for Spectrum ↔ Molecule representation learning.

Treats a batch of (spectrum, molecule) pairs as positive matches on the diagonal,
or multi-positive matches when multiple spectra share the same molecule ID.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cross_modal_infonce_loss(
    spec_emb: torch.Tensor,
    mol_emb: torch.Tensor,
    temperature: float = 0.07,
    mol_ids: list[str] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Symmetric cross-modal InfoNCE loss with multi-positive support.

    Args:
        spec_emb: (B, D) L2-normalized spectrum embeddings.
        mol_emb: (B, D) L2-normalized molecule graph embeddings.
        temperature: Softmax temperature parameter.
        mol_ids: Optional list or tensor of length B with molecule identifiers.
                 When provided, entries sharing the same molecule ID are treated
                 as true positive matches rather than false-negative distractors.

    Returns:
        Scalar loss tensor.
    """
    logits = torch.mm(spec_emb, mol_emb.t()) / temperature

    if mol_ids is None:
        labels = torch.arange(spec_emb.size(0), device=spec_emb.device)
        loss_s2m = F.cross_entropy(logits, labels)
        loss_m2s = F.cross_entropy(logits.t(), labels)
        return (loss_s2m + loss_m2s) / 2.0

    # Multi-positive mask: (B, B) where pos_mask[i, j] is True if same molecule
    if isinstance(mol_ids, list):
        arr = np.array(mol_ids)
        pos_mask = torch.from_numpy(arr[:, None] == arr[None, :]).to(spec_emb.device)
    else:
        pos_mask = (mol_ids.unsqueeze(1) == mol_ids.unsqueeze(0)).to(spec_emb.device)

    # Spectrum → Molecule direction
    log_prob_s2m = F.log_softmax(logits, dim=1)
    loss_s2m = -(log_prob_s2m * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp(min=1.0)

    # Molecule → Spectrum direction
    log_prob_m2s = F.log_softmax(logits.t(), dim=1)
    loss_m2s = -(log_prob_m2s * pos_mask.t().float()).sum(dim=1) / pos_mask.t().float().sum(dim=1).clamp(min=1.0)

    return (loss_s2m.mean() + loss_m2s.mean()) / 2.0


def cross_modal_accuracy(
    spec_emb: torch.Tensor,
    mol_emb: torch.Tensor,
    mol_ids: list[str] | torch.Tensor | None = None,
) -> tuple[float, float]:
    """Compute in-batch retrieval accuracy in both directions.

    Returns:
        (acc_s2m, acc_m2s): Accuracy of spectrum→molecule and molecule→spectrum.
    """
    sims = torch.mm(spec_emb, mol_emb.t())
    B = spec_emb.size(0)

    if mol_ids is None:
        labels = torch.arange(B, device=spec_emb.device)
        s2m_preds = sims.argmax(dim=1)
        acc_s2m = (s2m_preds == labels).float().mean().item()
        m2s_preds = sims.argmax(dim=0)
        acc_m2s = (m2s_preds == labels).float().mean().item()
        return acc_s2m, acc_m2s

    if isinstance(mol_ids, list):
        arr = np.array(mol_ids)
        pos_mask = torch.from_numpy(arr[:, None] == arr[None, :]).to(spec_emb.device)
    else:
        pos_mask = (mol_ids.unsqueeze(1) == mol_ids.unsqueeze(0)).to(spec_emb.device)

    s2m_preds = sims.argmax(dim=1)
    acc_s2m = pos_mask[torch.arange(B, device=spec_emb.device), s2m_preds].float().mean().item()

    m2s_preds = sims.argmax(dim=0)
    acc_m2s = pos_mask[m2s_preds, torch.arange(B, device=spec_emb.device)].float().mean().item()

    return acc_s2m, acc_m2s
