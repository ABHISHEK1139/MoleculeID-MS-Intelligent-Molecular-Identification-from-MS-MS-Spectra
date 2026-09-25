"""Consolidated evaluation metrics for CASMI26.

Merges the useful parts of the old ranking.py and evaluation.py into one
authoritative module. This is the single source of truth for MRR@25 and
related ranking metrics.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two flat vectors."""
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0 or y_norm == 0:
        return 0.0
    return float(np.dot(x, y) / (x_norm * y_norm))


def reciprocal_rank(rank: int, k: int = 25) -> float:
    """Reciprocal rank for a single query (1-based rank)."""
    return 1.0 / rank if 1 <= rank <= k else 0.0


def mrr_at_k(ranks: Sequence[int], k: int = 25) -> float:
    """Mean reciprocal rank @ k over a list of 1-based ranks."""
    rr = []
    for r in ranks:
        if 1 <= r <= k:
            rr.append(1.0 / r)
        else:
            rr.append(0.0)
    return float(np.mean(rr)) if rr else 0.0


def summarize_ranks(ranks: Sequence[int], k: int = 25) -> dict[str, float]:
    """Aggregate ranks (1-based; <=0 or missing means not retrieved).

    Returns a dict with: n, mrr, hit@1, hit@5, hit@k.
    """
    arr = np.asarray(ranks, dtype=np.int64)
    valid = arr > 0
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mrr": 0.0, "hit@1": 0.0, "hit@5": 0.0, f"hit@{k}": 0.0}
    rr = np.zeros(n, dtype=np.float64)
    rr[valid] = 1.0 / arr[valid]
    rr[arr > k] = 0.0
    return {
        "n": n,
        "mrr": float(rr.mean()),
        "hit@1": float((arr == 1).mean()),
        "hit@5": float((valid & (arr <= 5)).mean()),
        f"hit@{k}": float((valid & (arr <= k)).mean()),
    }


def rank_candidates(query_spectrum: Sequence[float], ref_spectra: Sequence[Sequence[float]]) -> list[tuple[int, float]]:
    """Simple baseline ranker: compute cosine similarity to every reference spectrum."""
    scores = []
    for idx, candidate in enumerate(ref_spectra):
        scores.append((idx, cosine_similarity(query_spectrum, candidate)))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores
