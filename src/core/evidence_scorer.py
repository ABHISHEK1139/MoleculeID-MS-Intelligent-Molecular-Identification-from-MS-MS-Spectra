"""Evidence Aggregation Scorer.

Computes normalized, additive evidence scores for candidate molecules using
experimental reference spectra (MoNA + GNPS + Baseline 839k).

Ablation sequence components:
1. Spectral Similarity: best external modified cosine in [0, 1.0].
2. Peak Evidence: min(1.0, matched_peaks / 6.0) in [0, 1.0].
3. CE Evidence: exp(-|CE_query - CE_ref| / 20.0) in (0, 1.0] (neutral default 0.60 if missing).
4. Multiplicity Evidence: log1p(n_supporting) / log(6) in [0, 1.0] with source corroboration bonus.

The resulting additive score:
    external_score = w1 * spectral_similarity + w2 * peak_evidence + w3 * CE_evidence + w4 * multiplicity_evidence
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EvidenceWeights:
    """Weights for the additive evidence aggregation components."""
    w_similarity: float = 1.0
    w_peaks: float = 0.0
    w_ce: float = 0.0
    w_multiplicity: float = 0.0
    boost_scale: float = 4.50
    use_quadratic: bool = True

    def normalized(self) -> EvidenceWeights:
        """Return normalized weights where w_similarity + w_peaks + w_ce + w_multiplicity = 1."""
        tot = self.w_similarity + self.w_peaks + self.w_ce + self.w_multiplicity
        if tot <= 0:
            return self
        return EvidenceWeights(
            w_similarity=self.w_similarity / tot,
            w_peaks=self.w_peaks / tot,
            w_ce=self.w_ce / tot,
            w_multiplicity=self.w_multiplicity / tot,
            boost_scale=self.boost_scale,
            use_quadratic=self.use_quadratic,
        )


class EvidenceScorer:
    """Computes additive evidence scores for candidate molecules."""

    def __init__(self, weights: EvidenceWeights | None = None):
        self.weights = weights or EvidenceWeights()

    @staticmethod
    def extract_candidate_features(
        best_cos: float,
        matched_peaks: int,
        ce_diff: float,
        n_supporting: int = 1,
        source_count: int = 1,
    ) -> dict[str, float]:
        """Extract normalized individual feature components in [0, 1.0]."""
        if best_cos < 0.10:
            return {
                "similarity": 0.0,
                "peaks": 0.0,
                "ce": 0.0,
                "multiplicity": 0.0,
            }

        # 1. Spectral Similarity in [0, 1.0]
        f_sim = min(1.0, max(0.0, float(best_cos)))

        # 2. Matched Peak Evidence in [0, 1.0] (saturates at 6 matched peaks)
        f_peaks = min(1.0, max(0.0, float(matched_peaks) / 6.0))

        # 3. Collision Energy Agreement in (0, 1.0]
        if np.isfinite(ce_diff):
            f_ce = float(np.exp(-abs(ce_diff) / 20.0))
        else:
            f_ce = 0.60  # Neutral prior when CE is unspecified

        # 4. Multiplicity & Corroboration in [0, 1.0]
        # log1p(1) = 0.693, log1p(5) = 1.791. Saturate at 5 supporting spectra
        f_multi = min(1.0, max(0.0, math.log1p(max(1, n_supporting)) / math.log1p(5)))
        if source_count >= 2:
            f_multi = min(1.0, f_multi + 0.15)

        return {
            "similarity": f_sim,
            "peaks": f_peaks,
            "ce": f_ce,
            "multiplicity": f_multi,
        }

    def compute_evidence_score(
        self,
        features: dict[str, float],
        weights: EvidenceWeights | None = None,
    ) -> float:
        """Compute the composite additive evidence score."""
        w = weights or self.weights
        if features["similarity"] < 0.10:
            return 0.0

        raw_score = (
            w.w_similarity * features["similarity"]
            + w.w_peaks * features["peaks"]
            + w.w_ce * features["ce"]
            + w.w_multiplicity * features["multiplicity"]
        )

        if w.use_quadratic:
            return float(w.boost_scale * (raw_score ** 2))
        return float(w.boost_scale * raw_score)
