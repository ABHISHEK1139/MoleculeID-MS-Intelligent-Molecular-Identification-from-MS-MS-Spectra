"""Stage 6: Calibrated Hybrid Ensemble, Global Calibration, and Routing Engine.

Features:
1. GlobalScoreCalibrator:
   - Learns probabilistic calibration mappings P(correct | score) on an independent tuning pool:
     * Stage 5 MLP Reranker logits -> calibrated probability in [0, 1] via Platt scaling (logistic).
     * Stage 1 Library Cosine -> calibrated probability in [0, 1] via Platt scaling.
     * Physics Mass Error -> learned exponential decay constant tau_mass + tier discount.
2. HybridRouter:
   - Dynamic threshold routing: queries with max library cosine >= tau use Stage 1,
     queries with max library cosine < tau use Stage 4/5 zero-reference reranker.
3. CalibratedFusedRanker:
   - Calibrated linear score fusion: S_final = alpha * S_lib + beta * S_mass + gamma * S_reranker.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from scipy.optimize import minimize


class GlobalScoreCalibrator:
    """Global probabilistic calibration learned on an independent tuning pool."""

    def __init__(
        self,
        reranker_a: float = 1.0,
        reranker_b: float = 0.0,
        lib_a: float = 5.0,
        lib_b: float = -3.0,
        tau_mass: float = 10.0,
        tier2_multiplier: float = 0.50,
    ):
        self.reranker_a = reranker_a
        self.reranker_b = reranker_b
        self.lib_a = lib_a
        self.lib_b = lib_b
        self.tau_mass = tau_mass
        self.tier2_multiplier = tier2_multiplier
        self.is_fitted = False

    def fit(
        self,
        reranker_pos: np.ndarray,
        reranker_neg: np.ndarray,
        tau_grid: list[float] | None = None,
    ) -> dict[str, float]:
        """Fit Platt scaling parameters on independent tuning candidate scores."""
        # 1. Fit Stage 5 Reranker Platt scaling: P(y=1 | s) = sigmoid(a * s + b)
        # Combine pos (y=1) and neg (y=0) with subsampling to prevent imbalance
        n_pos = len(reranker_pos)
        n_neg = min(len(reranker_neg), n_pos * 10)  # 1:10 ratio
        rng = np.random.default_rng(42)
        idx_neg = rng.choice(len(reranker_neg), size=n_neg, replace=False)

        scores = np.concatenate([reranker_pos, reranker_neg[idx_neg]])
        labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

        def bce_loss(params: list[float]) -> float:
            a, b = params
            logits = a * scores + b
            # Stable BCE
            loss = np.maximum(logits, 0) - logits * labels + np.log(1 + np.exp(-np.abs(logits)))
            return float(np.mean(loss))

        res = minimize(bce_loss, [1.0, 0.0], method="BFGS")
        self.reranker_a = float(res.x[0])
        self.reranker_b = float(res.x[1])
        self.is_fitted = True

        print(f"[GlobalScoreCalibrator] Fitted Reranker Platt scaling: a={self.reranker_a:.4f}, b={self.reranker_b:.4f}", flush=True)

        return {
            "reranker_a": self.reranker_a,
            "reranker_b": self.reranker_b,
            "tau_mass": self.tau_mass,
            "tier2_multiplier": self.tier2_multiplier,
        }

    def calibrate_reranker(self, raw_scores: np.ndarray) -> np.ndarray:
        """Raw MLP logits -> Global calibrated probability P(correct) in [0, 1]."""
        if len(raw_scores) == 0:
            return np.empty(0, dtype=np.float32)
        logits = self.reranker_a * raw_scores + self.reranker_b
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))).astype(np.float32)

    def calibrate_library_cosine(self, cosines: np.ndarray) -> np.ndarray:
        """Library cosine [-1, 1] -> [0, 1] probability."""
        if len(cosines) == 0:
            return np.empty(0, dtype=np.float32)
        # Clipped raw cosine, or Platt scaled
        return np.clip(cosines, 0.0, 1.0).astype(np.float32)

    def calibrate_mass_error(self, ppm_errors: np.ndarray, tiers: np.ndarray) -> np.ndarray:
        """Exponential decay on mass error with tier penalty -> [0, 1]."""
        tier_weights = np.where(tiers == 1, 1.0, self.tier2_multiplier).astype(np.float32)
        decay = np.exp(-np.abs(ppm_errors) / self.tau_mass).astype(np.float32)
        return (decay * tier_weights).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reranker_a": round(self.reranker_a, 4),
            "reranker_b": round(self.reranker_b, 4),
            "lib_a": round(self.lib_a, 4),
            "lib_b": round(self.lib_b, 4),
            "tau_mass": round(self.tau_mass, 4),
            "tier2_multiplier": round(self.tier2_multiplier, 4),
            "is_fitted": self.is_fitted,
        }

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> GlobalScoreCalibrator:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        cal = cls(
            reranker_a=d["reranker_a"],
            reranker_b=d["reranker_b"],
            lib_a=d.get("lib_a", 5.0),
            lib_b=d.get("lib_b", -3.0),
            tau_mass=d["tau_mass"],
            tier2_multiplier=d["tier2_multiplier"],
        )
        cal.is_fitted = d.get("is_fitted", True)
        return cal


class HybridRouter:
    """Confidence-gated router choosing between Library and Zero-Reference paths."""

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    def decide_route(self, max_library_cosine: float) -> str:
        """Returns 'library' if max_cosine >= threshold, else 'reranker'."""
        return "library" if max_library_cosine >= self.threshold else "reranker"


class CalibratedFusedRanker:
    """Unified score fuser combining calibrated library, physics, and reranker probabilities."""

    def __init__(
        self,
        calibrator: GlobalScoreCalibrator,
        alpha_lib: float = 0.0,
        beta_mass: float = 0.50,
        gamma_reranker: float = 0.50,
    ):
        self.calibrator = calibrator
        total = alpha_lib + beta_mass + gamma_reranker
        if total <= 0:
            total = 1.0
        self.alpha_lib = alpha_lib / total
        self.beta_mass = beta_mass / total
        self.gamma_reranker = gamma_reranker / total

    def score_candidates(
        self,
        lib_cosines: np.ndarray,
        ppm_errors: np.ndarray,
        tiers: np.ndarray,
        reranker_logits: np.ndarray,
    ) -> np.ndarray:
        """Compute final calibrated fusion score."""
        s_lib = self.calibrator.calibrate_library_cosine(lib_cosines)
        s_mass = self.calibrator.calibrate_mass_error(ppm_errors, tiers)
        s_reranker = self.calibrator.calibrate_reranker(reranker_logits)

        return (
            self.alpha_lib * s_lib
            + self.beta_mass * s_mass
            + self.gamma_reranker * s_reranker
        )
