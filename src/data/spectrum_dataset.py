"""PyTorch Dataset for contrastive spectrum learning.

Design decisions:
- Uses COARSE binning (1.0 Da, 1480 bins) for the neural encoder.
  The 0.01 Da grid (148K bins) is too sparse and large for dense neural input.
  We can increase resolution in later experiments (exp2c/2d).
- Positive pairs: two different spectra of the **same molecule**.
  With mean 5.7 spectra/molecule, most molecules have multiple spectra
  taken under different conditions (different CE, instruments, etc.).
- Negative pairs come from in-batch negatives via InfoNCE loss.
- Preprocessing: deisotope + noise filter + normalize (same as variant 1E).
- Metadata features: precursor_mz (normalized), ionization_mode (0/1),
  collision_energy (normalized), appended as extra channels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.core.config import TRAIN_PATH, MZ_BIN_MIN, MZ_BIN_MAX
from src.core.preprocessing import preprocess_variant, VariantConfig
from src.data.augmentations import apply_augmentations

# ── Coarse binning grid for neural encoder ─────────────────────────────────
COARSE_BIN_WIDTH = 1.0
COARSE_N_BINS = int(round((MZ_BIN_MAX - MZ_BIN_MIN) / COARSE_BIN_WIDTH))  # 1480

# Preprocessing config matching variant 1E (best non-modified-cosine variant)
_PREPROCESS_CFG = VariantConfig(
    key="neural", name="neural_preprocess",
    min_rel=0.01, remove_precursor=True, deisotope=True,
    max_peaks=100, mass_filter=False, modified_cosine=False,
)


def spectrum_to_coarse_bins(
    mz: np.ndarray,
    intensity: np.ndarray,
    n_bins: int = COARSE_N_BINS,
    bin_min: float = MZ_BIN_MIN,
    bin_width: float = COARSE_BIN_WIDTH,
) -> np.ndarray:
    """Bin a peak list into a fixed-length dense vector (1480 bins @ 1.0 Da)."""
    vec = np.zeros(n_bins, dtype=np.float32)
    if mz.size == 0:
        return vec
    indices = ((mz - bin_min) / bin_width).astype(np.int64)
    valid = (indices >= 0) & (indices < n_bins)
    # If multiple peaks fall in the same bin, keep the max intensity
    np.maximum.at(vec, indices[valid], intensity[valid].astype(np.float32))
    return vec


class SpectrumContrastiveDataset(Dataset):
    """Yields (anchor_spectrum, positive_spectrum) pairs for InfoNCE training.

    Each pair consists of two **different** spectra of the **same** molecule,
    independently augmented. The loss function handles negative sampling
    via in-batch negatives.

    Args:
        train_path: Path to train.parquet.
        subset_size: Max number of molecules to load (for curriculum learning).
        augment: Whether to apply spectral augmentations.
        augment_params: Dict with keys drop_prob, jitter_sigma, shift_ppm.
        min_spectra_per_mol: Only include molecules with >= N spectra.
        seed: Random seed for reproducibility.
        max_row_groups: Max row groups to read (None = all).
    """

    def __init__(
        self,
        train_path: str | Path = TRAIN_PATH,
        subset_size: int | None = 10_000,
        augment: bool = True,
        augment_params: dict[str, float] | None = None,
        min_spectra_per_mol: int = 2,
        seed: int = 42,
        max_row_groups: int | None = None,
    ):
        self.augment = augment
        self.augment_params = augment_params or {
            "drop_prob": 0.1,
            "jitter_sigma": 0.05,
            "shift_ppm": 10.0,
        }
        self.rng = np.random.default_rng(seed)

        # ── Load data ──────────────────────────────────────────────────
        train_path = Path(train_path)
        pf = pq.ParquetFile(train_path)
        n_rg = pf.metadata.num_row_groups
        if max_row_groups is not None:
            n_rg = min(n_rg, max_row_groups)

        columns = [
            "ms2_mzs", "ms2_normalized_intensities",
            "precursor_mz", "adduct", "ionization_mode",
            "collision_energy_ev", "inchikey",
        ]

        all_spectra: list[dict[str, Any]] = []
        mol_to_indices: dict[str, list[int]] = {}

        for rg_idx in range(n_rg):
            table = pf.read_row_group(rg_idx, columns=columns)
            df = table.to_pandas()

            mzs = df["ms2_mzs"].values
            intensities = df["ms2_normalized_intensities"].values
            precursor_mzs = df["precursor_mz"].values
            ces = df["collision_energy_ev"].values
            ion_modes = df["ionization_mode"].values
            inchikeys = df["inchikey"].values

            stop_early = False
            for mz_raw, int_raw, prec_raw, ce_raw, im_raw, ik in zip(
                mzs, intensities, precursor_mzs, ces, ion_modes, inchikeys
            ):
                mz = np.asarray(mz_raw, dtype=np.float32)
                intensity = np.asarray(int_raw, dtype=np.float32)
                if mz.size < 3:
                    continue

                # Preprocess (deisotope + noise filter)
                precursor_mz = float(prec_raw) if (prec_raw is not None and not np.isnan(prec_raw)) else None
                mz_proc, int_proc = preprocess_variant(mz, intensity, precursor_mz, _PREPROCESS_CFG)
                if mz_proc.size < 3:
                    continue

                mol = str(ik)
                idx = len(all_spectra)

                # Parse metadata
                if ce_raw is None or (isinstance(ce_raw, float) and np.isnan(ce_raw)):
                    ce_val = 0.0
                elif isinstance(ce_raw, (list, tuple, np.ndarray)):
                    ce_arr = np.asarray(ce_raw, dtype=np.float32)
                    ce_arr = ce_arr[np.isfinite(ce_arr)]
                    ce_val = float(np.mean(ce_arr)) if ce_arr.size > 0 else 0.0
                else:
                    try:
                        c_flt = float(ce_raw)
                        ce_val = c_flt if np.isfinite(c_flt) else 0.0
                    except (ValueError, TypeError):
                        ce_val = 0.0
                ion_mode = 1.0 if "pos" in str(im_raw).lower() else 0.0

                all_spectra.append({
                    "mz": mz_proc,
                    "intensity": int_proc,
                    "precursor_mz": precursor_mz or 0.0,
                    "ce": ce_val,
                    "ion_mode": ion_mode,
                    "mol": mol,
                })

                if mol not in mol_to_indices:
                    mol_to_indices[mol] = []
                mol_to_indices[mol].append(idx)

            # Check if we have enough molecules with >= min_spectra_per_mol
            n_valid = sum(1 for idxs in mol_to_indices.values() if len(idxs) >= min_spectra_per_mol)
            if subset_size is not None and n_valid >= subset_size:
                break

        # Filter to molecules with enough spectra for positive pairs
        valid_mols = [
            mol for mol, idxs in mol_to_indices.items()
            if len(idxs) >= min_spectra_per_mol
        ]
        if subset_size is not None:
            valid_mols = valid_mols[:subset_size]

        self.mol_to_indices = {mol: mol_to_indices[mol] for mol in valid_mols}
        self.all_spectra = all_spectra
        self.molecules = list(self.mol_to_indices.keys())
        self._cached_features: np.ndarray | None = None

        n_spectra = sum(len(v) for v in self.mol_to_indices.values())
        print(f"[SpectrumDataset] {len(self.molecules)} molecules, "
              f"{n_spectra} spectra, augment={augment}")

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (anchor, positive) as binned spectrum tensors."""
        mol = self.molecules[idx]
        indices = self.mol_to_indices[mol]

        # Pick two different spectra of the same molecule
        if len(indices) == 1:
            # Shouldn't happen due to min_spectra_per_mol, but handle gracefully
            i1 = i2 = indices[0]
        else:
            chosen = self.rng.choice(len(indices), size=2, replace=False)
            i1, i2 = indices[chosen[0]], indices[chosen[1]]

        anchor = self._make_tensor(i1)
        positive = self._make_tensor(i2)
        return anchor, positive

    def _make_tensor(self, spec_idx: int) -> torch.Tensor:
        """Convert one spectrum to an augmented binned tensor with metadata."""
        spec = self.all_spectra[spec_idx]
        mz = spec["mz"]
        intensity = spec["intensity"]

        if self.augment:
            mz, intensity = apply_augmentations(
                mz, intensity, rng=self.rng, **self.augment_params,
            )

        # Bin to coarse grid
        binned = spectrum_to_coarse_bins(mz, intensity)

        # Append metadata as extra features (3 values)
        meta = np.array([
            spec["precursor_mz"] / 1000.0,  # normalize to ~[0, 2]
            spec["ion_mode"],                 # 0 or 1
            spec["ce"] / 100.0,               # normalize to ~[0, 1]
        ], dtype=np.float32)

        # Concatenate: 1480 bins + 3 metadata = 1483 features
        features = np.concatenate([binned, meta])
        return torch.from_numpy(features)

    @property
    def feature_dim(self) -> int:
        """Total input dimension (bins + metadata)."""
        return COARSE_N_BINS + 3

    def get_all_features(self) -> np.ndarray:
        """Return (N, feature_dim) unaugmented binned features + metadata for all spectra."""
        if self._cached_features is not None:
            return self._cached_features

        features_list = []
        for spec in self.all_spectra:
            binned = spectrum_to_coarse_bins(spec["mz"], spec["intensity"])
            meta = np.array([
                spec["precursor_mz"] / 1000.0,
                spec["ion_mode"],
                spec["ce"] / 100.0,
            ], dtype=np.float32)
            features_list.append(np.concatenate([binned, meta]))

        self._cached_features = np.stack(features_list).astype(np.float32)
        return self._cached_features
