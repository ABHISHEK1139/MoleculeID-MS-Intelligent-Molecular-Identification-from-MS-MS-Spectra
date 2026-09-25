"""Paths, physical constants, and binning grid for the CASMI26 pipeline.

This file should contain ONLY immutable facts:
- Filesystem paths
- Physical / chemistry constants
- The binning grid definition
- Stage 0/1 evaluation defaults

Neural network hyperparameters belong in YAML config files under configs/.
"""
from __future__ import annotations

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# The Kaggle files live in dataset/ in this workspace; data/ is kept as a fallback.
if (ROOT_DIR / "dataset").exists():
    DATA_DIR = ROOT_DIR / "dataset"
elif (ROOT_DIR / "data").exists():
    DATA_DIR = ROOT_DIR / "data"
else:
    DATA_DIR = ROOT_DIR / "dataset"

RAW_DATA_DIR = DATA_DIR
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH = DATA_DIR / "test.parquet"
SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"
STAGE01_RESULTS_PATH = BASELINE_DIR / "stage01_results.json"
STAGE01_SUBMISSION_PATH = BASELINE_DIR / "submission_stage1.csv"

EXTERNAL_DIR = ROOT_DIR / "external"

# ── Adduct mass shifts (legacy lookup; authoritative parsing in core.adducts) ─

ADDUCT_MASS_SHIFT = {
    "[M+H]+": 1.007276,
    "[M+Na]+": 22.989220,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M-H]-": -1.007276,
    "[M+Cl]-": 34.969402,
}

# ── Spectral preprocessing defaults ───────────────────────────────────────

DEFAULT_PPM_TOL = 20.0
DEFAULT_TOP_K = 25
DEFAULT_MIN_REL_INTENSITY = 0.01
DEFAULT_MAX_PEAKS = 100
DEFAULT_PRECURSOR_TOL_DA = 0.5

# ── Binned-spectrum search grid ───────────────────────────────────────────

BIN_WIDTH = 0.01
MZ_BIN_MIN = 20.0
MZ_BIN_MAX = 1500.0
N_BINS = int(round((MZ_BIN_MAX - MZ_BIN_MIN) / BIN_WIDTH))
COARSE_BIN_WIDTH = 1.0

# ── Plausible precursor window ────────────────────────────────────────────
# (train has a few corrupted rows up to 2.4e6)

PRECURSOR_MZ_MIN = 30.0
PRECURSOR_MZ_MAX = 2000.0

# ── Candidate-pool capacity per query inside the accumulator ──────────────

ACC_CAP = 2000
COARSE_TOPK = 5000

# ── Stage 0/1 evaluation defaults ────────────────────────────────────────

SPLIT_SEED = 42
N_VAL_MOLECULES = 2000

# ── Neural pipeline defaults (overridden by YAML configs per experiment) ──

EMBEDDING_DIM = 256
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
MAX_EPOCHS = 20
