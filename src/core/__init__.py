"""Core shared utilities for the CASMI26 pipeline.

This subpackage contains modules used by ALL stages:
- config: paths, physical constants, binning grid
- adducts: neutral mass computation from adduct strings
- preprocessing: spectral filtering and VariantConfig
- evaluation: MRR@25 and ranking metrics
- data_loader: parquet I/O and streaming
- split: molecule-level train/val splits
"""

from src.core.config import (
    ARTIFACTS_DIR,
    BASELINE_DIR,
    DATA_DIR,
    ROOT_DIR,
    STAGE01_RESULTS_PATH,
    STAGE01_SUBMISSION_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
