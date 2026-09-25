"""CASMI26 MS/MS molecule identification pipeline.

Package structure:
    src/
    ├── core/           Shared utilities (config, adducts, preprocessing, evaluation, split)
    ├── search/         Classical spectral retrieval (Stage 0–1)
    ├── models/         Neural network models (Stages 2–5)
    ├── data/           PyTorch data loading (Stages 2–5)
    └── train_pipeline  Stage 0/1 orchestrator

    scripts/            Entry points — one script per stage
    configs/            Experiment configs (YAML) — one dir per stage
    artifacts/          All outputs — checkpoints, results, submissions
"""
