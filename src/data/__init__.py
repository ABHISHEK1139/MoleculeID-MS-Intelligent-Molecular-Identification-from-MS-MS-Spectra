"""PyTorch data loading for CASMI26 (Stages 2–5).

Submodules (to be added as stages progress):
- spectrum_dataset: PyTorch Dataset for spectra
- pair_dataset: Stage 3 — (spectrum, molecule) pairs
- candidate_dataset: Stage 5 — (spectrum, candidate_list) batches
- augmentations: peak dropout, intensity jitter, m/z shift
- molecule_features: RDKit → PyG graph conversion
"""
