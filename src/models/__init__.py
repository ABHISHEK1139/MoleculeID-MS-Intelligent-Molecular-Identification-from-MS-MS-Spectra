"""Neural network models for CASMI26 (Stages 2–5).

Submodules (to be added as stages progress):
- spectrum_encoder: Stage 2 — contrastive spectrum embedding
- molecule_encoder: Stage 3 — GNN on molecular graphs
- cross_modal: Stage 3 — spectrum ↔ molecule matching
- reranker: Stage 5 — candidate scoring model
- losses: InfoNCE, triplet, listwise ranking losses
"""
