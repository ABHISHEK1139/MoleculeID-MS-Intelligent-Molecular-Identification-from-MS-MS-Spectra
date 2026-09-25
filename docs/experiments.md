# Empirical Experiments, Ablations & Failure Mode Analysis

## 1. Experimental Overview

This document chronicles the empirical evolution of the **MoleculeID-MS** identification engine, comparing retrieval configurations, multi-channel representations, learned meta-rankers, and physical evidence fusion across both internal cross-validation and external generalization benchmarks.

---

## 2. Channel-by-Channel Ablation Study

Each component of the four-channel pipeline was ablated systematically across the clean benchmark queries to quantify its marginal contribution to Mean Reciprocal Rank (MRR) and Top-$k$ recall.

| Experiment | Channels Active | Scoring Formulation | MRR | Hit@1 | Hit@5 | Hit@25 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ablation 1** | Mass Retrieval Only | Precursor mass error ($-\Delta\text{ppm}$) | 0.1120 | 5.2% | 18.4% | 52.1% |
| **Ablation 2** | Mass + Direct Matching | Cosine similarity | 0.2840 | 21.0% | 41.2% | 68.5% |
| **Ablation 3** | Mass + Direct Matching | Spectral Entropy (normalized) | 0.3439 | 26.8% | 49.0% | 76.2% |
| **Ablation 4** | Mass + Direct + Analog | Dual-hypothesis mass-shift ($\pm 200$ Da) | 0.4170 | 32.4% | 58.1% | 84.6% |
| **Ablation 5** | Mass + Direct + Analog + FPNet | Fixed Continuous Linear Fusion | **0.4924** | **39.5%** | **68.2%** | **92.4%** |
| **Experiment 6** | Learned GBDT Meta-Ranker | LightGBM LambdaMART (In-Domain OOF) | *0.7517* | *64.2%* | *86.1%* | *96.8%* |

---

## 3. The GBDT Meta-Ranker: An In-Domain Breakthrough

In an effort to learn non-linear interactions between spectral channels, a **Gradient Boosted Decision Tree (LightGBM LambdaMART)** meta-ranker was trained using 18 engineered features:
- Precursor $\Delta\text{ppm}$, neutral mass difference
- Direct spectral entropy, cosine similarity, matched peak count
- Analog mass-shifted cosine, core scaffold similarity, modification mass $\Delta m$
- FPNet predicted bit dot-product, query-level z-score, candidate bit density
- Candidate library prior flags (COCONUT 2.0, ChEBI, PubChem)

### In-Domain Out-of-Fold (OOF) Results
Evaluated on 5-fold molecule-grouped cross-validation, the GBDT ranker demonstrated extraordinary performance improvements, particularly in reference-sparse cohorts:

| Stratification Cohort | Direct + Analog Baseline | Fixed Linear Fusion | Learned GBDT Ranker | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Benchmark** | 0.3439 | 0.4924 | **0.7517** | **+52.7%** |
| **Cohort C1 (Library Match)** | 0.6210 | 0.7140 | **0.8420** | +17.9% |
| **Cohort C2 (Database Known)** | 0.2450 | 0.4110 | **0.8110** | **+97.3%** |
| **Cohort C3 (Novel / De Novo)** | 0.1180 | 0.2850 | **0.7752** | **+172.0%** |

On paper, this represented a state-of-the-art result. However, rigorous scientific verification demanded testing whether this leap represented true generalization or domain overfitting.

---

## 4. The Critical Stress Test: External GNPS Generalization

To test generalization under genuine distribution shift, we deployed an external stress test: **Test B (50 natural product spectra from GNPS)**. These compounds satisfied strict boundary conditions:
- **Zero training overlap:** Never appeared in any training spectrum, reference library, or fine-tuning set.
- **Novel chemical space:** Distinct natural product scaffolds with complex polycyclic and glycosylated motifs.
- **Different instrument platforms:** Acquired on independent Q-TOF and Orbitrap platforms with distinct noise floors.

### The Domain-Shift Collapse

| Model Architecture | In-Domain OOF MRR | External GNPS MRR | External GNPS Hit@25 | Transfer Status |
| :--- | :--- | :--- | :--- | :--- |
| **Learned GBDT Meta-Ranker** | **0.7517** | **0.2301** | 62.0% | **Catastrophic Failure (-69.4%)** |
| **Fixed Continuous Physical Fusion** | 0.4924 | **0.6580** | **98.0%** | **Robust Generalization (+33.6%)** |

```
                       Generalization Gap on Unseen GNPS Molecules
    1.0 ┌─────────────────────────────────────────────────────────────┐
        │                                                             │
    0.8 │       0.7517 (In-Domain OOF)                                │
        │        ●                                                    │
    0.6 │        │                                  0.6580 (External) │
        │        │                                   ●                │
    0.4 │        │   Collapse                        │   Robust       │
        │        │   ▼                               │   Generalization
    0.2 │        └───► 0.2301 (External)  0.4924 ────┘                │
        │                                  (In-Domain)                │
    0.0 └─────────────────────────────────────────────────────────────┘
                Learned GBDT Meta-Ranker      Continuous Physical Fusion
```

---

## 5. Root-Cause Failure Analysis: Why Decision Trees Collapsed

Post-mortem diagnostic analysis identified three fundamental failure modes in tree-based meta-ranking for mass spectrometry:

### 1. Hard Axis-Aligned Splits on Uncalibrated Proxy Features
GBDTs partitioned candidate feature spaces with orthogonal decision boundaries (e.g., `n_candidates < 42`, `raw_direct_score > 0.31`). When external spectra exhibited slightly lower signal-to-noise ratios or wider precursor windows, candidates crossed these arbitrary thresholds and were penalized with severe negative leaf values.

### 2. Over-Reliance on Candidate Pool Density
In-domain folds exhibited characteristic candidate density distributions correlated with molecular mass. The GBDT memorized these priors rather than weighting chemical fragmentation evidence. In external GNPS data, where candidate density diverged, this prior inverted into a distractor.

### 3. Discontinuous Score Landscape
Unlike continuous physics-grounded functions where small changes in spectral similarity yield small changes in ranking score, decision trees create step-function discontinuities. A tiny spectral difference near a split threshold caused candidate scores to jump or drop precipitously.

---

## 6. The Production Solution: Continuous Physics-Grounded Fusion

To ensure invariance across instrument platforms and unseen chemical spaces, the learned tree ranker was discarded for production in favor of **Continuous Physical Evidence Fusion**:

$$\text{Score}(c) = \text{Score}_{\text{Mass}}(c) + \text{Prior}(c) + \mathbf{1}_{\{\text{Direct}(c) \ge 0.10\}} \cdot [2.0 \cdot \text{Direct}(c)] + 1.5 \cdot \text{Analog}(c) + 1.2 \cdot z(\text{FPNet}(c))$$

### Key Design Principles:
1. **Monotonicity:** An increase in spectral entropy or substructure match probability guarantees a non-negative delta in candidate score.
2. **Dynamic Range Invariance via Query-Level Z-Score:**
   $$z(\text{FPNet}(c)) = \frac{\mathbf{f}_c^T \mathbf{z}_{\text{pred}} - \mu_q}{\sigma_q + 10^{-9}}$$
   Standardizing neural logits across each candidate pool removes dependency on absolute logit magnitudes, making the score robust against variations in precursor charge or collision cell efficiency.
3. **Conservative Direct Matching Gate:** Requiring $\text{Direct}(c) \ge 0.10$ eliminates spurious noise hits from low-intensity fragment matches while rewarding genuine library matches with a large additive margin.

### Outcome
Continuous physical fusion achieved **0.6580 MRR and 98.0% Top-25 retrieval accuracy** on zero-overlap external GNPS molecules, proving its reliability as a production-grade molecular identification engine.
