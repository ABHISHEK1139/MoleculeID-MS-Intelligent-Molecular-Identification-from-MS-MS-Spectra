# MoleculeID-MS: Intelligent Molecular Identification from Tandem Mass Spectra (LC-MS/MS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.9+-green.svg)](https://www.rdkit.org/)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Benchmark: CASMI 2026](https://img.shields.io/badge/Benchmark-CASMI%202026-blueviolet.svg)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)

An end-to-end, high-throughput molecular identification engine for untargeted liquid chromatography-tandem mass spectrometry (LC-MS/MS). **MoleculeID-MS** combines multi-window concentric precursor retrieval, spectral entropy matching, mass-shifted analog propagation, and deep neural fingerprint inference into a continuous physics-grounded evidence fusion framework.

---

## 30-Second Architecture Overview

```text
                     MS/MS Spectrum (Precursor m/z + Fragment Peaks)
                                          │
                                          ▼
                               Mass Candidate Retrieval
                   (Multi-window ±20/50/100 ppm + 13C shifts, 776K+ library)
                                          │
                   ┌──────────────────────┼──────────────────────┐
                   │                      │                      │
                   ▼                      ▼                      ▼
        Direct Spectral Matching   Analog Propagation     FPNet Neural Inference
       (Spectral Entropy / Cosine) (±200 Da GNPS Shifts)   (6,930-bit GEMM Bayes)
                   │                      │                      │
                   └──────────────────────┼──────────────────────┘
                                          │
                                          ▼
                              Continuous Evidence Fusion
                    Score = Mass + Direct + 1.5*Analog + 1.2*z(FPNet)
                                          │
                                          ▼
                             Top-25 Molecule Identification
```

---

## Key Highlights

- **Multi-Tier Retrieval Ceiling:** Queries a curated catalog of **776,000+ candidate structures** across concentric precursor windows ($\pm 20$, $\pm 50$, $\pm 100$ ppm) and carbon-13 isotope shifts ($\pm 1.003355$ Da), achieving **$\sim 100\%$ candidate recall** while pruning candidate space by over $350\times$.
- **InChIKey14 Skeletal Canonicalization:** Enforces strict connectivity-based splitting to eliminate spectrum-level and collision-energy target leakage between train, validation, and reference libraries.
- **Deep Neural Substructure Inference:** Predicts **6,930 structural fingerprint bits** directly from raw MS/MS spectra using a 6-layer Transformer (`FPNet`). Candidate scoring is formulated as a vectorized Bayes log-likelihood solved via a single General Matrix Multiplication (**GEMM**) in sub-millisecond latency.
- **Empirical Generalization on Zero-Overlap Chemistry:** Evaluated on **50 authentic natural products from GNPS with zero training overlap**, achieving **0.6580 MRR** and **98.0% Top-25 retrieval accuracy** with continuous physical evidence fusion.
- **Failure Analysis of Tree Meta-Rankers:** Diagnosed the domain-shift collapse of Gradient Boosted Decision Trees (GBDT LambdaMART), which achieved an apparent $0.7517$ MRR on in-domain cross-validation but plummeted to $0.2301$ on unseen external data, establishing why continuous physics-grounded fusion is essential for scientific transfer.

---

## System Architecture

The pipeline processes biological extracts containing molecules with varying degrees of prior database representation:

| Cohort | Class Label | Real-World Context | Primary Evidence Channel |
| :--- | :--- | :--- | :--- |
| **C1** | **Library Match** | Known clinical metabolites with reference spectra | **Channel 1:** Direct Spectral Entropy |
| **C2** | **Database Known** | Known secondary metabolites lacking experimental MS/MS | **Channel 2:** Mass-Shifted Analog Search |
| **C3** | **De Novo / Novel** | Novel natural products and unseen chemical scaffolds | **Channel 4:** FPNet Neural Fingerprint Inference |

### 1. Channel 1: Direct Spectral Matching
Computes normalized spectral entropy similarity between query peaks and reference acquisitions:
$$S_{\text{entropy}}(Q, R) = 1 - \frac{2 S(M) - S(Q) - S(R)}{\ln 4}, \quad m_i = \frac{q_i + r_i}{2}$$

### 2. Channel 2: Mass-Shifted Analog Search
Identifies conserved core scaffolds across reference molecules differing by mass offset $\Delta m \in [-200\text{ Da}, +200\text{ Da}]$:
$$\text{AnalogScore}(c) = \max_{a \in \mathcal{A}} \left[ \text{Sim}_{\text{mod}}(q, a)^{2.0} \cdot \text{Tanimoto}(\mathbf{fp}_c, \mathbf{fp}_a) \right]$$

### 3. Channel 4: Deep Neural Fingerprint Inference (FPNet)
A 6-layer Transformer mapping normalized $m/z$ peaks and collision energy embeddings to 6,930 substructure probabilities. Formulating candidate ranking as an independent Bernoulli Bayes log-likelihood simplifies to a linear dot product:
$$\mathbf{S}_{\text{neural}} = \mathbf{F}_{\text{cand}} \mathbf{z}_{\text{pred}}$$

### 4. Continuous Physical Evidence Fusion
Final candidate scoring combines physical mass accuracy, library priors, direct matches, analog propagation, and query-standardized neural logits:
$$\text{Score}(c) = \text{Score}_{\text{Mass}}(c) + \text{Prior}(c) + \mathbf{1}_{\{\text{Direct}(c) \ge 0.10\}} \cdot [2.0 \cdot \text{Direct}(c)] + 1.5 \cdot \text{Analog}(c) + 1.2 \cdot z(\text{FPNet}(c))$$

where $z(\text{FPNet}(c)) = \frac{\mathbf{f}_c^T \mathbf{z}_{\text{pred}} - \mu_q}{\sigma_q + 10^{-9}}$ makes candidate scores invariant to global logit shifts across instrument platforms.

*For complete mathematical derivations and module specifications, see [docs/architecture.md](docs/architecture.md).*

---

## Validation & Verification Protocol

Scientific machine learning in metabolomics is notoriously vulnerable to **spectrum-level leakage**: a single molecule analyzed across multiple collision energies (10, 20, 40 eV) or instruments can appear in both training and validation splits.

```
       Quercetin [M+H]+ @ 20 eV (Train) ──► Quercetin [M+H]+ @ 40 eV (Val)
       [Memorized fragment peaks]             [Artificially inflated score]
```

### Three-Part Verification Protocol

1. **InChIKey14 Skeletal Isolation:** All datasets and reference libraries are split strictly on the first 14 characters of the InChIKey. No candidate sharing skeletal connectivity with a validation query is allowed in reference libraries.
2. **Bemis-Murcko Scaffold Clustering:** Molecules are partitioned into 370 disjoint core scaffold frameworks to evaluate out-of-scaffold generalization.
3. **External GNPS Stress Test (Level 3):** 50 authentic natural product spectra acquired on independent instrumentation with zero overlap in any training or reference set.

*For full protocol details, see [docs/validation.md](docs/validation.md).*

---

## Experimental Results

### Channel-by-Channel Ablation

| Pipeline Configuration | Scoring Formulation | MRR | Hit@1 | Hit@5 | Hit@25 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mass Retrieval Only** | Precursor mass error ($-\Delta\text{ppm}$) | 0.1120 | 5.2% | 18.4% | 52.1% |
| **+ Direct Spectral Matching** | Normalized spectral entropy | 0.3439 | 26.8% | 49.0% | 76.2% |
| **+ Mass-Shifted Analog Search** | GNPS $\pm 200$ Da dual-hypothesis | 0.4170 | 32.4% | 58.1% | 84.6% |
| **+ FPNet Neural Embeddings** | **Continuous Physical Fusion** | **0.4924** | **39.5%** | **68.2%** | **92.4%** |
| *Learned GBDT Meta-Ranker* | *LightGBM LambdaMART (In-Domain OOF)* | *0.7517* | *64.2%* | *86.1%* | *96.8%* |

---

## Failure Analysis: Why GBDT Meta-Ranking Failed External Transfer

A major empirical insight of this project arose from investigating why an apparently high-performing learned ranker collapsed when tested on unseen chemistry.

### The Phenomenon
On 5-fold in-domain cross-validation, a Gradient Boosted Decision Tree (LightGBM LambdaMART) meta-ranker delivered dramatic improvements across all cohorts:
- In-Domain OOF MRR: **0.7517** (C1: 0.8420, C2: 0.8110, C3: 0.7752).

However, when deployed on the **External GNPS Generalization Benchmark (Test B: 50 zero-overlap molecules)**, the models diverged:

| Model Architecture | In-Domain OOF MRR | External GNPS MRR | External GNPS Hit@25 | Transfer Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Learned GBDT Meta-Ranker** | **0.7517** | **0.2301** | 62.0% | **Catastrophic Failure (-69.4%)** |
| **Continuous Physical Evidence Fusion** | 0.4924 | **0.6580** | **98.0%** | **Robust Generalization (+33.6%)** |

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

### Root Causes
1. **Orthogonal Axis-Aligned Splits:** Trees set brittle hard thresholds on uncalibrated proxy features (e.g., candidate pool size, unstandardized cosine). Slight differences in instrument noise shifted external queries across thresholds into heavily penalized leaves.
2. **Prior Memorization:** In-domain folds exhibited characteristic candidate density distributions that the tree memorized; on external GNPS data, this became a distractor.
3. **Step-Function Discontinuity:** Minor spectral perturbations caused massive non-physical score swings.

### The Solution
Replacing the learned tree with **Continuous Physical Evidence Fusion** restored monotonic, physics-grounded score scaling, delivering **0.6580 MRR and 98% Top-25 accuracy** on external natural products.

*For complete ablation logs, see [docs/experiments.md](docs/experiments.md).*

---

## Repository Structure

```text
MoleculeID-MS/
├── README.md                 # Project overview, architecture, benchmark results
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── pytest.ini                # PyTest configuration
├── docs/                     # Scientific documentation
│   ├── architecture.md       # Mathematical formulations of all 4 channels
│   ├── validation.md         # InChIKey14 isolation & 3-part verification protocol
│   └── experiments.md        # Ablations, GBDT failure analysis, GNPS benchmarks
├── src/                      # Production source code
│   ├── core/                 # Adduct rules, InChIKey14 handling, candidate retrieval
│   │   ├── adducts.py        # Neutral mass & adduct conversion tables
│   │   ├── candidate_retrieval.py # Multi-window concentric retrieval engine
│   │   ├── formula.py        # Molecular formula generation and 7-Golden Rules
│   │   ├── preprocessing.py  # Spectral peak cleaning and normalization
│   │   └── split.py          # InChIKey14 and Murcko scaffold partitioners
│   ├── models/               # Neural representations
│   │   ├── fpnet.py          # 6,930-bit Transformer fingerprint predictor
│   │   ├── spectrum_encoder.py # 1D-CNN ResNet spectrum encoder
│   │   └── reranker.py       # Hard-negative pairwise rankers
│   ├── search/               # Classical spectral search engines
│   │   └── spectral_search.py # Spectral entropy, cosine, and analog search
│   └── inference/            # Continuous evidence fusion engine
├── scripts/                  # Standalone execution and evaluation utilities
│   ├── run_benchmark.py      # Automated benchmark evaluation runner
│   └── run_gnps_stress_test.py # Level-3 external generalization benchmark
└── tests/                    # Unit and integration test suite (19 passing)
    ├── test_candidate_retrieval.py
    ├── test_molecule_identity_join.py
    ├── test_stage4.py
    ├── test_stage5.py
    └── test_fpnet_weights.py
```

---

## Installation & Environment Setup

### 1. Prerequisites
- Python 3.10 or higher
- NVIDIA CUDA 12.1+ compatible GPU (recommended for neural inference)
- $\ge 16$ GB RAM

### 2. Clone and Setup Environment
```bash
git clone https://github.com/ABHISHEK1139/MoleculeID-MS-Intelligent-Molecular-Identification-from-MS-MS-Spectra.git
cd MoleculeID-MS-Intelligent-Molecular-Identification-from-MS-MS-Spectra

# Create virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt
```

### 3. Verify Test Suite
Run the automated test suite covering candidate retrieval, InChIKey14 joins, physical pruning, and neural weight structures:
```bash
pytest tests/ -v
# Output: 19 passed in ~20s
```

---

## Reproduction & Usage

### Running Candidate Retrieval
```python
from src.core.candidate_retrieval import retrieve_candidates_multi_window
from src.core.adducts import precursor_to_neutral_mass

# Calculate neutral mass from precursor m/z
neutral_mass = precursor_to_neutral_mass(precursor_mz=303.0499, adduct="[M+H]+")

# Query pre-indexed candidate catalog across concentric mass windows
candidates = retrieve_candidates_multi_window(
    neutral_mass=neutral_mass,
    candidate_db="candidate_library.parquet",
    windows_ppm=[20.0, 50.0, 100.0],
    enable_c13_shifts=True,
)
print(f"Retrieved {len(candidates)} candidates.")
```

### Running Continuous Evidence Fusion
```python
import numpy as np
from src.core.evidence_scorer import score_candidates_continuous_fusion

# Score candidates combining mass, direct match, analog, and FPNet logits
ranked_candidates = score_candidates_continuous_fusion(
    candidates=candidates,
    direct_scores=direct_entropy_scores,
    analog_scores=analog_shifted_scores,
    fpnet_logits=predicted_fingerprint_logits,
    w_direct=2.0,
    w_analog=1.5,
    w_fpnet=1.2,
    direct_threshold=0.10,
)
# Top-25 ranked molecules
top25 = ranked_candidates[:25]
```

---

## Benchmark Context

This system was developed and validated in the context of the **CASMI 2026** (Critical Assessment of Small Molecule Identification) challenge, evaluating untargeted structural elucidation of natural products and biological metabolites from high-resolution mass spectrometry.

---

## Citation

If you build upon this architecture or research methodology, please cite:

```bibtex
@software{moleculeid_ms_2026,
  author = {Abhishek Kumar},
  title = {MoleculeID-MS: Intelligent Molecular Identification from Tandem Mass Spectra},
  year = {2026},
  url = {https://github.com/ABHISHEK1139/MoleculeID-MS-Intelligent-Molecular-Identification-from-MS-MS-Spectra},
  note = {Developed for CASMI 2026 LC-MS/MS Structural Identification}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
