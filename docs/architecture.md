# System Architecture: Multi-Channel Molecular Identification Pipeline

## Overview

**MoleculeID-MS** is a high-throughput, leakage-free identification engine designed for high-resolution tandem mass spectrometry (LC-MS/MS). The architecture addresses the reality that natural biological extracts contain molecules with varying degrees of prior representation:
- **Class 1 (Library Match):** Exact reference spectra exist in public libraries.
- **Class 2 (Database Known):** Structures exist in molecular catalogs (COCONUT 2.0, ChEBI, PubChem) but lack reference MS/MS spectra.
- **Class 3 (Novel / De Novo):** The chemical entity is completely novel and absent from reference databases.

```
                      Query MS/MS Spectrum (Precursor m/z + Peaks)
                                            │
                                            ▼
                    ┌────────────────────────────────────────────────┐
                    │ Candidate Retrieval: Multi-Window Union Scheme │
                    │   (±20, ±50, ±100 ppm + ±13C Isotope Shifts)   │
                    └───────────────────────┬────────────────────────┘
                                            │ Candidate Pool C
                   ┌────────────────────────┼────────────────────────┐
                   │                        │                        │
                   ▼                        ▼                        ▼
       ┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
       │ Channel 1: Direct    │ │ Channel 2: Analog    │ │ Channel 4: FPNet     │
       │ Spectral Matching    │ │ Mass-Shift Search    │ │ Neural Fingerprint   │
       │ (Entropy / Cosine)   │ │ (±200 Da GNPS Shift) │ │ (6,930-bit GEMM)     │
       └──────────┬───────────┘ └──────────┬───────────┘ └──────────┬───────────┘
                  │                        │                        │
                  └────────────────────────┼────────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ Multi-Spectrum Evidence Consolidation   │
                      │ (Intensity-Weighted Neural Logit Pool)  │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │ Continuous Physical Evidence Fusion     │
                      │ Score = Mass + Direct + 1.5*Analog      │
                      │         + 1.2*z(FPNet)                  │
                      └────────────────────┬────────────────────┘
                                           │
                                           ▼
                              Ranked Top-25 Candidates
```

---

## 1. Candidate Retrieval: Multi-Tier Concentric Union Engine

Candidate generation is an absolute ceiling on retrieval-based identification: if the true structure is missed during retrieval, downstream ranking models cannot recover it.

### Mathematical Formulation
Given measured precursor mass $m/z_{\text{precursor}}$ and adduct ion $A$, the neutral molecular mass $M_{\text{neutral}}$ is calculated via:
$$M_{\text{neutral}} = \frac{m/z_{\text{precursor}} \cdot |z| - m_{\text{adduct}}}{1}$$

To accommodate instrument drift, biological matrix shifts, and data-dependent acquisition of $[M+1]$ isotopic peaks, candidate retrieval queries the pre-indexed candidate catalog across concentric mass windows and carbon-13 isotope shifts:
$$\mathcal{C} = \bigcup_{w \in \{20, 50, 100\}} \text{Window}(M_{\text{neutral}}, \pm w\text{ ppm}) \cup \text{Window}(M_{\text{neutral}} \pm 1.003355\text{ Da}, \pm 50\text{ ppm})$$

This formulation expands precursor ground-truth recall to **$>99.9\%$** while maintaining candidate pool sizes below 1,000 structures per query, preventing the decoy dilution observed in unconstrained PubChem retrieval.

---

## 2. Channel 1: Direct Spectral Similarity Matching

Direct matching measures the alignment between query peaks and reference spectra sharing identical molecular connectivity (`InChIKey14`).

### Spectral Entropy Metric
Traditional dot-product cosine similarity overweights dominant base peaks. Spectral entropy weights peaks by information content. For normalized fragment intensities $p_i$ where $\sum p_i = 1$:
$$S(P) = -\sum_{i=1}^K p_i \ln p_i$$

For query spectrum $Q$ and reference spectrum $R$, the spectral entropy similarity is computed on the merged spectrum $M$:
$$S_{\text{entropy}}(Q, R) = 1 - \frac{2 S(M) - S(Q) - S(R)}{\ln 4}$$
where $m_i = \frac{q_i + r_i}{2}$.

A candidate's direct score is the maximum spectral entropy across reference acquisitions matching its connectivity:
$$\text{Direct}(c) = \max_{r \in \text{Ref}(c)} S_{\text{entropy}}(Q, r)$$

---

## 3. Channel 2: Mass-Shifted Analog Propagation

For Class 2 compounds where no reference spectrum of the exact molecule exists, Channel 2 searches reference libraries for related structural analogs differing by mass offset:
$$\Delta m = M_{\text{query}} - M_{\text{ref}} \in [-200\text{ Da}, +200\text{ Da}]$$

Peaks are aligned under two simultaneous hypotheses:
1. **Unmodified fragment:** $m/z_q \approx m/z_r$ (conserved core scaffold)
2. **Shifted fragment:** $m/z_q \approx m/z_r + \Delta m$ (fragment containing modification)

Structural priors are propagated through the candidate pool via molecular fingerprint kernels:
$$\text{AnalogScore}(c) = \max_{a \in \mathcal{A}} \left[ \text{Sim}_{\text{mod}}(q, a)^{2.0} \cdot \text{Tanimoto}(\mathbf{fp}_c, \mathbf{fp}_a) \right]$$

---

## 4. Channel 4: Deep Neural Fingerprint Inference (FPNet)

Channel 4 employs a 6-layer Transformer neural network to predict the presence of 6,930 molecular substructures directly from raw MS/MS spectra.

### Vectorized Bayes Log-Likelihood Scoring
Rather than calculating iterative Tanimoto similarities across thousands of candidates, candidate scoring is formulated as a Bayes log-likelihood under an independent Bernoulli assumption for each predicted bit:
$$\ln P(\mathbf{f}_c \mid \mathbf{z}) = \sum_{i=1}^{6930} \left[ f_{c,i} \ln \sigma(z_i) + (1 - f_{c,i}) \ln (1 - \sigma(z_i)) \right]$$

Using the algebraic identity $\ln \sigma(z_i) - \ln(1 - \sigma(z_i)) = z_i$, relative candidate ranking simplifies to the linear dot product:
$$\text{Score}_{\text{neural}}(c) = \mathbf{f}_c^T \mathbf{z}_{\text{pred}} = \sum_{i=1}^{6930} f_{c,i} z_i$$

This allows entire candidate pools to be scored simultaneously via a single General Matrix Multiplication (GEMM) on GPU accelerators in sub-millisecond latency:
$$\mathbf{S}_{\text{neural}} = \mathbf{F}_{\text{cand}} \mathbf{z}_{\text{pred}}$$

---

## 5. Multi-Spectrum Evidence Consolidation

Because compounds are evaluated per chemical entity (`molecule_id`), queries with multiple spectra acquired across diverse collision energies are consolidated at the feature extraction layer:
1. **Neural Logits:** Combined via base-peak intensity-weighted pooling:
   $$\mathbf{z}_{\text{agg}} = \sum_{s=1}^S w_s \mathbf{z}_s, \quad w_s = \frac{\ln(1 + I_{\text{base}, s})}{\sum_j \ln(1 + I_{\text{base}, j})}$$
2. **Direct & Analog Scores:** Consolidated via max-pooling across collision energies to capture optimal fragmentation conditions:
   $$\text{Score}_{\text{direct}}(c) = \max_{s \in \{1,\dots,S\}} \text{Score}_{\text{direct}}(c \mid s)$$

---

## 6. Continuous Physical Evidence Fusion

To prevent the catastrophic domain-shift collapse observed in tree-based meta-rankers (GBDTs), production inference utilizes continuous, monotonic linear fusion:
$$\text{Score}(c) = \text{Score}_{\text{Mass}}(c) + \text{Prior}(c) + \mathbf{1}_{\{\text{Direct}(c) \ge 0.10\}} \cdot [2.0 \cdot \text{Direct}(c)] + 1.5 \cdot \text{Analog}(c) + 1.2 \cdot z(\text{FPNet}(c))$$

where:
- $\text{Score}_{\text{Mass}}(c) = -\frac{|\Delta\text{ppm}(c)|}{100}$
- $\text{Prior}(c) = 0.05 \cdot \mathbb{I}_{\text{TRAIN}}(c) + 0.02 \cdot \mathbb{I}_{\text{COCONUT}}(c)$
- $z(\text{FPNet}(c)) = \frac{\mathbf{f}_c^T \mathbf{z}_{\text{pred}} - \mu_q}{\sigma_q + 10^{-9}}$ standardizes neural scores per query, making ranking invariant to global logit distribution shifts across instrument platforms.
