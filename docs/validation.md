# Scientific Validation Protocol & Leakage Prevention

## 1. The Critical Leakage Problem in Metabolomics ML

In tandem mass spectrometry (LC-MS/MS), a single chemical compound is routinely analyzed across multiple experimental runs, varying collision energies (e.g., 10 eV, 20 eV, 40 eV, stepped NCE), and instrument platforms (Orbitrap, Q-TOF, FT-ICR). 

### Spectrum-Level vs. Molecule-Level Leakage
If cross-validation folds or train/test splits are partitioned by raw **Spectrum ID** or **Query ID**, different fragmentation spectra of the **identical chemical entity** land on both sides of the training barrier:

```
                         Catastrophic Leakage Split
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│          Training Fold          │     │         Validation Fold         │
│  Quercetin [M+H]+ @ 20 eV       │ ──► │  Quercetin [M+H]+ @ 40 eV       │
│  (Memorized fragment peaks)     │     │  (Artificially inflated score)  │
└─────────────────────────────────┘     └─────────────────────────────────┘
```

This structural leakage artificially inflates validation metrics: models memorize specific fragment peak patterns rather than learning transferable fragmentation chemistry.

### InChIKey14 Chemical Identity Canonicalization
To establish an uncompromised barrier, all compounds are canonicalized by the first 14 characters of their International Chemical Identifier Key (**InChIKey14**), which encodes skeletal connectivity and heavy-atom topology invariant to isotopic composition, stereochemistry, or ionization:

$$\text{InChIKey} = \underbrace{\text{XXXXXXXXXXXXXX}}_{14\text{ characters: Skeletal Connectivity}} - \underbrace{\text{YYYYYYYYYY}}_{10\text{ characters: Stereochemistry}} - \underbrace{\text{Z}}_{1\text{ character: Protonation}}$$

All splits and library searches strictly enforce InChIKey14 isolation: **no spectrum or candidate sharing an InChIKey14 with the test cohort is permitted in the training reference library**.

---

## 2. Multi-Cohort Evaluation Framework

To rigorously reflect real-world biological discovery, benchmarks are stratified into three distinct cohorts representing decreasing levels of reference database coverage:

| Cohort | Class Label | Real-World Context | Reference Library Overlap |
| :--- | :--- | :--- | :--- |
| **C1** | **Library Match** | Known clinical metabolites, recurring natural products | Exact InChIKey14 present in spectral library |
| **C2** | **Database Known** | Known secondary metabolites lacking experimental MS/MS | InChIKey14 absent from library; related analogs present |
| **C3** | **De Novo / Novel** | Novel natural products, newly synthesized bio-actives | Scaffold completely absent from reference libraries |

---

## 3. Bemis-Murcko Scaffold Clustering

While InChIKey14 isolation prevents exact chemical leakage, compounds within the same chemical class (e.g., flavonoids, steroids, macrolides) share conserved core ring architectures and side-chain cleavage pathways.

To test structural extrapolation, we apply **Bemis-Murcko Scaffold Clustering**:
1. Remove all side-chain substituents from molecular graphs, retaining ring systems and linkers.
2. Group compounds sharing identical Murcko frameworks into disjoint clusters (370 clusters identified across benchmark data).
3. Partition cross-validation folds by Murcko Cluster ID to ensure the evaluation fold contains chemical topologies never observed during training.

```
       Quercetin (Flavonol)                     Kaempferol (Flavonol)
           OH      OH                                 OH
            \     /                                    \
         O   \___/                                  O   \___
       // \  /   \                                // \  /   \
  HO──│    ││     │──OH                      HO──│    ││     │──OH
       \__// \___/                                \__// \___/
        │                                          │
        OH                                         OH
                     │                                  │
                     └─────────────────┬────────────────┘
                                       ▼
                         Shared Bemis-Murcko Core
                                    O
                                  // \  /───
                                 │    ││    │
                                  \__// \───┘
                                   │
```

---

## 4. The Three-Part Verification Protocol

To ensure models generalize beyond local benchmarks, all architectures must undergo three levels of empirical verification:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        VERIFICATION LEVEL 1                            │
│  InChIKey14-Isolated Out-of-Fold (OOF) 5-Fold Cross-Validation         │
│  - Prevents spectrum-level and collision-energy leakage                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Passed
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        VERIFICATION LEVEL 2                            │
│  Bemis-Murcko Scaffold-Disjoint Holdout Validation                     │
│  - Evaluates generalization to unseen core chemical frameworks         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Passed
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        VERIFICATION LEVEL 3                            │
│  External GNPS Stress Test (Zero Reference Overlap)                    │
│  - 50 fully unseen natural products from external GNPS repositories   │
│  - Complete zero overlap with any training, fine-tuning, or ref data   │
└────────────────────────────────────────────────────────────────────────┘
```

### Protocol Details:
1. **Level 1 (InChIKey14 OOF):** Evaluates overall candidate retrieval and direct matching fidelity across 5 stratified folds.
2. **Level 2 (Murcko Scaffold Holdout):** Evaluates analog propagation and neural fingerprint inference when the entire core scaffold has never been seen in training.
3. **Level 3 (External GNPS Stress Test):** 50 authentic natural products curated from the Global Natural Products Social Molecular Networking (GNPS) repository with zero training overlap, verifying cross-instrument transfer (e.g., Q-Exactive Orbitrap vs. Agilent Q-TOF).

---

## 5. Quantitative Evaluation Metrics

All evaluations report standard ranking and retrieval metrics calculated strictly at the molecule level:

### Mean Reciprocal Rank (MRR)
The primary evaluation metric for molecular identification, capturing how close the true structure is ranked to the top:
$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$
where $\text{rank}_q$ is the 1-based index of the correct InChIKey14 in the ranked candidate list. If the correct candidate is not retrieved, $\frac{1}{\text{rank}_q} = 0$.

### Top-$k$ Hit Rate ($\text{Hit}@k$)
The fraction of test queries where the true structure appears within the top-$k$ ranked candidates:
$$\text{Hit}@k = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \mathbb{I}(\text{rank}_q \le k), \quad k \in \{1, 3, 5, 10, 25\}$$
Target criteria for production deployment: $\text{Hit}@25 \ge 95\%$ and $\text{MRR} \ge 0.60$.
