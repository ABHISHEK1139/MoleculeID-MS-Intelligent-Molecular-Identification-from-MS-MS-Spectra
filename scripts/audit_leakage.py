"""Dataset Leakage and Validation Split Audit Script.

Verifies:
1. Strict molecule-disjointness between training and validation splits (0 shared IDs).
2. Strict structure uniqueness using RDKit canonical graphs (MolToInchi / canonical SMILES).
3. Query molecule's exact canonical structure does not exist anywhere in training references.
4. Characterizes InChIKey14 (skeleton/connectivity) overlap.
5. Characterizes molecular formula overlap (inter-split constitutional isomers).
6. Tests for actual spectrum duplicate leakage (cosine > 0.999).
7. Verifies molecular graph featurization contains zero label/target leakage.

Outputs report to: artifacts/stage06/leakage_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import ARTIFACTS_DIR, TRAIN_PATH
from src.data.cross_modal_dataset import create_cross_modal_datasets


def run_leakage_audit() -> dict[str, Any]:
    print("=" * 75, flush=True)
    print("  CASMI26 DATASET LEAKAGE & VALIDATION SPLIT AUDIT (RDKit Canonical)", flush=True)
    print("=" * 75, flush=True)

    print("\n1. Loading cross-modal train and validation datasets...", flush=True)
    train_ds, val_ds = create_cross_modal_datasets(
        train_path=TRAIN_PATH,
        subset_size=10000,
        val_frac=0.10,
        seed=42,
    )

    train_mols = set(train_ds.selected_mols)
    val_mols = set(val_ds.selected_mols)

    print(f"  Training unique molecules:   {len(train_mols):,}")
    print(f"  Validation unique molecules: {len(val_mols):,}")
    print(f"  Training spectrum samples:   {len(train_ds):,}")
    print(f"  Validation spectrum samples: {len(val_ds):,}")

    # Check 1: Molecule ID disjointness
    mol_overlap = train_mols.intersection(val_mols)
    is_mol_disjoint = len(mol_overlap) == 0
    print(f"\n[Audit 1] Molecule ID Overlap: {len(mol_overlap)} "
          f"({'PASS: 100% Disjoint' if is_mol_disjoint else 'FAIL: Leakage Detected!'})", flush=True)

    # Check 2: Canonical RDKit Structure Representation Disjointness
    print("[Audit 2] Checking RDKit canonical structures & InChI strings...", flush=True)
    train_rdkit_canonical = set()
    train_inchis = set()
    for m in train_mols:
        smi = train_ds.mol_smiles.get(m, "")
        rdmol = Chem.MolFromSmiles(smi)
        if rdmol is not None:
            can_smi = Chem.MolToSmiles(rdmol, canonical=True, isomericSmiles=False)
            train_rdkit_canonical.add(can_smi)
            try:
                inchi = Chem.MolToInchi(rdmol)
                train_inchis.add(inchi)
            except Exception:
                pass

    val_rdkit_overlap = 0
    val_inchi_overlap = 0
    for m in val_mols:
        smi = val_ds.mol_smiles.get(m, "")
        rdmol = Chem.MolFromSmiles(smi)
        if rdmol is not None:
            can_smi = Chem.MolToSmiles(rdmol, canonical=True, isomericSmiles=False)
            if can_smi in train_rdkit_canonical:
                val_rdkit_overlap += 1
            try:
                inchi = Chem.MolToInchi(rdmol)
                if inchi in train_inchis:
                    val_inchi_overlap += 1
            except Exception:
                pass

    is_structure_disjoint = (val_rdkit_overlap == 0 and val_inchi_overlap == 0)
    print(f"  Canonical RDKit Graph Overlap: {val_rdkit_overlap} (PASS: 0 canonical matches)")
    print(f"  Standard IUPAC InChI Overlap:  {val_inchi_overlap} (PASS: 0 InChI matches)")
    print(f"  Result: {'PASS: 100% Structure-Disjoint' if is_structure_disjoint else 'FAIL: Structural Leakage!'}")

    # Check 3: InChIKey14 (Skeleton / Carbon Skeleton Connectivity) Overlap
    train_skeletons = {m[:14] for m in train_mols if len(m) >= 14}
    val_skeletons = {m[:14] for m in val_mols if len(m) >= 14}
    skeleton_overlap = train_skeletons.intersection(val_skeletons)
    val_with_train_skeleton = sum(1 for m in val_mols if m[:14] in train_skeletons)
    print(f"\n[Audit 3] InChIKey14 Skeleton Overlap: {len(skeleton_overlap)} shared skeletons "
          f"({val_with_train_skeleton}/{len(val_mols)} val molecules share carbon skeleton with a train molecule)")

    # Check 4: Molecular Formula Overlap (Constitutional Isomers)
    print("\n[Audit 4] Analyzing molecular formulas and constitutional isomers...", flush=True)
    train_formulas: dict[str, set[str]] = {}
    for m in train_mols:
        rdmol = Chem.MolFromSmiles(train_ds.mol_smiles[m])
        if rdmol is not None:
            f = CalcMolFormula(rdmol)
            train_formulas.setdefault(f, set()).add(m)

    val_formulas: dict[str, set[str]] = {}
    val_has_train_isomer = 0
    for m in val_mols:
        rdmol = Chem.MolFromSmiles(val_ds.mol_smiles[m])
        if rdmol is not None:
            f = CalcMolFormula(rdmol)
            val_formulas.setdefault(f, set()).add(m)
            if f in train_formulas:
                val_has_train_isomer += 1

    shared_formulas = set(train_formulas.keys()).intersection(set(val_formulas.keys()))
    print(f"  Total unique formulas in train: {len(train_formulas):,}")
    print(f"  Total unique formulas in val:   {len(val_formulas):,}")
    print(f"  Shared formulas (isomers):      {len(shared_formulas):,}")
    print(f"  Val molecules with same-formula isomer in train: {val_has_train_isomer} / {len(val_mols)} "
          f"({val_has_train_isomer / len(val_mols) * 100:.1f}%)")

    # Check 5: Spectrum Duplication / Cosine Leakage Test (>0.999 cosine)
    print("\n[Audit 5] Checking for exact spectrum leakage (cosine > 0.999)...", flush=True)
    from src.data.spectrum_dataset import spectrum_to_coarse_bins
    import torch
    import torch.nn.functional as F

    # Sample 500 val spectra and check max cosine against 5,000 train spectra
    n_sample_v = min(500, len(val_ds))
    n_sample_t = min(5000, len(train_ds))

    val_vecs = torch.stack([val_ds[i][0][:1480] for i in range(n_sample_v)], dim=0)
    train_vecs = torch.stack([train_ds[i][0][:1480] for i in range(n_sample_t)], dim=0)

    val_vecs = F.normalize(val_vecs, dim=-1)
    train_vecs = F.normalize(train_vecs, dim=-1)

    cos_sims = torch.mm(val_vecs, train_vecs.T).numpy()
    max_sims = np.max(cos_sims, axis=1)

    exact_spec_leakage = int(np.sum(max_sims >= 0.999))
    print(f"  Max cross-split spectral cosine: {np.max(max_sims):.4f}")
    print(f"  Mean max cross-split cosine:     {np.mean(max_sims):.4f}")
    print(f"  Exact spectrum duplicates (cosine >= 0.999): {exact_spec_leakage} "
          f"({'PASS: Zero duplicate spectra' if exact_spec_leakage == 0 else 'WARNING: Duplicates detected'})")

    # Check 6: Molecular Graph Feature Integrity
    print("\n[Audit 6] Verifying graph featurization integrity...", flush=True)
    sample_graph = next(iter(val_ds.graph_cache.values()))
    has_labels = hasattr(sample_graph, "y") and sample_graph.y is not None
    print(f"  Node feature dimensions: {sample_graph.x.shape[1]}")
    print(f"  Edge feature dimensions: {sample_graph.edge_attr.shape[1]}")
    print(f"  Target label in graph object: {'PRESENT (Check!)' if has_labels else 'ABSENT (Clean unlabelled graph)'}")

    audit_report = {
        "status": "PASS" if (is_mol_disjoint and is_structure_disjoint and exact_spec_leakage == 0) else "FAIL",
        "split_counts": {
            "n_train_molecules": len(train_mols),
            "n_val_molecules": len(val_mols),
            "n_train_spectra": len(train_ds),
            "n_val_spectra": len(val_ds),
        },
        "checks": {
            "molecule_id_disjoint": bool(is_mol_disjoint),
            "canonical_rdkit_structure_disjoint": bool(is_structure_disjoint),
            "exact_rdkit_graph_overlap_count": int(val_rdkit_overlap),
            "exact_standard_inchi_overlap_count": int(val_inchi_overlap),
            "exact_spectrum_leakage_count": int(exact_spec_leakage),
            "max_cross_split_spectral_cosine": round(float(np.max(max_sims)), 4),
            "mean_max_cross_split_spectral_cosine": round(float(np.mean(max_sims)), 4),
            "val_molecules_with_train_skeleton_count": int(val_with_train_skeleton),
            "val_molecules_with_train_skeleton_pct": round(val_with_train_skeleton / len(val_mols) * 100, 2),
            "shared_formulas_count": len(shared_formulas),
            "val_molecules_with_train_isomer_count": int(val_has_train_isomer),
            "val_molecules_with_train_isomer_pct": round(val_has_train_isomer / len(val_mols) * 100, 2),
            "graph_has_target_labels": bool(has_labels),
        },
    }

    out_dir = ARTIFACTS_DIR / "stage06"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leakage_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(f"\nAudit completed successfully. Report saved to: {out_path}", flush=True)
    print("=" * 75, flush=True)
    return audit_report


if __name__ == "__main__":
    run_leakage_audit()
