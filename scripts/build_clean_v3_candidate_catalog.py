"""Build the versioned Clean v4 candidate universe.

Integrates:
1. All training molecules from dataset/train.parquet and legacy catalog (277,566 structures).
2. COCONUT 2.0 natural product catalog (436,389 structures).
3. ChEBI and LIPID MAPS bioactive metabolite catalog (62,744 structures).

Features:
- Deduplicates canonical structures with complete provenance tracking (source bitmask/list).
- Computes exact RDKit monoisotopic masses, formulas, InChIKey14 prefixes, and full InChIKeys.
- Strictly sorts ascending by exact_mass for monotonic binary search.
- Generates 2,048-bit packed Morgan fingerprints (256 bytes uint8) for all candidates.
- Precomputes 256-D GNN molecular embeddings on CUDA using fine-tuned MoleculeGNN.
- Exports candidate_union.parquet, candidate_fps.npy, candidate_embeddings.npy, and candidate_manifest.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.mol_graph import ATOM_FDIM, BOND_FDIM, smiles_to_graph
from src.models.molecule_encoder import MoleculeGNN


def derive_full_inchikey(smiles: str) -> str:
    if not isinstance(smiles, str) or not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        key = Chem.MolToInchiKey(mol) if mol is not None else ""
    except Exception:
        key = ""
    return key if len(key) == 27 else ""


def make_dummy_graph() -> Data:
    x = torch.zeros((1, ATOM_FDIM), dtype=torch.float32)
    x[0, 2] = 1.0  # carbon
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, BOND_FDIM), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _compute_formulas_chunk(smiles_chunk: list[str]) -> list[str]:
    """Helper worker to compute molecular formulas in parallel."""
    formulas = []
    for s in smiles_chunk:
        try:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                formulas.append(rdMolDescriptors.CalcMolFormula(m))
            else:
                formulas.append("")
        except Exception:
            formulas.append("")
    return formulas


def _compute_fps_chunk(smiles_chunk: list[str]) -> np.ndarray:
    """Helper worker to compute packed 2048-bit Morgan fingerprints in parallel."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    n = len(smiles_chunk)
    packed = np.zeros((n, 256), dtype=np.uint8)
    for i, s in enumerate(smiles_chunk):
        try:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                fp_bits = gen.GetFingerprintAsNumPy(m)
                packed[i] = np.packbits(fp_bits)
        except Exception:
            pass
    return packed


def _compute_inchikeys_chunk(smiles_chunk: list[str]) -> list[str]:
    return [derive_full_inchikey(smiles) for smiles in smiles_chunk]


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return ROOT / "artifacts" / "v4_clean"
    path = Path(output_dir)
    if not path.is_absolute():
        path = ROOT / path
    return path


def main(output_dir: str | Path | None = None):
    t_start_total = time.time()
    num_cpus = os.cpu_count() or 4
    n_workers = min(12, num_cpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = _resolve_output_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("  PHASE 3: UNIFIED CANDIDATE UNIVERSE BUILDER (COCONUT + ChEBI + LIPID MAPS + TRAIN)")
    print(f"  Workers: {n_workers} | Compute Device: {device} | Destination: {out_dir}")
    print("=" * 85, flush=True)

    # -------------------------------------------------------------------------
    # 1. Ingest All Sources
    # -------------------------------------------------------------------------
    print("\n[1/6] Ingesting candidate source datasets...", flush=True)

    # 1A. Legacy Train Catalog
    legacy_path = ROOT / "kaggle_dataset" / "candidate_library.parquet"
    print(f"  -> Reading legacy train catalog: {legacy_path}...")
    df_legacy = pd.read_parquet(legacy_path)
    legacy_smis = df_legacy["normalized_smiles"].tolist()
    legacy_forms = df_legacy["molecular_formula"].tolist()
    legacy_masses = df_legacy["exact_mass"].to_numpy(dtype=np.float64)
    print(f"     Loaded {len(df_legacy):,} legacy structures.")

    # 1B. Full train.parquet to catch any missing training SMILES
    train_pq = ROOT / "dataset" / "train.parquet"
    print(f"  -> Scanning full train dataset: {train_pq} for complete coverage...")
    df_train = pd.read_parquet(train_pq, columns=["normalized_smiles", "molecular_formula", "inchikey14"])
    train_map = {}
    for _, r in df_train.drop_duplicates(subset=["normalized_smiles"]).iterrows():
        train_map[r["normalized_smiles"]] = {
            "formula": r["molecular_formula"],
            "inchikey14": r["inchikey14"],
        }

    # 1C. COCONUT 2.0 Natural Products
    coco_meta_path = ROOT / "external_candidates" / "coco_meta.pkl"
    coco_mass_path = ROOT / "external_candidates" / "coco_mass.npy"
    print(f"  -> Reading COCONUT catalog: {coco_meta_path}...")
    with open(coco_meta_path, "rb") as f:
        coco_meta = pickle.load(f)
    coco_smis = coco_meta["smiles"].tolist()
    coco_keys = coco_meta["keys"].tolist()
    coco_masses = np.load(coco_mass_path).astype(np.float64)
    print(f"     Loaded {len(coco_smis):,} COCONUT structures.")

    # 1D. ChEBI + LIPID MAPS Bioactive Metabolites
    bio_meta_path = ROOT / "external_candidates" / "bio_meta.pkl"
    bio_mass_path = ROOT / "external_candidates" / "bio_mass.npy"
    print(f"  -> Reading ChEBI + LIPID MAPS catalog: {bio_meta_path}...")
    with open(bio_meta_path, "rb") as f:
        bio_meta = pickle.load(f)
    bio_smis = bio_meta["smiles"].tolist()
    bio_keys = bio_meta["keys"].tolist()
    bio_masses = np.load(bio_mass_path).astype(np.float64)
    print(f"     Loaded {len(bio_smis):,} ChEBI + LIPID MAPS structures.")

    # -------------------------------------------------------------------------
    # 2. Unified Deduplication and Provenance Tracking
    # -------------------------------------------------------------------------
    print("\n[2/6] Deduplicating and merging sources with provenance tracking...", flush=True)

    # Master registry keyed by canonical SMILES
    candidates: dict[str, dict[str, Any]] = {}

    # 2A. Ingest Train
    for smi, form, mass in zip(legacy_smis, legacy_forms, legacy_masses):
        ik14 = train_map.get(smi, {}).get("inchikey14", "")
        candidates[smi] = {
            "canonical_smiles": smi,
            "inchikey14": ik14,
            "inchikey": "",
            "molecular_formula": form,
            "exact_mass": mass,
            "sources": {"TRAIN"},
        }

    # Add any missing train molecules from train_map
    for smi, info in train_map.items():
        if smi not in candidates:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                mass = float(rdMolDescriptors.CalcExactMolWt(m))
                candidates[smi] = {
                    "canonical_smiles": smi,
                    "inchikey14": info["inchikey14"],
                    "inchikey": "",
                    "molecular_formula": info["formula"],
                    "exact_mass": mass,
                    "sources": {"TRAIN"},
                }

    print(f"  Total unique training structures registered: {len(candidates):,}")

    # 2B. Ingest COCONUT
    new_coco = 0
    merged_coco = 0
    for smi, ik14, mass in zip(coco_smis, coco_keys, coco_masses):
        if smi in candidates:
            candidates[smi]["sources"].add("COCONUT")
            merged_coco += 1
        else:
            candidates[smi] = {
                "canonical_smiles": smi,
                "inchikey14": ik14,
                "inchikey": "",
                "molecular_formula": None,  # Will batch compute
                "exact_mass": mass,
                "sources": {"COCONUT"},
            }
            new_coco += 1

    print(f"  COCONUT merged into existing: {merged_coco:,} | New COCONUT added: {new_coco:,}")

    # 2C. Ingest ChEBI + LIPID MAPS
    new_bio = 0
    merged_bio = 0
    for smi, ik14, mass in zip(bio_smis, bio_keys, bio_masses):
        if smi in candidates:
            candidates[smi]["sources"].add("CHEBI_LIPIDMAPS")
            merged_bio += 1
        else:
            candidates[smi] = {
                "canonical_smiles": smi,
                "inchikey14": ik14,
                "inchikey": "",
                "molecular_formula": None,  # Will batch compute
                "exact_mass": mass,
                "sources": {"CHEBI_LIPIDMAPS"},
            }
            new_bio += 1

    print(f"  ChEBI/LIPIDMAPS merged into existing: {merged_bio:,} | New bio added: {new_bio:,}")
    total_unique = len(candidates)
    print(f"\n  Total unified unique candidates: {total_unique:,}")

    # -------------------------------------------------------------------------
    # 3. Compute Missing Formulas & InChIKey14s in Parallel
    # -------------------------------------------------------------------------
    print("\n[3/6] Computing missing molecular formulas and connectivity keys...", flush=True)

    smis_missing_formula = [smi for smi, c in candidates.items() if not c["molecular_formula"]]
    print(f"  Computing formulas for {len(smis_missing_formula):,} external structures...")

    if smis_missing_formula:
        t0 = time.time()
        chunk_size = max(500, len(smis_missing_formula) // (n_workers * 4))
        chunks = [smis_missing_formula[i:i + chunk_size] for i in range(0, len(smis_missing_formula), chunk_size)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            chunk_results = list(executor.map(_compute_formulas_chunk, chunks))

        computed_forms = [f for cr in chunk_results for f in cr]
        for smi, f_str in zip(smis_missing_formula, computed_forms):
            candidates[smi]["molecular_formula"] = f_str
        print(f"  Computed all missing formulas in {time.time() - t0:.2f}s.")

    # Fill any missing InChIKey14 (e.g. for legacy entries that had none)
    smis_missing_ik = [smi for smi, c in candidates.items() if not c["inchikey14"]]
    if smis_missing_ik:
        print(f"  Filling InChIKey14 for {len(smis_missing_ik):,} entries...")
        for smi in smis_missing_ik:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                candidates[smi]["inchikey14"] = Chem.MolToInchiKey(m)[:14]
            else:
                candidates[smi]["inchikey14"] = "UNKNOWN"

    all_candidate_smis = list(candidates)
    key_chunk_size = max(500, len(all_candidate_smis) // (n_workers * 4))
    key_chunks = [all_candidate_smis[i:i + key_chunk_size] for i in range(0, len(all_candidate_smis), key_chunk_size)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        key_results = list(executor.map(_compute_inchikeys_chunk, key_chunks))
    computed_keys = [key for chunk in key_results for key in chunk]
    for smi, key in zip(all_candidate_smis, computed_keys):
        candidates[smi]["inchikey"] = key
    invalid_keys = [smi for smi, data in candidates.items() if not data["inchikey"] or len(data["inchikey"]) != 27]
    if invalid_keys:
        raise RuntimeError(f"RDKit failed to derive full InChIKey for {len(invalid_keys)} candidates")
    print(f"  Derived and validated full InChIKeys for {len(candidates):,} candidates.")

    # -------------------------------------------------------------------------
    # 4. Construct Sorted DataFrame & Save candidate_union.parquet
    # -------------------------------------------------------------------------
    print("\n[4/6] Constructing DataFrame and strictly sorting by exact_mass...", flush=True)

    rows = []
    for smi, data in candidates.items():
        src_str = ",".join(sorted(data["sources"]))
        rows.append({
            "canonical_smiles": smi,
            "inchikey14": data["inchikey14"],
            "inchikey": data["inchikey"],
            "molecular_formula": data["molecular_formula"],
            "exact_mass": data["exact_mass"],
            "source": src_str,
            "source_version": "v4_clean.1.0",
        })

    df_union = pd.DataFrame(rows)

    # Strictly sort by exact_mass ascending (mandatory for binary search)
    df_union.sort_values(by="exact_mass", ascending=True, inplace=True, ignore_index=True)
    df_union.insert(0, "candidate_id", np.arange(len(df_union), dtype=np.int32))

    assert df_union["exact_mass"].is_monotonic_increasing, "Candidate catalog MUST be strictly sorted by exact_mass!"
    print(f"  Sorted {len(df_union):,} candidates. Monotonic: {df_union['exact_mass'].is_monotonic_increasing}")
    print(f"  Mass range: {df_union['exact_mass'].min():.4f} Da to {df_union['exact_mass'].max():.4f} Da")

    cand_parquet_path = out_dir / "candidate_union.parquet"
    df_union.to_parquet(cand_parquet_path, index=False, engine="pyarrow")
    pq_size_mb = cand_parquet_path.stat().st_size / (1024 * 1024)
    print(f"  Saved {cand_parquet_path} ({pq_size_mb:.2f} MB)")

    # -------------------------------------------------------------------------
    # 5. Compute 2,048-Bit Morgan Fingerprints
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Precomputing 2048-bit Morgan fingerprints (256 bytes uint8) for {len(df_union):,} candidates...", flush=True)
    t_fp_start = time.time()

    all_smis = df_union["canonical_smiles"].tolist()
    chunk_size = max(500, len(all_smis) // (n_workers * 4))
    chunks = [all_smis[i:i + chunk_size] for i in range(0, len(all_smis), chunk_size)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        fp_results = list(executor.map(_compute_fps_chunk, chunks))

    all_fps = np.vstack(fp_results)
    assert all_fps.shape == (len(df_union), 256), f"Unexpected shape for all_fps: {all_fps.shape}"
    assert all_fps.dtype == np.uint8, f"Unexpected dtype: {all_fps.dtype}"

    fps_out_path = out_dir / "candidate_fps.npy"
    np.save(fps_out_path, all_fps)
    print(f"  Computed and saved Morgan fingerprints in {time.time() - t_fp_start:.2f}s -> {fps_out_path} ({fps_out_path.stat().st_size / 1e6:.2f} MB)")

    # -------------------------------------------------------------------------
    # 6. Precompute Molecular GNN Embeddings
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Precomputing 256-D fine-tuned MoleculeGNN embeddings on {device}...", flush=True)
    t_emb_start = time.time()

    s5_ckpt_path = ROOT / "artifacts" / "stage05" / "exp5_v2" / "checkpoints" / "best.pt"
    s5_data = torch.load(s5_ckpt_path, map_location=device, weights_only=False)
    mol_model = MoleculeGNN(embed_dim=256).to(device)
    if "mol_encoder_state_dict" in s5_data:
        mol_model.load_state_dict(s5_data["mol_encoder_state_dict"])
    elif "mol_encoder" in s5_data:
        mol_model.load_state_dict(s5_data["mol_encoder"])
    mol_model.eval()

    # Reuse legacy embeddings for exact SMILES matches to save time
    legacy_embs_path = ROOT / "kaggle_dataset" / "candidate_embeddings.npy"
    smi_to_legacy_emb = {}
    if legacy_embs_path.exists() and legacy_path.exists():
        print(f"  Indexing legacy embeddings from {legacy_embs_path} for fast reuse...")
        leg_embs = np.load(legacy_embs_path)
        for s, emb in zip(df_legacy["normalized_smiles"], leg_embs):
            smi_to_legacy_emb[s] = emb
        print(f"  Indexed {len(smi_to_legacy_emb):,} legacy embeddings.")

    n_total = len(df_union)
    embeddings = np.zeros((n_total, 256), dtype=np.float16)
    dummy_graph = make_dummy_graph()

    to_encode_indices = []
    to_encode_smiles = []

    for i, s in enumerate(all_smis):
        if s in smi_to_legacy_emb:
            embeddings[i] = smi_to_legacy_emb[s]
        else:
            to_encode_indices.append(i)
            to_encode_smiles.append(s)

    n_reused = n_total - len(to_encode_indices)
    n_to_encode = len(to_encode_indices)
    print(f"  Reused legacy embeddings: {n_reused:,} ({(n_reused / n_total) * 100:.1f}%)")
    print(f"  Remaining new structures to encode on GPU: {n_to_encode:,} ({(n_to_encode / n_total) * 100:.1f}%)")

    batch_size = 512
    n_batches = (n_to_encode + batch_size - 1) // batch_size
    n_failed = 0

    for b in range(n_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, n_to_encode)
        batch_smis = to_encode_smiles[b_start:b_end]
        batch_idx = to_encode_indices[b_start:b_end]

        graphs = []
        for s in batch_smis:
            g = smiles_to_graph(s, cache=False)
            if g is None or g.x is None or g.x.size(0) == 0:
                g = dummy_graph
                n_failed += 1
            graphs.append(g)

        bg = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            z = mol_model(bg)

        embeddings[batch_idx] = z.cpu().numpy().astype(np.float16)

        if (b % 100 == 0) or (b == n_batches - 1):
            elapsed = time.time() - t_emb_start
            done = b_end
            rate = done / max(elapsed, 0.001)
            eta_min = (n_to_encode - done) / max(rate, 0.001) / 60.0
            print(f"    Encoded [{done:6d}/{n_to_encode:6d}] ({done / n_to_encode * 100:.1f}%) | Rate: {rate:.0f} mols/s | ETA: {eta_min:.1f} min", flush=True)

    embs_out_path = out_dir / "candidate_embeddings.npy"
    np.save(embs_out_path, embeddings)
    print(f"  All embeddings saved to: {embs_out_path} ({embs_out_path.stat().st_size / 1e6:.2f} MB)")

    # -------------------------------------------------------------------------
    # 7. Generate Manifest with SHA-256 Hashes
    # -------------------------------------------------------------------------
    print("\n[7/7] Generating candidate catalog manifest with SHA-256 integrity hashes...", flush=True)

    def file_sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "catalog_version": "v4_clean.1.0",
        "output_directory": str(out_dir),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "molecule_key": "inchikey",
        "full_inchikey_column": "inchikey",
        "full_inchikey_nonempty": bool(df_union["inchikey"].notna().all() and (df_union["inchikey"].astype(str).str.len() == 27).all()),
        "total_candidates": len(df_union),
        "unique_full_inchikeys": int(df_union["inchikey"].nunique()),
        "source_counts": {
            "TRAIN_only": int((df_union["source"] == "TRAIN").sum()),
            "COCONUT_only": int((df_union["source"] == "COCONUT").sum()),
            "CHEBI_LIPIDMAPS_only": int((df_union["source"] == "CHEBI_LIPIDMAPS").sum()),
            "multi_source": int(df_union["source"].str.contains(",").sum()),
        },
        "mass_statistics": {
            "min_mass": float(df_union["exact_mass"].min()),
            "max_mass": float(df_union["exact_mass"].max()),
            "median_mass": float(df_union["exact_mass"].median()),
        },
        "artifacts": {
            "candidate_union_parquet": {
                "file": "candidate_union.parquet",
                "sha256": file_sha256(cand_parquet_path),
                "size_bytes": cand_parquet_path.stat().st_size,
            },
            "candidate_fps_npy": {
                "file": "candidate_fps.npy",
                "sha256": file_sha256(fps_out_path),
                "size_bytes": fps_out_path.stat().st_size,
                "shape": list(all_fps.shape),
                "dtype": str(all_fps.dtype),
            },
            "candidate_embeddings_npy": {
                "file": "candidate_embeddings.npy",
                "sha256": file_sha256(embs_out_path),
                "size_bytes": embs_out_path.stat().st_size,
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
            },
        },
    }

    manifest_path = out_dir / "candidate_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest written to: {manifest_path}")

    total_time = time.time() - t_start_total
    print("\n" + "=" * 85)
    print(f"  PHASE 3 CATALOG EXPANSION COMPLETED SUCCESSFULLY IN {total_time / 60.0:.2f} MINUTES")
    print(f"  Total Candidates: {len(df_union):,} | New Structures: {len(df_union) - len(df_legacy):,}")
    print("=" * 85, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)
    args = parser.parse_args()
    main(args.output_dir)
