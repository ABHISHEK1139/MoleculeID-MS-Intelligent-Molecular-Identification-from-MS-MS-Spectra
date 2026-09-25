"""Phase 2: High-Performance Multi-Core Parser for MoNA (.msp) and GNPS (.mgf).

Produces a unified, quality-filtered, indexed parquet file at:
  artifacts/external/external_spectra.parquet

Schema (preserving full provenance):
  spectrum_id        : str   - unique ID per spectrum
  molecule_id        : str   - canonical SMILES (RDKit)
  canonical_smiles   : str   - canonical SMILES (RDKit)
  precursor_mz       : float64
  precursor_charge   : int8
  precursor_type     : str   - adduct string, e.g. "[M+H]+"
  collision_energy   : float32 - normalized to eV, NaN if unknown
  ion_mode           : str   - "positive" or "negative"
  peaks_mz           : list[float32]
  peaks_intensity    : list[float32]  - sqrt-scaled, L2-normalized
  source_library     : str   - "MoNA" or "GNPS-<sublibrary>"
  neutral_mass       : float64
  formula            : str   - molecular formula (when available)
  peak_count         : int16
  inchikey           : str   - InChIKey (when available)
"""
from __future__ import annotations

import concurrent.futures
import os
import re
import sys
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from src.core.adducts import neutral_mass as compute_neutral_mass
from src.core.preprocessing import deisotope_peaks

# Suppress RDKit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "artifacts" / "external"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Collision energy normalization ────────────────────────────────────────
_CE_NUM_RE = re.compile(r"[-+]?\d+\.?\d*")


def normalize_ce(ce_str: str | None) -> float:
    """Parse collision energy string to a single float in eV."""
    if ce_str is None or ce_str.strip() == "":
        return float("nan")
    nums = _CE_NUM_RE.findall(ce_str.strip())
    if not nums:
        return float("nan")
    vals = [abs(float(x)) for x in nums]
    return float(np.median(vals))


def canonicalize_smiles(smi: str | None) -> str | None:
    """Canonicalize SMILES using RDKit. Returns None if invalid."""
    if not smi or smi == "N/A":
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception:
        return None


def preprocess_peaks(
    mzs: np.ndarray, intensities: np.ndarray, max_peaks: int = 60
) -> tuple[np.ndarray, np.ndarray] | None:
    """Deisotope, sqrt-scale, L2-normalize, top-N. Returns None if <3 peaks."""
    if mzs.size < 3:
        return None

    # Normalize raw intensities to [0, 1]
    imax = intensities.max()
    if imax > 0:
        intensities = intensities / imax

    # Deisotope
    if mzs.size > 1:
        mzs, intensities = deisotope_peaks(mzs, intensities)

    if mzs.size < 3:
        return None

    # Sqrt intensity scaling
    intensities = np.sqrt(np.maximum(intensities, 0.0))

    # Unit L2 normalization
    norm = np.linalg.norm(intensities)
    if norm <= 0:
        return None
    intensities = intensities / norm

    # Top-N peaks by intensity, then sort by m/z
    if mzs.size > max_peaks:
        top_idx = np.argsort(-intensities)[:max_peaks]
        top_idx = top_idx[np.argsort(mzs[top_idx])]
        mzs = mzs[top_idx]
        intensities = intensities[top_idx]

    return mzs, intensities


# ═══════════════════════════════════════════════════════════════════════════
# MoNA Multi-Core Worker & Boundary Logic
# ═══════════════════════════════════════════════════════════════════════════

def compute_mona_boundaries(filepath: Path, n_chunks: int) -> list[int]:
    """Divide file into exact block-aligned byte boundaries."""
    size = filepath.stat().st_size
    boundaries = [0]
    with open(filepath, "rb") as f:
        for i in range(1, n_chunks):
            target = i * size // n_chunks
            f.seek(target)
            f.readline()  # discard partial line
            while True:
                line = f.readline()
                if not line.strip():
                    break
            pos = f.tell()
            line = f.readline()
            while line and not line.strip():
                pos = f.tell()
                line = f.readline()
            boundaries.append(pos)
    boundaries.append(size)
    return boundaries


def parse_mona_chunk(
    args: tuple[str, int, int, int]
) -> tuple[list[dict], dict]:
    """Parse one byte slice of MoNA .msp file."""
    filepath_str, start_byte, end_byte, chunk_id = args
    filepath = Path(filepath_str)

    records = []
    block: dict[str, str] = {}
    peaks_mz: list[float] = []
    peaks_int: list[float] = []
    in_peaks = False

    n_total = 0
    n_parsed = 0
    n_no_smiles = 0
    n_no_precursor = 0
    n_bad_peaks = 0
    n_bad_smiles = 0

    t0 = time.time()

    def _flush_block():
        nonlocal n_total, n_parsed, n_no_smiles, n_no_precursor, n_bad_peaks, n_bad_smiles
        n_total += 1

        smiles_raw = None
        comments = block.get("comments", "")
        smi_match = re.search(r'"SMILES=([^"]+)"', comments)
        if smi_match:
            smiles_raw = smi_match.group(1).strip()

        if not smiles_raw:
            n_no_smiles += 1
            return

        can_smi = canonicalize_smiles(smiles_raw)
        if not can_smi:
            n_bad_smiles += 1
            return

        prec_mz_str = block.get("precursormz", "")
        if not prec_mz_str:
            n_no_precursor += 1
            return
        try:
            prec_mz = float(prec_mz_str)
        except ValueError:
            n_no_precursor += 1
            return

        if len(peaks_mz) < 3:
            n_bad_peaks += 1
            return

        mzs = np.array(peaks_mz, dtype=np.float32)
        ints = np.array(peaks_int, dtype=np.float32)

        result = preprocess_peaks(mzs, ints)
        if result is None:
            n_bad_peaks += 1
            return

        mzs_proc, ints_proc = result

        adduct = block.get("precursor_type", "").strip()
        ion_mode_raw = block.get("ion_mode", "").strip().upper()
        ion_mode = "negative" if ion_mode_raw.startswith("N") else "positive"
        ce_str = block.get("collision_energy", "")
        formula = block.get("formula", "").strip()
        inchikey = block.get("inchikey", "").strip()
        db_id = block.get("db#", "").strip()
        spectrum_id = f"MoNA:{db_id}" if db_id else f"MoNA:c{chunk_id}_{n_total}"

        nm = compute_neutral_mass(prec_mz, adduct) if adduct else None

        charge = 1
        charge_match = re.search(r"\](\d*)([+-])", adduct)
        if charge_match:
            charge = int(charge_match.group(1)) if charge_match.group(1) else 1
            if charge_match.group(2) == "-":
                charge = -charge

        records.append({
            "spectrum_id": spectrum_id,
            "molecule_id": can_smi,
            "canonical_smiles": can_smi,
            "precursor_mz": prec_mz,
            "precursor_charge": charge,
            "precursor_type": adduct,
            "collision_energy": normalize_ce(ce_str),
            "ion_mode": ion_mode,
            "peaks_mz": mzs_proc.tolist(),
            "peaks_intensity": ints_proc.tolist(),
            "source_library": "MoNA",
            "neutral_mass": float(nm) if nm is not None and nm > 0 else float("nan"),
            "formula": formula if formula else "",
            "peak_count": int(mzs_proc.size),
            "inchikey": inchikey if inchikey else "",
        })
        n_parsed += 1

    with open(filepath, "rb") as f:
        f.seek(start_byte)
        while f.tell() < end_byte or (in_peaks and block):
            line_b = f.readline()
            if not line_b:
                break
            line = line_b.decode("utf-8", errors="replace").rstrip("\r\n")

            if not line.strip():
                if in_peaks and block:
                    _flush_block()
                block = {}
                peaks_mz = []
                peaks_int = []
                in_peaks = False
                if f.tell() >= end_byte:
                    break
                continue

            if in_peaks:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks_mz.append(float(parts[0]))
                        peaks_int.append(float(parts[1]))
                    except ValueError:
                        pass
                continue

            if line.lower().startswith("num peaks:"):
                in_peaks = True
                continue

            colon_pos = line.find(":")
            if colon_pos > 0:
                key = line[:colon_pos].strip().lower()
                value = line[colon_pos + 1:].strip()
                block[key] = value

    if in_peaks and block:
        _flush_block()

    stats = {
        "chunk_id": chunk_id,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_no_smiles": n_no_smiles,
        "n_no_precursor": n_no_precursor,
        "n_bad_peaks": n_bad_peaks,
        "n_bad_smiles": n_bad_smiles,
        "elapsed": time.time() - t0,
    }
    return records, stats


# ═══════════════════════════════════════════════════════════════════════════
# GNPS .mgf Worker
# ═══════════════════════════════════════════════════════════════════════════

def parse_gnps_file(args: tuple[str, str]) -> tuple[list[dict], dict]:
    """Parse a single GNPS .mgf file."""
    filepath_str, source_name = args
    filepath = Path(filepath_str)

    t0 = time.time()
    records = []
    n_total = 0
    n_parsed = 0
    n_no_smiles = 0
    n_bad_smiles = 0
    n_bad_peaks = 0

    block: dict[str, str] = {}
    peaks_mz: list[float] = []
    peaks_int: list[float] = []
    in_block = False

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n\r")

            if line.strip() == "BEGIN IONS":
                block = {}
                peaks_mz = []
                peaks_int = []
                in_block = True
                continue

            if line.strip() == "END IONS":
                if not in_block:
                    continue
                in_block = False
                n_total += 1

                smiles_raw = block.get("smiles", "")
                if not smiles_raw or smiles_raw == "N/A":
                    n_no_smiles += 1
                    continue

                can_smi = canonicalize_smiles(smiles_raw)
                if not can_smi:
                    n_bad_smiles += 1
                    continue

                pepmass_str = block.get("pepmass", "")
                if not pepmass_str:
                    continue
                try:
                    prec_mz = float(pepmass_str.split()[0])
                except ValueError:
                    continue

                if len(peaks_mz) < 3:
                    n_bad_peaks += 1
                    continue

                mzs = np.array(peaks_mz, dtype=np.float32)
                ints = np.array(peaks_int, dtype=np.float32)

                result = preprocess_peaks(mzs, ints)
                if result is None:
                    n_bad_peaks += 1
                    continue

                mzs_proc, ints_proc = result

                ion_mode_raw = block.get("ionmode", "Positive").strip()
                ion_mode = "negative" if "neg" in ion_mode_raw.lower() else "positive"

                charge_str = block.get("charge", "1")
                try:
                    charge = int(charge_str.replace("+", "").replace("-", ""))
                    if charge == 0:
                        charge = 1
                    if "-" in charge_str:
                        charge = -charge
                except ValueError:
                    charge = 1

                name = block.get("name", "")
                adduct = ""
                adduct_match = re.search(r"(\[.*?\][+-]?\d*[+-]?)", name)
                if adduct_match:
                    adduct = adduct_match.group(1)
                if not adduct:
                    if "M+H" in name:
                        adduct = "[M+H]+"
                    elif "M-H" in name:
                        adduct = "[M-H]-"
                    elif ion_mode == "positive":
                        adduct = "[M+H]+"
                    else:
                        adduct = "[M-H]-"

                spectrum_id = block.get("spectrumid", f"GNPS:{source_name}:{n_total}")
                formula = block.get("formula", "")
                nm = compute_neutral_mass(prec_mz, adduct) if adduct else None

                records.append({
                    "spectrum_id": spectrum_id,
                    "molecule_id": can_smi,
                    "canonical_smiles": can_smi,
                    "precursor_mz": prec_mz,
                    "precursor_charge": charge,
                    "precursor_type": adduct,
                    "collision_energy": float("nan"),
                    "ion_mode": ion_mode,
                    "peaks_mz": mzs_proc.tolist(),
                    "peaks_intensity": ints_proc.tolist(),
                    "source_library": f"GNPS-{source_name}",
                    "neutral_mass": float(nm) if nm is not None and nm > 0 else float("nan"),
                    "formula": formula if formula else "",
                    "peak_count": int(mzs_proc.size),
                    "inchikey": block.get("inchikey", block.get("inchiauxinfo", "")),
                })
                n_parsed += 1
                continue

            if not in_block:
                continue

            if "=" in line:
                eq_pos = line.index("=")
                key = line[:eq_pos].strip().lower()
                value = line[eq_pos + 1:].strip()
                block[key] = value
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks_mz.append(float(parts[0]))
                        peaks_int.append(float(parts[1]))
                    except ValueError:
                        pass

    stats = {
        "file": filepath.name,
        "n_total": n_total,
        "n_parsed": n_parsed,
        "n_no_smiles": n_no_smiles,
        "n_bad_smiles": n_bad_smiles,
        "n_bad_peaks": n_bad_peaks,
        "elapsed": time.time() - t0,
    }
    return records, stats


# ═══════════════════════════════════════════════════════════════════════════
# Main Execution Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    num_cpus = os.cpu_count() or 4
    n_workers = min(12, num_cpus)

    print("=" * 60)
    print(f"Phase 2 External Spectral Database Parser")
    print(f"Using {n_workers} parallel workers on {num_cpus}-core system")
    print("=" * 60, flush=True)

    mona_parquet_path = OUTPUT_DIR / "mona_spectra.parquet"
    gnps_parquet_path = OUTPUT_DIR / "gnps_spectra.parquet"
    final_parquet_path = OUTPUT_DIR / "external_spectra.parquet"

    # ── 1. Parse MoNA in Parallel ─────────────────────────────────────────
    mona_path = DATASET_DIR / "MoNA-export-LC-MS-MS_Spectra.msp"
    if mona_parquet_path.exists():
        print(f"\n[Checkpoint] Found existing MoNA parquet: {mona_parquet_path}")
        df_mona = pd.read_parquet(mona_parquet_path)
        print(f"Loaded {len(df_mona):,} MoNA spectra from checkpoint.")
    elif mona_path.exists():
        print(f"\nComputing byte boundaries for {n_workers} chunks of MoNA (9.46 GB)...")
        boundaries = compute_mona_boundaries(mona_path, n_workers)

        tasks = []
        for i in range(n_workers):
            tasks.append((str(mona_path), boundaries[i], boundaries[i + 1], i))

        print(f"Launching {n_workers} parallel MoNA worker processes...")
        t_mona = time.time()
        mona_records = []
        total_scanned = 0
        total_kept = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(parse_mona_chunk, t) for t in tasks]
            for f in concurrent.futures.as_completed(futures):
                chunk_records, stats = f.result()
                mona_records.extend(chunk_records)
                total_scanned += stats["n_total"]
                total_kept += stats["n_parsed"]
                print(
                    f"  Chunk {stats['chunk_id']:02d} done: {stats['n_total']:>7,} scanned -> "
                    f"{stats['n_parsed']:>7,} kept in {stats['elapsed']:.1f}s",
                    flush=True
                )

        print(f"\nMoNA Parallel Parsing Complete in {time.time() - t_mona:.1f}s!")
        print(f"Total blocks scanned: {total_scanned:,} | Spectra kept: {total_kept:,}")

        # Checkpoint MoNA to parquet immediately
        df_mona = pd.DataFrame(mona_records)
        df_mona.to_parquet(mona_parquet_path, compression="zstd")
        print(f"Saved MoNA checkpoint to {mona_parquet_path} ({mona_parquet_path.stat().st_size / 1e6:.1f} MB)")
        del mona_records
    else:
        print(f"WARNING: MoNA file not found at {mona_path}")
        df_mona = pd.DataFrame()

    # ── 2. Parse GNPS in Parallel ─────────────────────────────────────────
    gnps_files = sorted(DATASET_DIR.glob("*.mgf"))
    if gnps_parquet_path.exists():
        print(f"\n[Checkpoint] Found existing GNPS parquet: {gnps_parquet_path}")
        df_gnps = pd.read_parquet(gnps_parquet_path)
        print(f"Loaded {len(df_gnps):,} GNPS spectra from checkpoint.")
    elif gnps_files:
        print(f"\n{'='*60}")
        print(f"Parsing {len(gnps_files)} GNPS .mgf files in parallel across {n_workers} workers")
        print(f"{'='*60}", flush=True)

        gnps_tasks = [(str(p), p.stem) for p in gnps_files]
        t_gnps = time.time()
        gnps_records = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_workers, len(gnps_files))) as executor:
            futures = [executor.submit(parse_gnps_file, t) for t in gnps_tasks]
            for f in concurrent.futures.as_completed(futures):
                file_records, stats = f.result()
                gnps_records.extend(file_records)
                print(
                    f"  {stats['file']:<45s} {stats['n_total']:>6,} blocks -> "
                    f"{stats['n_parsed']:>6,} kept ({stats['elapsed']:.1f}s)",
                    flush=True
                )

        print(f"\nGNPS Parallel Parsing Complete in {time.time() - t_gnps:.1f}s!")
        print(f"Extracted {len(gnps_records):,} GNPS spectra.")

        # Checkpoint GNPS to parquet immediately
        df_gnps = pd.DataFrame(gnps_records)
        df_gnps.to_parquet(gnps_parquet_path, compression="zstd")
        print(f"Saved GNPS checkpoint to {gnps_parquet_path} ({gnps_parquet_path.stat().st_size / 1e6:.1f} MB)")
        del gnps_records
    else:
        print("No GNPS files found.")
        df_gnps = pd.DataFrame()

    # ── 3. Combine and Deduplicate ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Building unified deduplicated external spectral library")
    print(f"{'='*60}", flush=True)

    df_list = []
    if not df_mona.empty:
        df_list.append(df_mona)
    if not df_gnps.empty:
        df_list.append(df_gnps)

    if not df_list:
        print("ERROR: No records parsed from any source!")
        return

    df = pd.concat(df_list, ignore_index=True)
    del df_list, df_mona, df_gnps

    print(f"Total raw combined spectra: {len(df):,}")
    print(f"Unique molecules (canonical SMILES): {df['canonical_smiles'].nunique():,}")

    # Remove rows with invalid neutral mass
    n_before = len(df)
    df = df[df["neutral_mass"].notna() & (df["neutral_mass"] > 0)].copy()
    print(f"After valid neutral mass filter: {len(df):,} ({n_before - len(df):,} removed)")

    # ── 4. Deduplicate: keep richest spectrum per (molecule, CE-bin) ──────
    CE_BIN_WIDTH = 5.0

    def ce_bin(ce: float) -> int:
        if np.isfinite(ce):
            return int(round(ce / CE_BIN_WIDTH))
        return -9999

    df["_ce_bin"] = df["collision_energy"].apply(ce_bin)

    # Sort by peak_count descending so groupby.first keeps highest quality spectrum
    df = df.sort_values("peak_count", ascending=False)
    df_dedup = df.groupby(["canonical_smiles", "_ce_bin"]).first().reset_index()
    df_dedup = df_dedup.drop(columns=["_ce_bin"])

    print(f"After dedup (mol + CE-bin): {len(df_dedup):,}")
    print(f"Unique molecules after dedup: {df_dedup['canonical_smiles'].nunique():,}")

    # ── 5. Check Overlap with Existing Train Reference Library ───────────
    ref_path = ROOT / "kaggle_dataset" / "reference_library_multice.parquet"
    overlap_count = 0
    novel_count = 0
    if ref_path.exists():
        ref_smiles = set(
            pd.read_parquet(ref_path, columns=["normalized_smiles"])["normalized_smiles"]
        )
        ext_smiles = set(df_dedup["canonical_smiles"])
        overlap = ext_smiles & ref_smiles
        novel = ext_smiles - ref_smiles
        overlap_count = len(overlap)
        novel_count = len(novel)
        print("\n--- Overlap Analysis with Existing Train Library ---")
        print(f"Existing train library molecules: {len(ref_smiles):,}")
        print(f"External library unique molecules: {len(ext_smiles):,}")
        print(f"Overlap (already in train library): {overlap_count:,}")
        print(f"Novel (brand new external molecules): {novel_count:,}")

    # ── 6. Sort by neutral mass for binary search indexing ────────────────
    df_dedup = df_dedup.sort_values("neutral_mass").reset_index(drop=True)

    # ── 7. Cast dtypes ────────────────────────────────────────────────────
    df_dedup["precursor_charge"] = df_dedup["precursor_charge"].astype(np.int8)
    df_dedup["peak_count"] = df_dedup["peak_count"].astype(np.int16)
    df_dedup["collision_energy"] = df_dedup["collision_energy"].astype(np.float32)

    # ── 8. Save Final Unified Dataset ─────────────────────────────────────
    df_dedup.to_parquet(final_parquet_path, compression="zstd")
    print(f"\nFinal unified database written to {final_parquet_path}")
    print(f"File size: {final_parquet_path.stat().st_size / 1e6:.2f} MB")

    # ── 9. Summary Statistics & Diagnostic Log ───────────────────────────
    n_mols = df_dedup["canonical_smiles"].nunique()
    specs_per_mol = df_dedup.groupby("canonical_smiles").size()
    by_source = df_dedup["source_library"].value_counts()

    print("\n" + "=" * 60)
    print("External Spectral Database Built Successfully")
    print("=" * 60)
    print(f"Total reference spectra: {len(df_dedup):,}")
    print(f"Unique molecules:        {n_mols:,}")
    print(f"Avg spectra/molecule:    {specs_per_mol.mean():.2f}")
    print(f"Median spectra/molecule: {specs_per_mol.median():.1f}")
    print(f"Max spectra/molecule:    {specs_per_mol.max()}")
    print("\nBy source library:")
    for src, cnt in by_source.items():
        smis = df_dedup[df_dedup["source_library"] == src]["canonical_smiles"].nunique()
        print(f"  {src:<50s} {cnt:>7,} spectra ({smis:>6,} molecules)")
    print(
        f"\nNeutral mass range: [{df_dedup['neutral_mass'].min():.2f}, {df_dedup['neutral_mass'].max():.2f}]"
    )
    print(f"Total elapsed time: {time.time() - t_start:.1f}s")

    summary_path = OUTPUT_DIR / "parsing_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("External Spectral Database Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total spectra: {len(df_dedup):,}\n")
        f.write(f"Unique molecules: {n_mols:,}\n")
        f.write(f"Avg spectra/mol: {specs_per_mol.mean():.2f}\n")
        f.write(f"Median spectra/mol: {specs_per_mol.median():.1f}\n\n")
        f.write("By source:\n")
        for src, cnt in by_source.items():
            smis = df_dedup[df_dedup["source_library"] == src]["canonical_smiles"].nunique()
            f.write(f"  {src}: {cnt:,} spectra, {smis:,} molecules\n")
        if ref_path.exists():
            f.write(f"\nOverlap with train library: {overlap_count:,} molecules\n")
            f.write(f"Novel external molecules: {novel_count:,} molecules\n")

    print(f"Diagnostic summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
