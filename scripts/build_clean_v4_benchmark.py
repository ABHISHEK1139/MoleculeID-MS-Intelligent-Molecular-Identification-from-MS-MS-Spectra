"""Build the repaired Clean v4 benchmark and reference library.

The benchmark uses full RDKit InChIKeys for identity, observed precursor and
spectrum fields for retrieval, and separate train-derived C1/C2 and external C3
cohorts. The reference library is rebuilt from the unified source library and
never from the stale v3 clean reference.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem

from src.core.candidate_retrieval import C13_DIFF, retrieve_candidates_union
from src.core.preprocessing_v3 import neutral_mass, preprocess_spectrum

DEFAULT_OUT_DIR = ROOT / "artifacts" / "v4_clean"
OUT_DIR = DEFAULT_OUT_DIR
RETRIEVAL_PARAMETERS = {
    "ppm_windows": [20.0, 50.0, 100.0],
    "use_c13_isotopes": True,
    "c13_ppm": 30.0,
    "nominal_tol": 0.5,
    "early_stop": False,
    "union": True,
}
FULL_KEY_PATTERN = re.compile(r"^[A-Z0-9]{14}-[A-Z0-9]{10}-[A-Z0-9]$")
STREAM_BATCH_SIZE = 8192
STREAM_PROGRESS_ROWS = 100000


def is_valid_full_inchikey(value: Any) -> bool:
    return isinstance(value, str) and bool(FULL_KEY_PATTERN.fullmatch(value))


def derive_full_inchikey(smiles: Any) -> str:
    if not isinstance(smiles, str) or not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        key = Chem.MolToInchiKey(mol) if mol is not None else ""
    except Exception:
        key = ""
    return key if is_valid_full_inchikey(key) else ""


def _key_map_for_smiles(smiles: Iterable[Any], initial: dict[str, str] | None = None) -> dict[str, str]:
    initial = initial or {}
    values = list(dict.fromkeys(str(s) for s in smiles if isinstance(s, str) and s))
    result: dict[str, str] = {}
    invalid = []
    for smiles_value in values:
        key = initial.get(smiles_value, "")
        if not is_valid_full_inchikey(key):
            key = derive_full_inchikey(smiles_value)
        if not is_valid_full_inchikey(key):
            invalid.append(smiles_value)
        else:
            result[smiles_value] = key
    if invalid:
        raise ValueError(f"RDKit failed to derive full InChIKey for {len(invalid)} structures")
    return result


def _key_column(frame: pd.DataFrame, query: bool = False) -> str:
    names = ("true_inchikey", "inchikey", "full_inchikey") if query else ("inchikey", "full_inchikey")
    for name in names:
        if name in frame.columns:
            return name
    raise KeyError(f"No full InChIKey column found in columns: {list(frame.columns)}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def spectrum_fingerprint(mzs: Any, intensities: Any, decimals: int = 4) -> str:
    mz_values = np.asarray(mzs, dtype=np.float64).reshape(-1)
    intensity_values = np.asarray(intensities, dtype=np.float64).reshape(-1)
    if mz_values.size != intensity_values.size or mz_values.size == 0:
        return ""
    mz_q = np.round(mz_values, decimals)
    intensity_q = np.round(intensity_values, decimals)
    payload = f"{mz_q.size}:".encode("ascii") + mz_q.tobytes() + intensity_q.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _mz_fingerprint(mzs: Any, decimals: int = 3) -> str:
    values = np.asarray(mzs, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return ""
    quantized = np.round(values, decimals)
    return hashlib.sha256(f"{quantized.size}:".encode("ascii") + quantized.tobytes()).hexdigest()


def _spectrum_fingerprints(mzs: Any, intensities: Any) -> list[str]:
    values = {spectrum_fingerprint(mzs, intensities), _mz_fingerprint(mzs)}
    values.discard("")
    return sorted(values)


def _as_ce(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return float("nan")
        value = value[0]
    try:
        if isinstance(value, str):
            match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
            if match is None:
                return float("nan")
            value = match.group(0)
        result = float(value)
        return result if np.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _arrays_equivalent(left_mzs: Any, left_intensities: Any, right_mzs: Any, right_intensities: Any) -> bool:
    left_mz = np.asarray(left_mzs, dtype=np.float64).reshape(-1)
    left_int = np.asarray(left_intensities, dtype=np.float64).reshape(-1)
    right_mz = np.asarray(right_mzs, dtype=np.float64).reshape(-1)
    right_int = np.asarray(right_intensities, dtype=np.float64).reshape(-1)
    if left_mz.shape != right_mz.shape or left_int.shape != right_int.shape or left_mz.size == 0:
        return False
    return bool(np.allclose(left_mz, right_mz, atol=1e-4, rtol=0.0) and np.allclose(left_int, right_int, atol=1e-4, rtol=0.0))


def _query_collision_energy(query: dict[str, Any]) -> Any:
    if "observed_collision_energy_raw" in query:
        return query["observed_collision_energy_raw"]
    if "collision_energy" in query:
        return query["collision_energy"]
    if "collision_energy_ev" in query:
        return query["collision_energy_ev"]
    return query.get("observed_collision_energy", np.nan)


def _same_or_near_ce(query_ce: Any, row_ce: Any, tolerance: float) -> bool:
    q_ce = _as_ce(query_ce)
    r_ce = _as_ce(row_ce)
    return bool(np.isfinite(q_ce) and np.isfinite(r_ce) and abs(q_ce - r_ce) <= tolerance)


def _row_collision_energy(row: pd.Series) -> Any:
    if "collision_energy" in row:
        return row["collision_energy"]
    if "collision_energy_ev" in row:
        return row["collision_energy_ev"]
    return np.nan


def _row_matches_query(row: pd.Series, query: dict[str, Any], mz_column: str, intensity_column: str) -> bool:
    row_fingerprints = set(_spectrum_fingerprints(row[mz_column], row[intensity_column]))
    raw_query_fingerprints = query.get("query_peak_fingerprints", [])
    if isinstance(raw_query_fingerprints, str):
        query_fingerprints = {raw_query_fingerprints}
    else:
        query_fingerprints = set(raw_query_fingerprints or [])
    if query.get("query_peak_fingerprint"):
        query_fingerprints.add(str(query["query_peak_fingerprint"]))
    if query.get("peak_fingerprint"):
        query_fingerprints.add(str(query["peak_fingerprint"]))
    if row_fingerprints & query_fingerprints:
        return True
    if "peak_fingerprint" in row and str(row["peak_fingerprint"]) in query_fingerprints:
        return True
    if "query_peak_mzs" in query and "query_peak_intensities" in query:
        return _arrays_equivalent(
            row[mz_column],
            row[intensity_column],
            query["query_peak_mzs"],
            query["query_peak_intensities"],
        )
    return False


def _row_key(row: pd.Series, key_column: str) -> str:
    value = row[key_column]
    return "" if pd.isna(value) else str(value)


def purge_reference_rows(
    reference: pd.DataFrame,
    c1_queries: Iterable[dict[str, Any]],
    c2_keys: Iterable[str],
    c3_keys: Iterable[str],
    validation_keys: Iterable[str],
    near_ce_tolerance: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Purge prohibited full keys and C1 query/near-CE rows from a source library."""
    if reference.empty:
        return reference.copy(), {
            "status": "PASSED_EMPTY_REFERENCE",
            "prohibited_full_key_overlap_count": 0,
            "c1_exact_query_matches_remaining": 0,
            "c1_near_ce_matches_remaining": 0,
            "c2_reference_full_key_overlap": 0,
            "c3_reference_full_key_overlap": 0,
            "validation_reference_full_key_overlap": 0,
            "c1_retained_sibling_spectra": 0,
            "purged": {"prohibited_full_key_spectra": 0, "c1_exact_query_spectra": 0, "c1_near_ce_spectra": 0},
            "audit_status": "PASSED_EMPTY_REFERENCE",
            "leakage_audit_status": "PASSED_EMPTY_REFERENCE",
            "c2_target_reference_count": 0,
            "c3_target_reference_count": 0,
            "zero_prohibited_full_key_overlap": True,
        }

    frame = reference.copy().reset_index(drop=True)
    key_column = _key_column(frame)
    smiles_column = "normalized_smiles" if "normalized_smiles" in frame.columns else "canonical_smiles"
    mz_column = "peaks_mz" if "peaks_mz" in frame.columns else "ms2_mzs"
    intensity_column = "peaks_intensity" if "peaks_intensity" in frame.columns else "ms2_intensities"
    if smiles_column not in frame.columns or mz_column not in frame.columns or intensity_column not in frame.columns:
        raise KeyError("Reference library lacks structure or peak columns")

    c1_list = list(c1_queries)
    c2_set = {str(key) for key in c2_keys if key}
    c3_set = {str(key) for key in c3_keys if key}
    validation_set = {str(key) for key in validation_keys if key}
    prohibited = c2_set | c3_set | validation_set
    c1_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in c1_list:
        key = str(query.get("true_inchikey", query.get("full_inchikey", query.get("inchikey", ""))))
        if key:
            c1_by_key[key].append(query)

    keys = frame[key_column].astype(str).to_numpy()
    keep = ~np.isin(keys, list(prohibited))
    prohibited_mask = ~keep
    exact_mask = np.zeros(len(frame), dtype=bool)
    near_mask = np.zeros(len(frame), dtype=bool)

    for key, queries in c1_by_key.items():
        indices = np.flatnonzero(keys == key)
        for index in indices:
            row = frame.iloc[index]
            if any(_row_matches_query(row, query, mz_column, intensity_column) for query in queries):
                exact_mask[index] = True
                keep[index] = False
                continue
            if any(_same_or_near_ce(_query_collision_energy(query), _row_collision_energy(row), near_ce_tolerance) for query in queries):
                near_mask[index] = True
                keep[index] = False

    filtered = frame.loc[keep].reset_index(drop=True)
    filtered_keys = set(filtered[key_column].astype(str))
    exact_remaining = 0
    near_remaining = 0
    for query in c1_list:
        key = str(query.get("true_inchikey", query.get("full_inchikey", query.get("inchikey", ""))))
        for index in np.flatnonzero(keys == key):
            if not keep[index] or index >= len(frame):
                continue
            row = frame.iloc[index]
            if _row_matches_query(row, query, mz_column, intensity_column):
                exact_remaining += 1
            if _same_or_near_ce(_query_collision_energy(query), _row_collision_energy(row), near_ce_tolerance):
                near_remaining += 1
    c1_retained = int(np.isin(filtered[key_column].astype(str).to_numpy(), list(c1_by_key)).sum()) if c1_by_key else 0
    audit = {
        "status": "PASSED_STRICT_FULL_KEY_AUDIT",
        "prohibited_full_key_overlap_count": int(len(prohibited & filtered_keys)),
        "c2_reference_full_key_overlap": int(len(c2_set & filtered_keys)),
        "c3_reference_full_key_overlap": int(len(c3_set & filtered_keys)),
        "validation_reference_full_key_overlap": int(len(validation_set & filtered_keys)),
        "c1_exact_query_matches_remaining": int(exact_remaining),
        "c1_near_ce_matches_remaining": int(near_remaining),
        "c1_sibling_full_key_overlap": int(len(set(c1_by_key) & filtered_keys)),
        "c1_retained_sibling_spectra": c1_retained,
        "purged": {
            "prohibited_full_key_spectra": int(prohibited_mask.sum()),
            "c1_exact_query_spectra": int(exact_mask.sum()),
            "c1_near_ce_spectra": int(near_mask.sum()),
        },
        "near_ce_tolerance_ev": float(near_ce_tolerance),
        "molecule_key": "inchikey",
    }
    if audit["prohibited_full_key_overlap_count"] or audit["c1_exact_query_matches_remaining"] or audit["c1_near_ce_matches_remaining"]:
        audit["status"] = "FAILED_LEAKAGE_AUDIT"
    audit["audit_status"] = audit["status"]
    audit["leakage_audit_status"] = audit["status"]
    audit["c2_target_reference_count"] = audit["c2_reference_full_key_overlap"]
    audit["c3_target_reference_count"] = audit["c3_reference_full_key_overlap"]
    audit["zero_prohibited_full_key_overlap"] = audit["prohibited_full_key_overlap_count"] == 0
    return filtered, audit


def _load_candidate_catalog(path: Path) -> tuple[pd.DataFrame, bool]:
    frame = pq.read_table(path).to_pandas()
    if "canonical_smiles" not in frame.columns or "exact_mass" not in frame.columns:
        raise KeyError(f"Candidate catalog is missing canonical_smiles/exact_mass: {path}")
    smiles = frame["canonical_smiles"].astype(str).tolist()
    initial: dict[str, str] = {}
    key_column = None
    for name in ("inchikey", "full_inchikey"):
        if name in frame.columns:
            key_column = name
            for smi, value in zip(smiles, frame[name].tolist()):
                if is_valid_full_inchikey(value):
                    initial[smi] = str(value)
            break
    key_map = _key_map_for_smiles(smiles, initial)
    derived = key_column is None
    if key_column is not None:
        derived = any(not is_valid_full_inchikey(value) for value in frame[key_column].tolist())
    frame["inchikey"] = [key_map[smi] for smi in smiles]
    frame["exact_mass"] = pd.to_numeric(frame["exact_mass"], errors="coerce")
    if frame["exact_mass"].isna().any() or not frame["exact_mass"].is_monotonic_increasing:
        raise ValueError("Candidate catalog must have finite, monotonically sorted exact_mass")
    if not all(is_valid_full_inchikey(value) for value in frame["inchikey"]):
        raise ValueError("Candidate catalog contains an invalid full InChIKey")
    return frame, bool(derived)


def _mass_is_retrievable(
    observed_mass: float,
    precursor_mz: float,
    candidate_masses: Iterable[float],
) -> bool:
    if not np.isfinite(observed_mass) or observed_mass <= 0:
        return False
    masses = np.asarray(candidate_masses, dtype=np.float64)
    if masses.size == 0:
        return False
    nominal = np.isfinite(precursor_mz) and abs(precursor_mz - round(precursor_mz)) <= 0.01
    deltas = np.abs(masses - observed_mass)
    if np.any(deltas <= observed_mass * 100.0 / 1e6):
        return True
    if np.any(np.abs(deltas - C13_DIFF) <= 0.05):
        return True
    return bool(nominal and np.any(deltas <= RETRIEVAL_PARAMETERS["nominal_tol"]))


def _retrieval_contains_key(
    observed_mass: float,
    precursor_mz: float,
    candidate_masses: np.ndarray,
    candidate_keys: np.ndarray,
    target_key: str,
) -> bool:
    indices = retrieve_candidates_union(
        observed_mass,
        candidate_masses,
        ppm_windows=tuple(RETRIEVAL_PARAMETERS["ppm_windows"]),
        use_c13_isotopes=RETRIEVAL_PARAMETERS["use_c13_isotopes"],
        c13_ppm=RETRIEVAL_PARAMETERS["c13_ppm"],
        precursor_mz=precursor_mz,
        nominal_tol=RETRIEVAL_PARAMETERS["nominal_tol"],
    )
    return bool(indices.size and np.any(candidate_keys[indices] == target_key))


def _observed_mass(precursor_mz: Any, adduct: Any) -> float:
    try:
        value = neutral_mass(float(precursor_mz), str(adduct) if adduct is not None else None)
        return float(value) if value is not None and np.isfinite(value) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _load_train_metadata(
    train_path: Path,
    candidate_smiles_to_key: dict[str, str],
    candidate_masses: np.ndarray,
    candidate_keys: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, list[int]], set[str], int]:
    available = set(pq.ParquetFile(train_path).schema_arrow.names)
    required = ["normalized_smiles", "precursor_mz", "adduct", "collision_energy_ev"]
    missing = [name for name in required if name not in available]
    if missing:
        raise KeyError(f"Train metadata is missing columns: {missing}")
    optional = [name for name in ("ionization_mode", "num_peaks") if name in available]
    frame = pq.read_table(train_path, columns=required + optional).to_pandas()
    key_map = _key_map_for_smiles(frame["normalized_smiles"].tolist(), candidate_smiles_to_key)
    frame["_full_inchikey"] = [key_map[str(smi)] for smi in frame["normalized_smiles"].tolist()]
    frame["_observed_neutral_mass"] = np.asarray(
        [_observed_mass(precursor, adduct) for precursor, adduct in zip(frame["precursor_mz"], frame["adduct"])],
        dtype=np.float64,
    )
    if "num_peaks" in frame.columns:
        enough_peaks = pd.to_numeric(frame["num_peaks"], errors="coerce").fillna(0).to_numpy() >= 3
    else:
        enough_peaks = np.ones(len(frame), dtype=bool)
    valid = np.asarray([
        bool(
            np.isfinite(mass)
            and mass > 0
            and _retrieval_contains_key(mass, float(precursor), candidate_masses, candidate_keys, str(key))
        )
        for key, mass, precursor in zip(frame["_full_inchikey"], frame["_observed_neutral_mass"], frame["precursor_mz"])
    ], dtype=bool)
    valid &= enough_peaks
    groups: dict[str, list[int]] = defaultdict(list)
    key_values = frame["_full_inchikey"].astype(str).to_numpy()
    for index in np.flatnonzero(valid):
        groups[str(key_values[index])].append(int(index))
    return frame, groups, set(frame["_full_inchikey"].astype(str)), len(key_map)


def _read_train_rows(train_path: Path, row_indices: Iterable[int], columns: list[str]) -> pd.DataFrame:
    indices = sorted({int(index) for index in row_indices})
    if not indices:
        return pd.DataFrame(columns=columns + ["_row_index"])
    parquet_file = pq.ParquetFile(train_path)
    available = set(parquet_file.schema_arrow.names)
    read_columns = [column for column in columns if column in available]
    tables = []
    start = 0
    for row_group in range(parquet_file.num_row_groups):
        length = parquet_file.metadata.row_group(row_group).num_rows
        end = start + length
        local = [index - start for index in indices if start <= index < end]
        if local:
            table = parquet_file.read_row_group(row_group, columns=read_columns)
            table = table.take(pa.array(local, type=pa.int64()))
            table = table.append_column("_row_index", pa.array([index for index in indices if start <= index < end], type=pa.int64()))
            tables.append(table)
        start = end
    if not tables:
        return pd.DataFrame(columns=read_columns + ["_row_index"])
    return pa.concat_tables(tables).to_pandas()


def _choose_c1_pairs(groups: dict[str, list[int]], rng: random.Random, limit: int) -> list[dict[str, Any]]:
    keys = list(groups)
    rng.shuffle(keys)
    pairs: list[dict[str, Any]] = []
    for key in keys:
        rows = list(groups[key])
        if len(rows) < 2:
            continue
        rng.shuffle(rows)
        chosen = None
        for query_index in rows:
            siblings = [index for index in rows if index != query_index]
            chosen = (query_index, siblings[0])
            break
        if chosen is None:
            continue
        query_index, sibling_index = chosen
        pairs.append({
            "true_inchikey": key,
            "query_row_index": query_index,
            "sibling_row_indices": [index for index in rows if index != query_index],
            "verified_sibling_row_index": sibling_index,
        })
        if len(pairs) >= limit:
            break
    return pairs


def _source_has_c1_sibling(
    source_by_key: dict[str, list[int]],
    source: pd.DataFrame,
    query: dict[str, Any],
    query_mzs: Any,
    query_intensities: Any,
    query_ce: float,
    near_ce_tolerance: float,
    mz_column: str,
    intensity_column: str,
) -> bool:
    key = str(query["true_inchikey"])
    for index in source_by_key.get(key, []):
        row = source.iloc[index]
        if _row_matches_query(row, {**query, "query_peak_mzs": query_mzs, "query_peak_intensities": query_intensities}, mz_column, intensity_column):
            continue
        if _same_or_near_ce(query_ce, _row_collision_energy(row), near_ce_tolerance):
            continue
        if np.isfinite(query_ce) and not np.isfinite(_as_ce(_row_collision_energy(row))):
            continue
        return True
    return False


def _index_keyed_reference(
    path: Path,
    candidate_smiles_to_key: dict[str, str],
) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    source_names = list(parquet_file.schema_arrow.names)
    smiles_column = "canonical_smiles" if "canonical_smiles" in source_names else "normalized_smiles"
    mz_column = "peaks_mz" if "peaks_mz" in source_names else "ms2_mzs"
    intensity_column = "peaks_intensity" if "peaks_intensity" in source_names else "ms2_intensities"
    if smiles_column not in source_names or mz_column not in source_names or intensity_column not in source_names:
        raise KeyError(f"Unified reference lacks structure or peak columns: {path}")
    metadata_names = []
    for name in (smiles_column, "neutral_mass", "precursor_mz", "collision_energy", "n_supporting_spectra", "source_count", "source_library"):
        if name in source_names and name not in metadata_names:
            metadata_names.append(name)
    source_name_to_index = {name: index for index, name in enumerate(source_names)}
    metadata_name_to_index = {name: metadata_names.index(name) for name in metadata_names}
    n_rows = parquet_file.metadata.num_rows
    source_keys = np.empty(n_rows, dtype=object)
    source_ce = np.full(n_rows, np.nan, dtype=np.float64)
    key_cache = candidate_smiles_to_key
    offset = 0
    last_progress = 0
    print(f"Indexing unified reference metadata: {n_rows:,} rows (batches of {STREAM_BATCH_SIZE:,})...", flush=True)
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(columns=metadata_names, batch_size=STREAM_BATCH_SIZE, use_threads=True)
    ):
        n_batch = batch.num_rows
        smiles_values = batch.column(metadata_name_to_index[smiles_column]).to_pylist()
        ce_values = batch.column(metadata_name_to_index["collision_energy"]).to_pylist() if "collision_energy" in metadata_name_to_index else [np.nan] * n_batch
        for local_index in range(n_batch):
            smiles = str(smiles_values[local_index])
            key = key_cache.get(smiles, "")
            if not is_valid_full_inchikey(key):
                key = derive_full_inchikey(smiles)
                if key:
                    key_cache[smiles] = key
            if not is_valid_full_inchikey(key):
                raise ValueError(f"RDKit failed to derive a full InChIKey for unified reference structure {smiles!r}")
            row_index = offset + local_index
            source_keys[row_index] = key
            source_ce[row_index] = _as_ce(ce_values[local_index])
        offset += n_batch
        if offset - last_progress >= STREAM_PROGRESS_ROWS or offset == n_rows:
            print(f"  reference metadata: {offset:,}/{n_rows:,} rows", flush=True)
            last_progress = offset
        del batch, smiles_values, ce_values
        if batch_number % 64 == 0:
            gc.collect()
    if offset != n_rows:
        raise RuntimeError(f"Unified reference metadata row count mismatch: {offset} != {n_rows}")
    return {
        "path": path,
        "parquet_schema": parquet_file.schema_arrow,
        "source_names": source_names,
        "source_name_to_index": source_name_to_index,
        "smiles_column": smiles_column,
        "mz_column": mz_column,
        "intensity_column": intensity_column,
        "source_keys": source_keys,
        "source_ce": source_ce,
        "n_rows": n_rows,
    }


def _build_c1_query_matchers(
    pairs: list[dict[str, Any]],
    prelim_rows: pd.DataFrame,
    train: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    matchers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        row_index = int(pair["query_row_index"])
        if row_index not in prelim_rows.index:
            continue
        raw = prelim_rows.loc[row_index]
        mzs = np.asarray(raw["ms2_mzs"], dtype=np.float32)
        intensities = np.asarray(raw["ms2_normalized_intensities"], dtype=np.float32)
        clean_mzs, clean_intensities = preprocess_spectrum(mzs, intensities, max_peaks=256, deisotope=True)
        matchers[str(pair["true_inchikey"])].append({
            "query_peak_fingerprints": sorted(set(_spectrum_fingerprints(mzs, intensities) + _spectrum_fingerprints(clean_mzs, clean_intensities))),
            "query_peak_mzs": mzs,
            "query_peak_intensities": intensities,
            "observed_collision_energy_raw": _as_ce(train.iloc[row_index]["collision_energy_ev"]),
        })
    return matchers


def _fingerprint_match_arrays(mzs: Any, intensities: Any, query: dict[str, Any]) -> bool:
    row_fingerprints = set(_spectrum_fingerprints(mzs, intensities))
    query_fingerprints = set(query.get("query_peak_fingerprints", []))
    if row_fingerprints & query_fingerprints:
        return True
    return _arrays_equivalent(
        mzs,
        intensities,
        query.get("query_peak_mzs", []),
        query.get("query_peak_intensities", []),
    )


def _scan_c1_reference_peaks(
    reference_index: dict[str, Any],
    query_matchers: dict[str, list[dict[str, Any]]],
    near_ce_tolerance: float,
) -> tuple[dict[str, list[int]], dict[str, set[int]], dict[str, set[int]], set[str]]:
    parquet_file = pq.ParquetFile(reference_index["path"])
    source_keys = reference_index["source_keys"]
    source_ce = reference_index["source_ce"]
    source_indices: dict[str, list[int]] = defaultdict(list)
    exact_rows: dict[str, set[int]] = defaultdict(set)
    near_rows: dict[str, set[int]] = defaultdict(set)
    sibling_keys: set[str] = set()
    peak_columns = [reference_index["mz_column"], reference_index["intensity_column"]]
    peak_name_to_index = {name: index for index, name in enumerate(peak_columns)}
    offset = 0
    last_progress = 0
    total_rows = reference_index["n_rows"]
    print(f"Scanning unified reference peaks for C1 fingerprints: {total_rows:,} rows...", flush=True)
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(columns=peak_columns, batch_size=STREAM_BATCH_SIZE, use_threads=True)
    ):
        n_batch = batch.num_rows
        for local_index in range(n_batch):
            row_index = offset + local_index
            key = str(source_keys[row_index])
            queries = query_matchers.get(key)
            if not queries:
                continue
            source_indices[key].append(row_index)
            mzs = batch.column(peak_name_to_index[reference_index["mz_column"]])[local_index].as_py()
            intensities = batch.column(peak_name_to_index[reference_index["intensity_column"]])[local_index].as_py()
            for query in queries:
                is_exact = _fingerprint_match_arrays(mzs, intensities, query)
                is_near = _same_or_near_ce(query.get("observed_collision_energy_raw"), source_ce[row_index], near_ce_tolerance)
                if is_exact:
                    exact_rows[key].add(row_index)
                elif is_near:
                    near_rows[key].add(row_index)
                elif not (
                    np.isfinite(_as_ce(query.get("observed_collision_energy_raw")))
                    and not np.isfinite(source_ce[row_index])
                ):
                    sibling_keys.add(key)
            del mzs, intensities
        offset += n_batch
        if offset - last_progress >= STREAM_PROGRESS_ROWS or offset == total_rows:
            print(f"  C1 peak scan: {offset:,}/{total_rows:,} rows", flush=True)
            last_progress = offset
        del batch
        if batch_number % 64 == 0:
            gc.collect()
    if offset != total_rows:
        raise RuntimeError(f"Unified reference peak row count mismatch: {offset} != {total_rows}")
    return source_indices, exact_rows, near_rows, sibling_keys


def _prepare_reference_drop_state(
    reference_index: dict[str, Any],
    c1_keys: set[str],
    c2_keys: set[str],
    c3_keys: set[str],
    validation_keys: set[str],
    source_indices: dict[str, list[int]],
    exact_rows: dict[str, set[int]],
    near_rows: dict[str, set[int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    source_keys = reference_index["source_keys"]
    prohibited = c2_keys | c3_keys | validation_keys
    drop_mask = np.isin(source_keys, list(prohibited))
    reason = np.zeros(reference_index["n_rows"], dtype=np.uint8)
    reason[drop_mask] = 1
    for key in c1_keys:
        for row_index in source_indices.get(key, []):
            if row_index in exact_rows.get(key, set()):
                reason[row_index] = 3
            elif row_index in near_rows.get(key, set()):
                reason[row_index] = 2
    drop_mask = reason != 0
    exact_count = int((reason == 3).sum())
    near_count = int((reason == 2).sum())
    audit = {
        "status": "PASSED_STRICT_FULL_KEY_AUDIT",
        "prohibited_full_key_overlap_count": 0,
        "c2_reference_full_key_overlap": 0,
        "c3_reference_full_key_overlap": 0,
        "validation_reference_full_key_overlap": 0,
        "c1_exact_query_matches_remaining": 0,
        "c1_near_ce_matches_remaining": 0,
        "c1_sibling_full_key_overlap": 0,
        "c1_retained_sibling_spectra": 0,
        "purged": {
            "prohibited_full_key_spectra": int((reason == 1).sum()),
            "c1_exact_query_spectra": exact_count,
            "c1_near_ce_spectra": near_count,
        },
        "near_ce_tolerance_ev": 5.0,
        "molecule_key": "inchikey",
    }
    audit["audit_status"] = audit["status"]
    audit["leakage_audit_status"] = audit["status"]
    audit["c2_target_reference_count"] = 0
    audit["c3_target_reference_count"] = 0
    audit["zero_prohibited_full_key_overlap"] = True
    return drop_mask, audit


def _reference_output_schema(reference_index: dict[str, Any]) -> pa.Schema:
    source_schema = reference_index["parquet_schema"]
    source_names = reference_index["source_names"]
    smiles_column = reference_index["smiles_column"]
    excluded = {smiles_column, "normalized_smiles", "inchikey"}
    fields = [pa.field("normalized_smiles", source_schema.field(smiles_column).type)]
    for name in source_names:
        if name in excluded:
            continue
        fields.append(source_schema.field(name))
    fields.append(pa.field("inchikey", pa.string()))
    if "source_library" not in source_names:
        fields.append(pa.field("source_library", pa.string()))
    return pa.schema(fields)


def _write_filtered_reference(
    reference_index: dict[str, Any],
    drop_mask: np.ndarray,
    output_path: Path,
    c1_keys: set[str],
    prohibited_keys: set[str],
) -> dict[str, int]:
    parquet_file = pq.ParquetFile(reference_index["path"])
    source_names = reference_index["source_names"]
    source_name_to_index = reference_index["source_name_to_index"]
    output_schema = _reference_output_schema(reference_index)
    source_keys = reference_index["source_keys"]
    total_rows = reference_index["n_rows"]
    offset = 0
    retained = 0
    retained_c1 = 0
    retained_c1_by_key: dict[str, int] = defaultdict(int)
    retained_c1_keys: set[str] = set()
    retained_forbidden = 0
    last_progress = 0
    c1_array = np.fromiter(c1_keys, dtype=object) if c1_keys else np.empty(0, dtype=object)
    prohibited_array = np.fromiter(prohibited_keys, dtype=object) if prohibited_keys else np.empty(0, dtype=object)
    print(f"Writing filtered reference with ParquetWriter: {total_rows:,} source rows...", flush=True)
    with pq.ParquetWriter(output_path, output_schema, compression="zstd") as writer:
        for batch_number, batch in enumerate(
            parquet_file.iter_batches(columns=source_names, batch_size=STREAM_BATCH_SIZE, use_threads=True)
        ):
            n_batch = batch.num_rows
            local_keep = np.flatnonzero(~drop_mask[offset:offset + n_batch])
            if local_keep.size:
                global_indices = local_keep + offset
                filtered = batch.filter(pa.array(~drop_mask[offset:offset + n_batch], type=pa.bool_()))
                arrays = []
                for field in output_schema:
                    if field.name == "normalized_smiles":
                        arrays.append(filtered.column(source_name_to_index[reference_index["smiles_column"]]))
                    elif field.name == "inchikey":
                        arrays.append(pa.array(source_keys[global_indices].tolist(), type=pa.string()))
                    elif field.name == "source_library" and "source_library" not in source_names:
                        arrays.append(pa.array(["unified_reference_library"] * len(global_indices), type=pa.string()))
                    else:
                        arrays.append(filtered.column(source_name_to_index[field.name]))
                output_batch = pa.RecordBatch.from_arrays(arrays, schema=output_schema)
                writer.write_batch(output_batch)
                retained += len(global_indices)
                retained_keys = source_keys[global_indices]
                if c1_array.size:
                    retained_c1 += int(np.isin(retained_keys, c1_array).sum())
                    for key in retained_keys:
                        key_text = str(key)
                        if key_text in c1_keys:
                            retained_c1_by_key[key_text] += 1
                    retained_c1_keys.update(str(key) for key in retained_keys if str(key) in c1_keys)
                if prohibited_array.size:
                    retained_forbidden += int(np.isin(retained_keys, prohibited_array).sum())
                del filtered, arrays, output_batch, retained_keys, global_indices
            offset += n_batch
            if offset - last_progress >= STREAM_PROGRESS_ROWS or offset == total_rows:
                print(f"  filtered reference: {offset:,}/{total_rows:,} scanned, {retained:,} retained", flush=True)
                last_progress = offset
            del batch, local_keep
            if batch_number % 64 == 0:
                gc.collect()
    if offset != total_rows:
        raise RuntimeError(f"Filtered reference row count mismatch: {offset} != {total_rows}")
    return {
        "source_spectra": total_rows,
        "retained_spectra": retained,
        "purged_spectra": total_rows - retained,
        "c1_retained_sibling_spectra": retained_c1,
        "c1_retained_by_key": dict(retained_c1_by_key),
        "c1_sibling_full_key_overlap": len(retained_c1_keys),
        "retained_forbidden_keys": retained_forbidden,
    }


def _select_external_queries(
    external_path: Path,
    candidate_frame: pd.DataFrame,
    candidate_smiles_to_key: dict[str, str],
    train_keys: set[str],
    rng: random.Random,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not external_path.exists():
        return [], {
            "requested": limit,
            "selected": 0,
            "available": False,
            "status": "UNAVAILABLE_EXTERNAL_FILE_MISSING",
            "limitation": "artifacts/external/external_spectra.parquet is unavailable",
            "novel_external_keys": 0,
            "no_candidate_external_keys": 0,
        }
    parquet_file = pq.ParquetFile(external_path)
    source_names = list(parquet_file.schema_arrow.names)
    required_metadata = ["canonical_smiles", "precursor_mz", "precursor_type"]
    missing = [name for name in required_metadata if name not in source_names]
    mz_column = "peaks_mz" if "peaks_mz" in source_names else "ms2_mzs"
    intensity_column = "peaks_intensity" if "peaks_intensity" in source_names else "ms2_intensities"
    if mz_column not in source_names or intensity_column not in source_names:
        missing.extend([name for name in (mz_column, intensity_column) if name not in source_names])
    if missing:
        return [], {
            "requested": limit,
            "selected": 0,
            "available": False,
            "status": "UNAVAILABLE_EXTERNAL_SCHEMA",
            "limitation": f"Missing external columns: {sorted(set(missing))}",
            "novel_external_keys": 0,
            "no_candidate_external_keys": 0,
        }
    optional = [name for name in ("spectrum_id", "source_library", "collision_energy", "ion_mode") if name in source_names]
    metadata_names = required_metadata + optional
    metadata_index = {name: metadata_names.index(name) for name in metadata_names}
    candidate_key_values = candidate_frame["inchikey"].astype(str).to_numpy()
    candidate_mass_values = candidate_frame["exact_mass"].to_numpy(dtype=np.float64)
    candidate_keys = set(candidate_key_values)
    key_cache = candidate_smiles_to_key
    eligible_by_key: dict[str, dict[str, Any]] = {}
    novel_keys: set[str] = set()
    no_candidate_keys: set[str] = set()
    offset = 0
    last_progress = 0
    total_rows = parquet_file.metadata.num_rows
    print(f"Indexing external metadata: {total_rows:,} rows...", flush=True)
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(columns=metadata_names, batch_size=STREAM_BATCH_SIZE, use_threads=True)
    ):
        n_batch = batch.num_rows
        smiles_values = batch.column(metadata_index["canonical_smiles"]).to_pylist()
        precursor_values = batch.column(metadata_index["precursor_mz"]).to_pylist()
        adduct_values = batch.column(metadata_index["precursor_type"]).to_pylist()
        spectrum_ids = batch.column(metadata_index["spectrum_id"]).to_pylist() if "spectrum_id" in metadata_index else [""] * n_batch
        source_libraries = batch.column(metadata_index["source_library"]).to_pylist() if "source_library" in metadata_index else ["external"] * n_batch
        collision_energies = batch.column(metadata_index["collision_energy"]).to_pylist() if "collision_energy" in metadata_index else [np.nan] * n_batch
        ion_modes = batch.column(metadata_index["ion_mode"]).to_pylist() if "ion_mode" in metadata_index else ["positive"] * n_batch
        for local_index in range(n_batch):
            row_index = offset + local_index
            smiles = str(smiles_values[local_index])
            key = key_cache.get(smiles, "")
            if not is_valid_full_inchikey(key):
                key = derive_full_inchikey(smiles)
                if key:
                    key_cache[smiles] = key
            if not is_valid_full_inchikey(key):
                raise ValueError(f"RDKit failed to derive a full InChIKey for external structure {smiles!r}")
            if key not in train_keys:
                novel_keys.add(key)
            if key not in candidate_keys:
                no_candidate_keys.add(key)
                continue
            mass = _observed_mass(precursor_values[local_index], adduct_values[local_index])
            if not _retrieval_contains_key(
                mass,
                float(precursor_values[local_index]),
                candidate_mass_values,
                candidate_key_values,
                key,
            ):
                continue
            if key in eligible_by_key:
                continue
            eligible_by_key[key] = {
                "source_row_index": row_index,
                "spectrum_id": str(spectrum_ids[local_index] or f"external_row_{row_index}"),
                "true_inchikey": key,
                "true_smiles": smiles,
                "precursor_mz": float(precursor_values[local_index]),
                "adduct": str(adduct_values[local_index]),
                "collision_energy": _as_ce(collision_energies[local_index]),
                "ionization_mode": _ionization_mode(ion_modes[local_index]),
                "source_library": str(source_libraries[local_index] or "external"),
            }
        offset += n_batch
        if offset - last_progress >= STREAM_PROGRESS_ROWS or offset == total_rows:
            print(f"  external metadata: {offset:,}/{total_rows:,} rows", flush=True)
            last_progress = offset
        del batch, smiles_values, precursor_values, adduct_values, spectrum_ids, source_libraries, collision_energies, ion_modes
        if batch_number % 64 == 0:
            gc.collect()
    if offset != total_rows:
        raise RuntimeError(f"External metadata row count mismatch: {offset} != {total_rows}")
    eligible = list(eligible_by_key.values())
    rng.shuffle(eligible)
    selected = eligible[:limit]
    if selected:
        selected_by_row = {int(record["source_row_index"]): record for record in selected}
        peak_parquet = pq.ParquetFile(external_path)
        peak_names = [mz_column, intensity_column]
        peak_index = {name: index for index, name in enumerate(peak_names)}
        peak_offset = 0
        for batch_number, batch in enumerate(
            peak_parquet.iter_batches(columns=peak_names, batch_size=STREAM_BATCH_SIZE, use_threads=True)
        ):
            n_batch = batch.num_rows
            mz_values = batch.column(peak_index[mz_column])
            intensity_values = batch.column(peak_index[intensity_column])
            for local_index in range(n_batch):
                record = selected_by_row.get(peak_offset + local_index)
                if record is None:
                    continue
                record["mzs"] = np.asarray(mz_values[local_index].as_py(), dtype=np.float32)
                record["intensities"] = np.asarray(intensity_values[local_index].as_py(), dtype=np.float32)
            peak_offset += n_batch
            del batch, mz_values, intensity_values
            if batch_number % 64 == 0:
                gc.collect()
        if peak_offset != total_rows:
            raise RuntimeError(f"External peak row count mismatch: {peak_offset} != {total_rows}")
    novel_count = len(novel_keys)
    no_candidate_count = len(no_candidate_keys)
    if not selected:
        status = "UNAVAILABLE_NO_EXTERNAL_CANDIDATE_MATCH"
        limitation = "No external spectrum has a full InChIKey absent from train and present in the candidate catalog"
    elif len(selected) < limit:
        status = "LIMITED_AVAILABLE"
        limitation = f"Only {len(selected)} external full-key candidate structures were available; requested {limit}"
    else:
        status = "AVAILABLE"
        limitation = ""
    del eligible_by_key, eligible, key_cache
    gc.collect()
    return selected, {
        "requested": limit,
        "selected": len(selected),
        "available": bool(selected),
        "status": status,
        "limitation": limitation,
        "novel_external_keys": novel_count,
        "no_candidate_external_keys": no_candidate_count,
    }


def _candidate_recall_for_queries(
    query_records: Iterable[dict[str, Any]],
    candidate_masses: np.ndarray,
    candidate_keys: np.ndarray,
) -> tuple[float, dict[str, float], dict[str, int]]:
    hits = defaultdict(int)
    totals = defaultdict(int)
    for record in query_records:
        group = str(record["group"])
        totals[group] += 1
        mass = _observed_mass(record["observed_precursor_mz"], record["observed_adduct"])
        if _retrieval_contains_key(
            mass,
            float(record["observed_precursor_mz"]),
            candidate_masses,
            candidate_keys,
            str(record["true_inchikey"]),
        ):
            hits[group] += 1
    for group in ("C1", "C2", "C3"):
        totals.setdefault(group, 0)
    total_hits = sum(hits.values())
    total_queries = sum(totals.values())
    return (
        float(total_hits / total_queries) if total_queries else 0.0,
        {group: float(hits[group] / totals[group]) if totals[group] else 0.0 for group in totals},
        {group: int(hits[group]) for group in totals},
    )


def _ionization_mode(value: Any) -> str:
    if value is None:
        return "positive"
    try:
        if pd.isna(value):
            return "positive"
    except (TypeError, ValueError):
        pass
    text = str(value).lower()
    return "negative" if "neg" in text else "positive"


def _query_record(
    query_id: str,
    group: str,
    true_inchikey: str,
    true_smiles: str,
    precursor_mz: float,
    adduct: str,
    collision_energy: float,
    ionization_mode: str,
    mzs: Any,
    intensities: Any,
    source: str,
    source_row_index: int,
    source_spectrum_id: str = "",
    sibling_row_indices: Iterable[int] = (),
    verified_sibling_count: int = 0,
) -> dict[str, Any]:
    raw_mzs = np.asarray(mzs, dtype=np.float32).reshape(-1)
    raw_intensities = np.asarray(intensities, dtype=np.float32).reshape(-1)
    clean_mzs, clean_intensities = preprocess_spectrum(raw_mzs, raw_intensities, max_peaks=256, deisotope=True)
    clean_fingerprint = spectrum_fingerprint(clean_mzs, clean_intensities)
    fingerprints = sorted(set(_spectrum_fingerprints(raw_mzs, raw_intensities) + _spectrum_fingerprints(clean_mzs, clean_intensities)))
    known_ce = bool(np.isfinite(collision_energy))
    return {
        "query_id": query_id,
        "group": group,
        "query_source": source,
        "source_row_index": int(source_row_index),
        "source_spectrum_id": str(source_spectrum_id),
        "observed_precursor_mz": float(precursor_mz),
        "observed_adduct": str(adduct),
        "observed_collision_energy": float(collision_energy) if known_ce else 30.0,
        "observed_collision_energy_raw": float(collision_energy) if known_ce else float("nan"),
        "observed_ionization_mode": str(ionization_mode),
        "observed_ms2_mzs": clean_mzs.tolist(),
        "observed_ms2_intensities": clean_intensities.tolist(),
        "true_inchikey": str(true_inchikey),
        "true_smiles": str(true_smiles),
        "query_peak_fingerprint": clean_fingerprint or (fingerprints[0] if fingerprints else ""),
        "query_peak_fingerprints": fingerprints,
        "withheld_exact_query": group == "C1",
        "withheld_row_index": int(source_row_index) if group == "C1" else -1,
        "sibling_row_indices": [int(index) for index in sibling_row_indices],
        "sibling_spectrum_indices": [int(index) for index in sibling_row_indices],
        "verified_sibling_count": int(verified_sibling_count),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def build_clean_v4_benchmark(
    output_dir: str | Path = DEFAULT_OUT_DIR,
    candidate_path: str | Path | None = None,
    train_path: str | Path = ROOT / "dataset" / "train.parquet",
    unified_path: str | Path = ROOT / "kaggle_dataset" / "unified_reference_library.parquet",
    external_path: str | Path = ROOT / "artifacts" / "external" / "external_spectra.parquet",
    n_per_group: int = 150,
    seed: int = 2026,
) -> dict[str, Any]:
    out_dir = _resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = _resolve_path(train_path)
    unified_path = _resolve_path(unified_path)
    external_path = _resolve_path(external_path)
    if candidate_path is None:
        candidate_candidates = [out_dir / "candidate_union.parquet", DEFAULT_OUT_DIR / "candidate_union.parquet"]
        candidate_file = next((path for path in candidate_candidates if path.exists()), None)
    else:
        candidate_file = _resolve_path(candidate_path)
    if candidate_file is None or not candidate_file.exists():
        raise FileNotFoundError("A candidate_union.parquet with full InChIKeys is required")

    started = time.time()
    candidate, candidate_keys_rederived = _load_candidate_catalog(candidate_file)
    candidate_smiles_to_key = dict(zip(candidate["canonical_smiles"].astype(str), candidate["inchikey"].astype(str)))
    candidate_masses = candidate["exact_mass"].to_numpy(dtype=np.float64)
    candidate_keys = candidate["inchikey"].astype(str).to_numpy()
    print(f"Loaded candidate catalog: {candidate_file} ({len(candidate):,} rows, full-key rederived={candidate_keys_rederived})")

    train, train_groups, train_key_set, n_train_structures = _load_train_metadata(
        train_path,
        candidate_smiles_to_key,
        candidate_masses,
        candidate_keys,
    )
    print(f"Train query candidates: {len(train_groups):,} full keys, {n_train_structures:,} structures")

    reference_index = _index_keyed_reference(unified_path, candidate_smiles_to_key)
    print(
        f"Unified reference metadata indexed: {reference_index['n_rows']:,} rows; "
        f"peak columns deferred to bounded scans",
        flush=True,
    )

    rng = random.Random(seed)
    prelim_limit = max(n_per_group * 5, n_per_group + 100)
    prelim_c1 = _choose_c1_pairs(train_groups, rng, prelim_limit)
    prelim_indices = [pair["query_row_index"] for pair in prelim_c1]
    prelim_rows = _read_train_rows(
        train_path,
        prelim_indices,
        ["ms2_mzs", "ms2_normalized_intensities"],
    ).set_index("_row_index")
    query_matchers = _build_c1_query_matchers(prelim_c1, prelim_rows, train)
    source_indices, exact_rows, near_rows, sibling_keys = _scan_c1_reference_peaks(
        reference_index,
        query_matchers,
        5.0,
    )
    verified_c1 = [pair for pair in prelim_c1 if str(pair["true_inchikey"]) in sibling_keys]
    c1_pairs = verified_c1[:n_per_group]
    c1_keys = {pair["true_inchikey"] for pair in c1_pairs}
    print(f"C1 verified train cohorts: {len(c1_pairs):,}", flush=True)
    del prelim_rows, query_matchers
    gc.collect()

    remaining_keys = [key for key in train_groups if key not in c1_keys]
    rng.shuffle(remaining_keys)
    c2_pairs = []
    for key in remaining_keys:
        if len(c2_pairs) >= n_per_group:
            break
        rows = list(train_groups[key])
        if rows:
            rng.shuffle(rows)
            c2_pairs.append({"true_inchikey": key, "query_row_index": rows[0]})
    c2_keys = {pair["true_inchikey"] for pair in c2_pairs}
    remaining_keys = [key for key in remaining_keys if key not in c2_keys]
    validation_keys = set(remaining_keys[:1000])
    train_partition_keys = set(remaining_keys[1000:])
    if c1_keys & c2_keys or c1_keys & validation_keys or c2_keys & validation_keys:
        raise AssertionError("C1, C2, and validation full-key cohorts overlap")
    print(f"C2 train cohorts: {len(c2_pairs):,}; validation keys: {len(validation_keys):,}")

    external_records, c3_stats = _select_external_queries(
        external_path,
        candidate,
        candidate_smiles_to_key,
        train_key_set,
        rng,
        n_per_group,
    )
    c3_keys = {record["true_inchikey"] for record in external_records}
    if c3_keys & train_key_set:
        raise AssertionError("C3 contains a full key present in train")
    if c3_keys & (c1_keys | c2_keys | validation_keys):
        raise AssertionError("C3 full keys overlap train-derived benchmark or validation keys")
    print(f"C3 external cohorts: {len(external_records):,} ({c3_stats['status']})")

    final_train_indices = [pair["query_row_index"] for pair in c1_pairs + c2_pairs]
    train_query_rows = _read_train_rows(
        train_path,
        final_train_indices,
        ["ms2_mzs", "ms2_normalized_intensities"],
    ).set_index("_row_index")
    query_records = []
    c1_query_ids = []
    for number, pair in enumerate(c1_pairs):
        row_index = pair["query_row_index"]
        row = train.iloc[row_index]
        raw = train_query_rows.loc[row_index]
        query_id = f"c1_train_{number:04d}"
        c1_query_ids.append(query_id)
        query_records.append(_query_record(
            query_id,
            "C1",
            pair["true_inchikey"],
            str(row["normalized_smiles"]),
            float(row["precursor_mz"]),
            str(row["adduct"]),
            _as_ce(row["collision_energy_ev"]),
            _ionization_mode(row.get("ionization_mode", "positive")),
            raw["ms2_mzs"],
            raw["ms2_normalized_intensities"],
            "train",
            row_index,
            sibling_row_indices=pair["sibling_row_indices"],
            verified_sibling_count=1,
        ))
    c2_query_ids = []
    for number, pair in enumerate(c2_pairs):
        row_index = pair["query_row_index"]
        row = train.iloc[row_index]
        raw = train_query_rows.loc[row_index]
        query_id = f"c2_train_{number:04d}"
        c2_query_ids.append(query_id)
        query_records.append(_query_record(
            query_id,
            "C2",
            pair["true_inchikey"],
            str(row["normalized_smiles"]),
            float(row["precursor_mz"]),
            str(row["adduct"]),
            _as_ce(row["collision_energy_ev"]),
            _ionization_mode(row.get("ionization_mode", "positive")),
            raw["ms2_mzs"],
            raw["ms2_normalized_intensities"],
            "train",
            row_index,
        ))
    c3_query_ids = []
    for number, record in enumerate(external_records):
        query_id = f"c3_external_{number:04d}"
        c3_query_ids.append(query_id)
        query_records.append(_query_record(
            query_id,
            "C3",
            record["true_inchikey"],
            record["true_smiles"],
            record["precursor_mz"],
            record["adduct"],
            record["collision_energy"],
            record["ionization_mode"],
            record["mzs"],
            record["intensities"],
            "external",
            record["source_row_index"],
            source_spectrum_id=record["spectrum_id"],
        ))

    candidate_recall, candidate_recall_by_group, candidate_recall_hits = _candidate_recall_for_queries(
        query_records,
        candidate_masses,
        candidate_keys,
    )
    if query_records and candidate_recall < 0.99:
        raise AssertionError(f"Selected benchmark candidate recall {candidate_recall:.4f} is below 0.99")
    drop_mask, audit = _prepare_reference_drop_state(
        reference_index,
        c1_keys,
        c2_keys,
        c3_keys,
        validation_keys,
        source_indices,
        exact_rows,
        near_rows,
    )
    query_frame = pd.DataFrame(query_records)
    query_path = out_dir / "benchmark_v4_queries.parquet"
    reference_path = out_dir / "clean_v4_reference_library.parquet"
    query_frame.to_parquet(query_path, index=False, engine="pyarrow", compression="zstd")
    stream_stats = _write_filtered_reference(
        reference_index,
        drop_mask,
        reference_path,
        c1_keys,
        c2_keys | c3_keys | validation_keys,
    )
    audit["c1_retained_sibling_spectra"] = stream_stats["c1_retained_sibling_spectra"]
    audit["c1_sibling_full_key_overlap"] = stream_stats["c1_sibling_full_key_overlap"]
    audit["retained_forbidden_keys"] = stream_stats["retained_forbidden_keys"]
    if stream_stats["retained_forbidden_keys"]:
        audit["status"] = "FAILED_LEAKAGE_AUDIT"
        audit["audit_status"] = audit["status"]
        audit["leakage_audit_status"] = audit["status"]
    if audit["status"] != "PASSED_STRICT_FULL_KEY_AUDIT":
        raise AssertionError(f"Reference leakage audit failed: {audit}")
    if stream_stats["c1_retained_sibling_spectra"] == 0 and c1_keys:
        raise AssertionError("C1 queries have no retained sibling spectra")
    retained_c1_by_key = stream_stats["c1_retained_by_key"]
    for record in query_records:
        if record["group"] == "C1":
            record["verified_sibling_count"] = int(retained_c1_by_key.get(record["true_inchikey"], 0))
    c1_without_siblings = [
        record["query_id"] for record in query_records
        if record["group"] == "C1" and record["verified_sibling_count"] == 0
    ]
    if c1_without_siblings:
        raise AssertionError(f"C1 queries without a retained sibling: {c1_without_siblings[:5]}")
    query_frame = pd.DataFrame(query_records)
    query_frame.to_parquet(query_path, index=False, engine="pyarrow", compression="zstd")


    query_sha = _sha256_file(query_path)
    reference_sha = _sha256_file(reference_path)
    candidate_sha = _sha256_file(candidate_file)
    candidate_manifest_path = candidate_file.parent / "candidate_manifest.json"
    candidate_manifest_sha = _sha256_file(candidate_manifest_path) if candidate_manifest_path.exists() else ""
    unified_sha = _sha256_file(unified_path)
    external_sha = _sha256_file(external_path) if external_path.exists() else ""
    split_path = out_dir / "clean_v4_split.json"
    query_id_sets = {
        "C1": c1_query_ids,
        "C2": c2_query_ids,
        "C3": c3_query_ids,
    }
    full_key_sets = {
        "C1": sorted(c1_keys),
        "C2": sorted(c2_keys),
        "C3": sorted(c3_keys),
        "validation": sorted(validation_keys),
        "train": sorted(train_key_set),
        "train_partition": sorted(train_partition_keys),
    }
    split_payload = {
        "version": "v4_clean.1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "molecule_key": "inchikey",
        "counts": {
            "queries": int(len(query_frame)),
            "C1": len(c1_query_ids),
            "C2": len(c2_query_ids),
            "C3": len(c3_query_ids),
            "validation_full_keys": len(validation_keys),
            "all_train_full_keys": len(train_key_set),
            "train_partition_full_keys": len(train_partition_keys),
        },
        "query_ids": query_id_sets,
        "full_key_sets": full_key_sets,
        "c3_selection": c3_stats,
        "c3_limitation": c3_stats["limitation"],
        "candidate_recall": {
            "overall": candidate_recall,
            "by_group": candidate_recall_by_group,
            "hits_by_group": candidate_recall_hits,
        },
        "candidate_catalog": {
            "path": str(candidate_file),
            "sha256": candidate_sha,
            "manifest_sha256": candidate_manifest_sha,
            "full_keys_rederived_for_benchmark": candidate_keys_rederived,
        },
        "source_hashes": {
            "unified_reference_library": unified_sha,
            "external_spectra": external_sha,
        },
        "artifacts": {
            "query_file": query_path.name,
            "query_sha256": query_sha,
            "reference_file": reference_path.name,
            "reference_sha256": reference_sha,
        },
        "retrieval_parameters": RETRIEVAL_PARAMETERS,
    }
    _write_json(split_path, split_payload)
    split_sha = _sha256_file(split_path)

    manifest_payload = {
        "manifest_version": "v4_clean.1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "molecule_key": "inchikey",
        "source_library": str(unified_path),
        "source_library_sha256": unified_sha,
        "candidate_catalog": str(candidate_file),
        "candidate_catalog_sha256": candidate_sha,
        "candidate_manifest_sha256": candidate_manifest_sha,
        "split_file": split_path.name,
        "split_sha256": split_sha,
        "query_file": query_path.name,
        "query_sha256": query_sha,
        "reference_file": reference_path.name,
        "reference_sha256": reference_sha,
        "counts": {
            "source_spectra": int(stream_stats["source_spectra"]),
            "retained_spectra": int(stream_stats["retained_spectra"]),
            "purged_spectra": int(stream_stats["purged_spectra"]),
            "C1_queries": len(c1_query_ids),
            "C2_queries": len(c2_query_ids),
            "C3_queries": len(c3_query_ids),
            "validation_full_keys": len(validation_keys),
            "all_train_full_keys": len(train_key_set),
            "train_partition_full_keys": len(train_partition_keys),
        },
        "query_ids": query_id_sets,
        "full_key_sets": {
            "C1": sorted(c1_keys),
            "C2": sorted(c2_keys),
            "C3": sorted(c3_keys),
            "validation": sorted(validation_keys),
            "all_train": sorted(train_key_set),
            "train_partition": sorted(train_partition_keys),
        },
        "audit": audit,
        "audit_status": audit["status"],
        "leakage_audit_status": audit["status"],
        "c3_selection": c3_stats,
        "c3_limitation": c3_stats["limitation"],
        "candidate_recall": {
            "overall": candidate_recall,
            "by_group": candidate_recall_by_group,
            "hits_by_group": candidate_recall_hits,
        },
        "retrieval_parameters": RETRIEVAL_PARAMETERS,
    }
    manifest_path = out_dir / "clean_v4_reference_manifest.json"
    _write_json(manifest_path, manifest_payload)
    _write_json(out_dir / "clean_v4_reference_library_metadata.json", manifest_payload)

    del train
    del reference_index, drop_mask, source_indices, exact_rows, near_rows, sibling_keys
    gc.collect()
    print(f"Saved {query_path}")
    print(f"Saved {reference_path}")
    print(f"Split SHA-256: {split_sha}")
    print(f"Reference manifest: {manifest_path}")
    print(f"Audit: {audit['status']}")
    print(f"Completed in {time.time() - started:.1f}s")
    return {
        "output_dir": str(out_dir),
        "split_path": str(split_path),
        "reference_path": str(reference_path),
        "query_path": str(query_path),
        "manifest_path": str(manifest_path),
        "audit": audit,
        "c3_selection": c3_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--candidate-path", dest="candidate_path", default=None)
    parser.add_argument("--train-path", dest="train_path", default=str(ROOT / "dataset" / "train.parquet"))
    parser.add_argument("--unified-path", dest="unified_path", default=str(ROOT / "kaggle_dataset" / "unified_reference_library.parquet"))
    parser.add_argument("--external-path", dest="external_path", default=str(ROOT / "artifacts" / "external" / "external_spectra.parquet"))
    parser.add_argument("--n-per-group", dest="n_per_group", type=int, default=150)
    parser.add_argument("--seed", dest="seed", type=int, default=2026)
    args = parser.parse_args()
    build_clean_v4_benchmark(
        output_dir=args.output_dir,
        candidate_path=args.candidate_path,
        train_path=args.train_path,
        unified_path=args.unified_path,
        external_path=args.external_path,
        n_per_group=args.n_per_group,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
