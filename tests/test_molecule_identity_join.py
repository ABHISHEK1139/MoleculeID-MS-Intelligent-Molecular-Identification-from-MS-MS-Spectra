"""Tests for full InChIKey identity joins and repaired reference purging."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from scripts.build_clean_v4_benchmark import (
    derive_full_inchikey,
    purge_reference_rows,
    spectrum_fingerprint,
)
from src.core.preprocessing_v3 import fast_mutual_cosine


@pytest.fixture(scope="module")
def clean_ref_sample():
    candidates = [
        ROOT / "artifacts" / "v4_clean" / "clean_v4_reference_library.parquet",
        ROOT / "artifacts" / "v3_clean" / "clean_reference_library.parquet",
    ]
    ref_path = next((path for path in candidates if path.exists()), None)
    assert ref_path is not None, f"Reference library missing at {candidates[0]}"
    frame = pq.read_table(ref_path).to_pandas().head(100)
    if "normalized_smiles" not in frame.columns:
        frame["normalized_smiles"] = frame["canonical_smiles"]
    if "inchikey" not in frame.columns:
        frame["inchikey"] = [derive_full_inchikey(smiles) for smiles in frame["normalized_smiles"]]
    assert frame["normalized_smiles"].map(derive_full_inchikey).eq(frame["inchikey"]).all()
    return frame


def test_known_molecule_direct_score(clean_ref_sample):
    row = clean_ref_sample.iloc[0]
    true_key = str(row["inchikey"])
    assert len(true_key) == 27

    q_mzs = np.asarray(row["peaks_mz"], dtype=np.float32)
    q_ints = np.asarray(row["peaks_intensity"], dtype=np.float32)
    q_prec = float(row["precursor_mz"])
    q_neutral = float(row["neutral_mass"])

    lib_keys = clean_ref_sample["inchikey"].astype(str).to_numpy()
    lib_masses = clean_ref_sample["neutral_mass"].to_numpy(dtype=np.float64)
    lib_precs = clean_ref_sample["precursor_mz"].to_numpy(dtype=np.float64)
    ppm_tol = q_neutral * 20.0 / 1e6
    cand_indices = np.where(np.abs(lib_masses - q_neutral) <= ppm_tol)[0]

    lib_hits = {}
    for ci in cand_indices:
        ref_key = lib_keys[ci]
        cos_sim, _ = fast_mutual_cosine(
            q_mzs,
            q_ints,
            np.asarray(clean_ref_sample["peaks_mz"].iloc[ci], dtype=np.float32),
            np.asarray(clean_ref_sample["peaks_intensity"].iloc[ci], dtype=np.float32),
            delta=q_prec - lib_precs[ci],
        )
        if cos_sim >= 0.10:
            if ref_key not in lib_hits or cos_sim > lib_hits[ref_key]:
                lib_hits[ref_key] = float(cos_sim)

    assert true_key in lib_hits
    assert lib_hits[true_key] > 0.0
    assert lib_hits[true_key] >= 0.99


def test_known_analog_score_attaches(clean_ref_sample):
    row_target = clean_ref_sample.iloc[0]
    target_key = str(row_target["inchikey"])
    target_mass = float(row_target["neutral_mass"])
    diffs = np.abs(clean_ref_sample["neutral_mass"] - target_mass)
    analog_candidates = clean_ref_sample[
        (clean_ref_sample["inchikey"].astype(str) != target_key)
        & (diffs < 200.0)
        & (diffs > 1.0)
    ]
    assert len(analog_candidates) > 0
    analog_key = str(analog_candidates.iloc[0]["inchikey"])
    pool_keys = [target_key, analog_key]
    pool_k2i = {key: index for index, key in enumerate(pool_keys)}
    simulated_analog_hits = [(analog_key, 0.75)]
    attached_ids = []
    attached_sims = []
    for key, score in simulated_analog_hits:
        index = pool_k2i.get(key, -1)
        if index >= 0:
            attached_ids.append(index)
            attached_sims.append(score)
    assert attached_ids == [pool_k2i[analog_key]]
    assert attached_sims == [0.75]


def test_unknown_full_key_no_accidental_match(clean_ref_sample):
    unknown_key = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
    library_keys = set(clean_ref_sample["inchikey"].astype(str))
    assert unknown_key not in library_keys
    direct_hits = {key: 0.85 for key in list(library_keys)[:5]}
    assert direct_hits.get(unknown_key, 0.0) == 0.0
    pool_k2i = {key: index for index, key in enumerate(clean_ref_sample["inchikey"].astype(str))}
    assert pool_k2i.get(unknown_key, -1) == -1


def test_full_key_join_does_not_use_inchikey14():
    left_smiles = "F/C=C/F"
    right_smiles = "F/C=C\\F"
    left_key = derive_full_inchikey(left_smiles)
    right_key = derive_full_inchikey(right_smiles)
    assert left_key[:14] == right_key[:14]
    assert left_key != right_key
    reference_keys = {left_key}
    assert right_key not in reference_keys
    assert left_key in reference_keys


def _synthetic_reference(keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "normalized_smiles": ["C"] * len(keys),
        "inchikey": keys,
        "neutral_mass": np.arange(100.0, 100.0 + len(keys)),
        "precursor_mz": np.arange(101.0, 101.0 + len(keys)),
        "collision_energy": [20.0] * len(keys),
        "peaks_mz": pd.Series([[100.0, 200.0] for _ in keys], dtype=object),
        "peaks_intensity": pd.Series([[0.5, 0.5] for _ in keys], dtype=object),
    })


def test_c2_c3_purge_uses_full_keys():
    c2_key = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
    c3_key = "AAAAAAAAAAAAAA-CCCCCCCCCC-D"
    validation_key = "AAAAAAAAAAAAAA-DDDDDDDDDD-E"
    retained_key = "AAAAAAAAAAAAAA-EEEEEEEEEE-F"
    reference = _synthetic_reference([c2_key, c3_key, validation_key, retained_key])
    filtered, audit = purge_reference_rows(reference, [], [c2_key], [c3_key], [validation_key])
    assert set(filtered["inchikey"]) == {retained_key}
    assert audit["c2_reference_full_key_overlap"] == 0
    assert audit["c3_reference_full_key_overlap"] == 0
    assert audit["validation_reference_full_key_overlap"] == 0
    assert audit["prohibited_full_key_overlap_count"] == 0
    assert audit["status"] == "PASSED_STRICT_FULL_KEY_AUDIT"


def test_c1_exact_query_purge_preserves_legitimate_sibling():
    key = "AAAAAAAAAAAAAA-BBBBBBBBBB-C"
    other_key = "AAAAAAAAAAAAAA-CCCCCCCCCC-D"
    query_mzs = np.asarray([100.0, 200.0], dtype=np.float32)
    query_intensities = np.asarray([0.5, 0.5], dtype=np.float32)
    reference = _synthetic_reference([key, key, key, other_key])
    reference["peaks_mz"] = [
        [100.0, 200.0],
        [110.0, 210.0],
        [120.0, 220.0],
        [100.0, 200.0],
    ]
    reference["peaks_intensity"] = [
        [0.5, 0.5],
        [0.4, 0.6],
        [0.3, 0.7],
        [0.5, 0.5],
    ]
    reference.loc[1, "collision_energy"] = 24.0
    reference.loc[2, "collision_energy"] = 40.0
    query = {
        "true_inchikey": key,
        "query_peak_fingerprints": [spectrum_fingerprint(query_mzs, query_intensities)],
        "query_peak_mzs": query_mzs,
        "query_peak_intensities": query_intensities,
        "observed_collision_energy_raw": 20.0,
    }
    filtered, audit = purge_reference_rows(reference, [query], [], [], [])
    assert len(filtered) == 2
    assert filtered["inchikey"].tolist().count(key) == 1
    assert audit["purged"]["c1_exact_query_spectra"] == 1
    assert audit["purged"]["c1_near_ce_spectra"] == 1
    assert audit["c1_retained_sibling_spectra"] == 1
    assert audit["c1_exact_query_matches_remaining"] == 0
    assert audit["status"] == "PASSED_STRICT_FULL_KEY_AUDIT"
