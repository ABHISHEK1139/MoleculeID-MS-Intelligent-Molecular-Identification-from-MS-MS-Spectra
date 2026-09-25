"""Classical spectral search engine.

Moved from src/spectral_search.py to src/search/spectral_search.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse

from src.core.config import (
    ACC_CAP,
    BIN_WIDTH,
    MZ_BIN_MAX,
    MZ_BIN_MIN,
    N_BINS,
)
from src.core.preprocessing import VariantConfig, deisotope_peaks, preprocess_variant

# Pre-cap for pathological rows (max num_peaks in train is ~73k) before deisotoping.
PRE_DEISO_CAP = 2000

# Per-query candidates retained from each chunk result before merging.
CAND_CAP = 800


def bin_indices(mz: np.ndarray) -> np.ndarray:
    return np.floor((mz - MZ_BIN_MIN) / BIN_WIDTH).astype(np.int64)


def valid_bin_mask(idx: np.ndarray) -> np.ndarray:
    return (idx >= 0) & (idx < N_BINS)


def profile_for_variant(key: str) -> str:
    return {"1A": "A", "1B": "B", "1C": "C", "1D": "D", "1E": "D", "1F": "D"}.get(key, "D")


def process_row_profiles(
    mz: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float | None,
    variants: dict[str, VariantConfig],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute the distinct preprocessing profiles needed by the variants.

    Variants 1A-1F only differ along a chain: raw -> noise -> precursor ->
    deisotope, so at most four spectra are derived per row.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    arr = np.asarray(mz, dtype=np.float32)
    vals = np.asarray(intensities, dtype=np.float32)
    if vals.size:
        peak_max = float(vals.max())
        if peak_max > 0:
            vals = vals / peak_max

    max_peaks = 100
    min_rel = 0.0
    remove_prec = False
    do_deiso = False
    for cfg in variants.values():
        min_rel = max(min_rel, cfg.min_rel)
        remove_prec = remove_prec or cfg.remove_precursor
        do_deiso = do_deiso or cfg.deisotope
        if cfg.max_peaks:
            max_peaks = cfg.max_peaks if max_peaks == 0 else max(max_peaks, cfg.max_peaks)

    # Profile A: normalize + top-N cap only.
    arr_a, vals_a = arr, vals
    if max_peaks and arr_a.size > max_peaks:
        order = np.argsort(vals_a)[::-1][:max_peaks]
        arr_a, vals_a = arr_a[order], vals_a[order]
    out["A"] = (arr_a, vals_a)

    # Profile B: + relative-intensity threshold.
    arr_b, vals_b = out["A"]
    if min_rel > 0 and arr_b.size:
        mask = vals_b >= min_rel
        arr_b, vals_b = arr_b[mask], vals_b[mask]
    out["B"] = (arr_b, vals_b)

    # Profile C: + precursor removal.
    arr_c, vals_c = out["B"]
    if remove_prec and precursor_mz is not None and arr_c.size:
        mask = np.abs(arr_c - precursor_mz) > 0.5
        arr_c, vals_c = arr_c[mask], vals_c[mask]
    out["C"] = (arr_c, vals_c)

    # Profile D: + deisotoping (re-apply top-N cap afterwards).
    arr_d, vals_d = out["C"]
    if do_deiso and arr_d.size > 1:
        if arr_d.size > PRE_DEISO_CAP:
            order = np.argsort(vals_d)[::-1][:PRE_DEISO_CAP]
            arr_d, vals_d = arr_d[order], vals_d[order]
        arr_d, vals_d = deisotope_peaks(arr_d, vals_d)
    if max_peaks and arr_d.size > max_peaks:
        order = np.argsort(vals_d)[::-1][:max_peaks]
        arr_d, vals_d = arr_d[order], vals_d[order]
    out["D"] = (arr_d, vals_d)

    # Fallback for future variants not on the A-D chain.
    for key, cfg in variants.items():
        if key in {"1A", "1B", "1C", "1D", "1E", "1F"}:
            out[key] = out[profile_for_variant(key)]
        else:
            out[key] = preprocess_variant(arr, vals, precursor_mz, cfg)
    return out


def unit_sparse_rows(
    mz: np.ndarray,
    intensities: np.ndarray,
    neighbor: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_idx, value) with unit-L2 intensities; optionally expand to neighbor bins."""
    if mz.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    norm = float(np.sqrt(np.sum(intensities.astype(np.float64) ** 2)))
    if norm <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    vals = (intensities / norm).astype(np.float32)
    idx = bin_indices(mz)
    ok = valid_bin_mask(idx)
    idx, vals = idx[ok], vals[ok]
    if idx.size == 0:
        return idx, vals
    if neighbor:
        idx = np.concatenate([idx - 1, idx, idx + 1])
        vals = np.concatenate([vals, vals, vals])
        ok = valid_bin_mask(idx)
        idx, vals = idx[ok], vals[ok]
    order = np.argsort(idx, kind="stable")
    idx, vals = idx[order], vals[order]
    uniq, start = np.unique(idx, return_index=True)
    sums = np.add.reduceat(vals, start)
    return uniq, sums.astype(np.float32)


def _stack_sparse(rows: list[tuple[np.ndarray, np.ndarray]]) -> sparse.csr_matrix:
    indptr = [0]
    indices: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for idx, vals in rows:
        indices.append(idx)
        data.append(vals)
        indptr.append(indptr[-1] + idx.size)
    return sparse.csr_matrix(
        (
            np.concatenate(data) if data else np.empty(0, dtype=np.float32),
            np.concatenate(indices) if indices else np.empty(0, dtype=np.int64),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(rows), N_BINS),
    )


def _chunk_csr(
    local_ids: np.ndarray,
    peaks: dict[int, tuple[np.ndarray, np.ndarray]],
) -> sparse.csr_matrix:
    rows = []
    for lid in local_ids:
        item = peaks.get(int(lid))
        if item is None:
            rows.append((np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)))
        else:
            rows.append(unit_sparse_rows(item[0], item[1], neighbor=False))
    return _stack_sparse(rows)


@dataclass
class QueryBlock:
    """Preprocessed queries for one pass."""

    name: str
    true_mol: np.ndarray
    neutral: np.ndarray
    polarity: np.ndarray
    precursor: np.ndarray
    peaks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    sparse: dict[str, sparse.csr_matrix] = field(default_factory=dict)
    meta: pd.DataFrame | None = None

    @property
    def n(self) -> int:
        return int(self.true_mol.size)


class Accumulator:
    """Fixed-capacity per-query (molecule, score) store."""

    def __init__(self, n_queries: int, cap: int = ACC_CAP):
        self.mols = np.full((n_queries, cap), -1, dtype=np.int32)
        self.scores = np.full((n_queries, cap), -1.0, dtype=np.float32)
        self.cap = cap

    def update(self, q: int, new_mols: np.ndarray, new_scores: np.ndarray) -> None:
        if new_mols.size == 0:
            return
        m = self.mols[q]
        s = self.scores[q]
        all_m = np.concatenate([m, new_mols]).astype(np.int64)
        all_s = np.concatenate([s, new_scores]).astype(np.float64)
        # Score-descending deduplication keeping highest-scoring occurrence of each molecule
        order = np.argsort(-all_s)
        all_m = all_m[order]
        all_s = all_s[order]
        valid = all_m >= 0
        all_m = all_m[valid]
        all_s = all_s[valid]
        if all_m.size:
            _, first_idx = np.unique(all_m, return_index=True)
            first_idx = np.sort(first_idx)
            all_m = all_m[first_idx]
            all_s = all_s[first_idx]
        if all_m.size > self.cap:
            all_m = all_m[: self.cap]
            all_s = all_s[: self.cap]
        n = all_m.size
        self.mols[q, :n] = all_m
        self.mols[q, n:] = -1
        self.scores[q, :n] = all_s
        self.scores[q, n:] = -1.0

    def ranks_for(self, true_mol: np.ndarray) -> np.ndarray:
        ranks = np.zeros(self.mols.shape[0], dtype=np.int64)
        for q in range(self.mols.shape[0]):
            target = int(true_mol[q])
            if target < 0:
                continue
            hit = np.nonzero(self.mols[q] == target)[0]
            if hit.size:
                ranks[q] = int(hit[0]) + 1
        return ranks

    def has_true(self, true_mol: np.ndarray) -> np.ndarray:
        out = np.zeros(self.mols.shape[0], dtype=bool)
        for q in range(self.mols.shape[0]):
            target = int(true_mol[q])
            if target >= 0:
                out[q] = bool(np.any(self.mols[q] == target))
        return out


def _mol_max_reduce(
    cols: np.ndarray,
    scores: np.ndarray,
    lib_mol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce (position, score) pairs to per-molecule max scores."""
    if cols.size == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    mols = lib_mol[cols]
    order = np.argsort(mols, kind="stable")
    mols = mols[order]
    sc = scores[order]
    uniq, start = np.unique(mols, return_index=True)
    max_sc = np.maximum.reduceat(sc, start)
    return uniq.astype(np.int32), max_sc.astype(np.float32)


def build_eligible_mask(
    lib_neutral: np.ndarray,
    lib_polarity: np.ndarray,
    query_neutral: np.ndarray,
    query_polarity: np.ndarray,
    ppm: float,
) -> np.ndarray:
    """Rows whose neutral mass/polarity match at least one query window (clean 20 ppm)."""
    eligible = np.zeros(lib_neutral.size, dtype=bool)

    for pol in (1, -1):
        lib_mask = (lib_polarity == pol) & np.isfinite(lib_neutral)
        lib_idx = np.nonzero(lib_mask)[0]
        if lib_idx.size == 0:
            continue
        nm_sorted = lib_neutral[lib_idx]
        order = np.argsort(nm_sorted, kind="stable")
        lib_idx = lib_idx[order]
        nm_sorted = nm_sorted[order]
        q_mask = (query_polarity == pol) & np.isfinite(query_neutral)
        for nm_q in query_neutral[q_mask]:
            tol = nm_q * ppm / 1e6
            lo = np.searchsorted(nm_sorted, nm_q - tol, side="left")
            hi = np.searchsorted(nm_sorted, nm_q + tol, side="right")
            if hi > lo:
                eligible[lib_idx[lo:hi]] = True
    return eligible


def _peak_matched_cosine(
    q_mz: np.ndarray,
    q_i: np.ndarray,
    l_mz: np.ndarray,
    l_i: np.ndarray,
    shift: float,
    tol: float = 0.01,
) -> float:
    if q_mz.size == 0 or l_mz.size == 0:
        return 0.0
    qn = float(np.sqrt(np.sum(q_i.astype(np.float64) ** 2)))
    ln = float(np.sqrt(np.sum(l_i.astype(np.float64) ** 2)))
    if qn <= 0 or ln <= 0:
        return 0.0
    q_mz_shift = q_mz + shift
    order_q = np.argsort(q_mz_shift)
    qs = q_mz_shift[order_q]
    qis = q_i[order_q]
    lo = np.searchsorted(qs, l_mz - tol, side="left")
    hi = np.searchsorted(qs, l_mz + tol, side="right")
    dot = 0.0
    for j in range(l_mz.size):
        a, b = int(lo[j]), int(hi[j])
        if b > a:
            dot += float(l_i[j]) * float(np.sum(qis[a:b]))
    return dot / (qn * ln)


def modified_cosine(
    q_mz: np.ndarray,
    q_i: np.ndarray,
    l_mz: np.ndarray,
    l_i: np.ndarray,
    delta: float,
) -> float:
    std = _peak_matched_cosine(q_mz, q_i, l_mz, l_i, shift=0.0)
    if abs(delta) < 1e-4:
        return std
    shifted_pos = _peak_matched_cosine(q_mz, q_i, l_mz, l_i, shift=delta)
    shifted_neg = _peak_matched_cosine(q_mz, q_i, l_mz, l_i, shift=-delta)
    return max(std, shifted_pos, shifted_neg)


def list_column_to_arrays(column: pa.Array | pa.ChunkedArray) -> list[np.ndarray]:
    """Convert a list<double> column into a list of float32 numpy arrays."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks() if column.num_chunks != 1 else column.chunk(0)
    flat = column.flatten()
    vals = np.asarray(flat.to_numpy(zero_copy_only=False), dtype=np.float32)
    offsets = np.asarray(column.offsets.to_numpy(zero_copy_only=False))
    return [vals[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


def extract_query_peaks(
    train_path,
    query_row_ids: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read (mz, intensity) arrays for specific global row ids in the train file."""
    from src.core.data_loader import iter_row_groups

    wanted = set(int(r) for r in query_row_ids)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, offset, table in iter_row_groups(
        train_path, columns=["ms2_mzs", "ms2_normalized_intensities"]
    ):
        n = table.num_rows
        local_wanted = sorted(r - offset for r in wanted if offset <= r < offset + n)
        if not local_wanted:
            continue
        mz_arrays = list_column_to_arrays(table.column("ms2_mzs"))
        int_arrays = list_column_to_arrays(table.column("ms2_normalized_intensities"))
        for lid in local_wanted:
            out[offset + lid] = (mz_arrays[lid], int_arrays[lid])
    missing = wanted - set(out)
    if missing:
        raise RuntimeError(f"Missing peaks for {len(missing)} query rows")
    return out


def _read_full_peak_file(path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read all spectra from a small parquet file keyed by positional index."""
    df_mz = pd.read_parquet(path, columns=["ms2_mzs", "ms2_normalized_intensities"])
    out = {}
    for i, (mz, inten) in enumerate(zip(df_mz["ms2_mzs"], df_mz["ms2_normalized_intensities"])):
        out[i] = (
            np.asarray(mz, dtype=np.float32),
            np.asarray(inten, dtype=np.float32),
        )
    return out


def prepare_query_block(
    name: str,
    query_df: pd.DataFrame,
    raw_peaks: dict[int, tuple[np.ndarray, np.ndarray]],
    variants: dict[str, VariantConfig],
    mol_lookup: dict[str, int] | None = None,
) -> QueryBlock:
    from src.core.adducts import neutral_mass_series

    n = len(query_df)
    if "row_id" in query_df.columns:
        row_ids = query_df["row_id"].to_numpy(dtype=np.int64)
    else:
        row_ids = np.arange(n, dtype=np.int64)
    precursor = query_df["precursor_mz"].to_numpy(dtype=np.float64)
    adduct = query_df["adduct"]
    neutral = neutral_mass_series(pd.Series(precursor), pd.Series(adduct.astype(str)))
    pol_str = query_df["ionization_mode"].astype(str)
    polarity = np.where(pol_str.str.contains("neg").to_numpy(), -1, 1).astype(np.int8)

    if mol_lookup is not None and "inchikey" in query_df.columns:
        true_mol = np.array([mol_lookup.get(str(m), -1) for m in query_df["inchikey"]], dtype=np.int32)
    else:
        true_mol = np.full(n, -1, dtype=np.int32)

    peaks: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {k: [] for k in variants}
    for i, rid in enumerate(row_ids):
        mz, inten = raw_peaks[int(rid)]
        prec = float(precursor[i]) if np.isfinite(precursor[i]) else None
        profiles = process_row_profiles(mz, inten, prec, variants)
        for key in variants:
            peaks[key].append(profiles[profile_for_variant(key)])

    sparse_rows: dict[str, sparse.csr_matrix] = {}
    for key in variants:
        rows = [unit_sparse_rows(m, i, neighbor=True) for m, i in peaks[key]]
        sparse_rows[key] = _stack_sparse(rows)

    return QueryBlock(
        name=name,
        true_mol=true_mol,
        neutral=neutral,
        polarity=polarity,
        precursor=precursor,
        peaks=peaks,
        sparse=sparse_rows,
        meta=query_df,
    )


def _init_worker(state: dict) -> None:
    """Initializer for spawned search workers."""
    global _WORKER_STATE
    _WORKER_STATE = state


_WORKER_STATE: dict = {}


def _collect(out: dict[str, dict[str, list]], key: str, qid: int, mols: np.ndarray, scores: np.ndarray) -> None:
    if mols.size == 0:
        return
    if mols.size > CAND_CAP:
        sel = np.argpartition(scores, -CAND_CAP)[-CAND_CAP:]
        mols = mols[sel]
        scores = scores[sel]
    bucket = out[key]
    n = mols.size
    bucket["q"].append(np.full(n, qid, dtype=np.int32))
    bucket["mols"].append(mols.astype(np.int32))
    bucket["scores"].append(scores.astype(np.float32))


def _process_chunk(chunk_i: int) -> dict:
    """Process one train row-group in a worker (or serially) and return candidate deltas."""
    import pyarrow.parquet as pq

    st = _WORKER_STATE
    variants: dict[str, VariantConfig] = st["variants"]
    offset = int(st["offsets"][chunk_i])
    train_path = st["train_path"]
    ppm = st["ppm"]
    batch_size = st["batch_size"]
    exclude_arr = st["exclude_arr"]
    eligible = st["eligible"]

    parquet = pq.ParquetFile(train_path)
    table = parquet.read_row_group(chunk_i, columns=["ms2_mzs", "ms2_normalized_intensities"])
    n_local = table.num_rows
    global_idx = np.arange(offset, offset + n_local, dtype=np.int64)
    lib_mask = ~np.isin(global_idx, exclude_arr)

    mol_ids = st["mol_ids"]
    neutral = st["neutral"]
    polarity = st["polarity"]
    precursor_all = st["precursor_all"]

    mz_arrays = list_column_to_arrays(table.column("ms2_mzs"))
    int_arrays = list_column_to_arrays(table.column("ms2_normalized_intensities"))
    prec_local = precursor_all[offset : offset + n_local]

    profiles_needed = sorted({profile_for_variant(k) for k in variants})
    mass_variants = [k for k, cfg in variants.items() if cfg.mass_filter]
    plain_variants = [k for k, cfg in variants.items() if not cfg.mass_filter]

    peak_maps: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {p: {} for p in profiles_needed}
    for lid in np.nonzero(lib_mask)[0]:
        prec = float(prec_local[lid]) if np.isfinite(prec_local[lid]) else None
        profiles = process_row_profiles(mz_arrays[lid], int_arrays[lid], prec, variants)
        for p in profiles_needed:
            peak_maps[p][int(lid)] = profiles[p]

    lib_mol_chunk = mol_ids[offset : offset + n_local]
    lib_neutral_chunk = neutral[offset : offset + n_local]
    lib_polarity_chunk = polarity[offset : offset + n_local]

    full_rows = np.nonzero(lib_mask)[0]
    csr_by_profile: dict[str, sparse.csr_matrix] = {}
    for p in profiles_needed:
        csr_by_profile[p] = _chunk_csr(full_rows, peak_maps[p])

    out: dict[str, dict[str, list]] = {
        k: {"q": [], "mols": [], "scores": []} for k in variants
    }
    seconds: dict[str, float] = {k: 0.0 for k in variants}
    n_q = int(st["n_q"])

    # --- Non-mass-filtered variants: batched full-library products ---
    for key in plain_variants:
        t0 = time.time()
        profile = profile_for_variant(key)
        lib_csr = csr_by_profile[profile]
        q_csr = st["q_sparse"][key]
        mol_view = lib_mol_chunk[full_rows]
        for b0 in range(0, n_q, batch_size):
            b1 = min(b0 + batch_size, n_q)
            product = (q_csr[b0:b1] @ lib_csr.T).tocsr()
            indptr, indices, data = product.indptr, product.indices, product.data
            for local_q in range(b1 - b0):
                a, b = indptr[local_q], indptr[local_q + 1]
                if b <= a:
                    continue
                mols, msc = _mol_max_reduce(indices[a:b], data[a:b], mol_view)
                _collect(out, key, b0 + local_q, mols, msc)
        seconds[key] += time.time() - t0

    # --- Mass-filtered variants ---
    if mass_variants and eligible is not None:
        local_eligible = eligible[offset : offset + n_local] & lib_mask
        elig_rows = np.nonzero(local_eligible)[0]
    else:
        elig_rows = None

    for key in mass_variants:
        t0 = time.time()
        cfg = variants[key]
        profile = profile_for_variant(key)
        lib_rows = full_rows if elig_rows is None else elig_rows
        if lib_rows.size == 0:
            continue
        profile_csr = csr_by_profile[profile]
        pos = np.searchsorted(full_rows, lib_rows)
        lib_csr = profile_csr[pos]

        q_csr = st["q_sparse"][key]
        product = (q_csr @ lib_csr.T).tocoo()
        row_q = product.row
        col_pos = product.col
        pair_scores = product.data.astype(np.float32)

        local_lids = lib_rows[col_pos]
        nm_lib = lib_neutral_chunk[local_lids]
        pol_lib = lib_polarity_chunk[local_lids]
        nm_q_sel = st["q_neutral"][row_q]
        pol_ok = st["q_polarity"][row_q] == pol_lib

        with np.errstate(invalid="ignore"):
            mass_ok = np.abs(nm_q_sel - nm_lib) <= (nm_q_sel * ppm / 1e6)
        mass_ok = mass_ok | ~np.isfinite(nm_q_sel)
        keep = pol_ok & mass_ok & np.isfinite(nm_lib)
        row_q = row_q[keep]
        col_pos = col_pos[keep]
        local_lids = local_lids[keep]
        pair_scores = pair_scores[keep]

        if cfg.modified_cosine and row_q.size:
            new_scores = np.empty(row_q.size, dtype=np.float32)
            peaks_map = peak_maps[profile]
            q_peaks = st["q_peaks"][key]
            for p in range(row_q.size):
                qid = int(row_q[p])
                item = peaks_map.get(int(local_lids[p]))
                if item is None:
                    new_scores[p] = pair_scores[p]
                    continue
                q_mz, q_i = q_peaks[qid]
                delta = st["q_precursor"][qid] - prec_local[int(local_lids[p])]
                new_scores[p] = modified_cosine(q_mz, q_i, item[0], item[1], delta)
            pair_scores = new_scores

        if row_q.size:
            order = np.argsort(row_q, kind="stable")
            row_s = row_q[order]
            col_s = col_pos[order]
            sc_s = pair_scores[order]
            bounds = np.searchsorted(row_s, np.arange(n_q + 1))
            mol_view = lib_mol_chunk[lib_rows]
            for qid in range(n_q):
                a, b = bounds[qid], bounds[qid + 1]
                if b <= a:
                    continue
                mols, msc = _mol_max_reduce(col_s[a:b], sc_s[a:b], mol_view)
                _collect(out, key, qid, mols, msc)
        seconds[key] += time.time() - t0

    results = {
        key: {
            "q": np.concatenate(bucket["q"]) if bucket["q"] else np.empty(0, dtype=np.int32),
            "mols": np.concatenate(bucket["mols"]) if bucket["mols"] else np.empty(0, dtype=np.int32),
            "scores": np.concatenate(bucket["scores"]) if bucket["scores"] else np.empty(0, dtype=np.float32),
        }
        for key, bucket in out.items()
    }
    return {"results": results, "seconds": seconds}


def _merge_chunk_result(accumulators: dict[str, Accumulator], result: dict) -> dict[str, float]:
    seconds = result.get("seconds", {})
    n_q = next(iter(accumulators.values())).mols.shape[0] if accumulators else 0
    for key, payload in result.get("results", {}).items():
        q = payload["q"]
        if q.size == 0:
            continue
        order = np.argsort(q, kind="stable")
        q = q[order]
        mols = payload["mols"][order]
        scores = payload["scores"][order]
        uniq, starts = np.unique(q, return_index=True)
        ends = np.r_[starts[1:], q.size]
        acc = accumulators[key]
        for qi, a, e in zip(uniq, starts, ends):
            acc.update(int(qi), mols[a:e], scores[a:e])
    return seconds


def run_search_pass(
    train_path,
    variants: dict[str, VariantConfig],
    query_block: QueryBlock,
    meta: pd.DataFrame,
    exclude_rows: np.ndarray | None = None,
    eligible: np.ndarray | None = None,
    ppm: float = 20.0,
    batch_size: int = 256,
    max_chunks: int | None = None,
    progress: bool = True,
    workers: int = 1,
    checkpoint_path=None,
    checkpoint_fingerprint: str | None = None,
    checkpoint_every: int = 3,
    resume: bool = True,
) -> tuple[dict[str, Accumulator], dict[str, float]]:
    """Score queries against the streamed train library.

    With workers > 1, train row-groups are processed in parallel worker
    processes; each worker returns capped per-chunk candidates that the parent
    merges into per-query accumulators.

    If checkpoint_path is set, accumulator state is saved every
    ``checkpoint_every`` completed chunks (and on clean exit). A later call
    with the same fingerprint resumes and skips already-merged chunks.
    """
    import pyarrow.parquet as pq

    from src.core.checkpoint import load_search_checkpoint, save_search_checkpoint

    exclude_arr = (
        np.asarray(exclude_rows, dtype=np.int64) if exclude_rows is not None else np.empty(0, dtype=np.int64)
    )

    parquet = pq.ParquetFile(train_path)
    n_groups = parquet.metadata.num_row_groups
    offsets: list[int] = []
    total = 0
    for i in range(n_groups):
        offsets.append(total)
        total += parquet.metadata.row_group(i).num_rows
    if max_chunks is not None:
        n_groups = min(n_groups, max_chunks)

    state = {
        "train_path": str(train_path),
        "variants": variants,
        "offsets": offsets,
        "mol_ids": meta["mol_id"].to_numpy(dtype=np.int32),
        "neutral": meta["neutral_mass"].to_numpy(dtype=np.float64),
        "polarity": meta["polarity"].to_numpy(dtype=np.int8),
        "precursor_all": meta["precursor_mz"].to_numpy(dtype=np.float64),
        "eligible": eligible,
        "exclude_arr": exclude_arr,
        "ppm": ppm,
        "batch_size": batch_size,
        "q_sparse": query_block.sparse,
        "q_neutral": query_block.neutral,
        "q_polarity": query_block.polarity,
        "q_precursor": query_block.precursor,
        "q_peaks": query_block.peaks,
        "n_q": query_block.n,
    }

    variant_keys = list(variants.keys())
    accumulators = {key: Accumulator(query_block.n) for key in variant_keys}
    timings = {key: 0.0 for key in variant_keys}

    done_chunks: set[int] = set()
    if (
        resume
        and checkpoint_path is not None
        and checkpoint_fingerprint is not None
    ):
        restored = load_search_checkpoint(
            checkpoint_path,
            fingerprint=checkpoint_fingerprint,
            variant_keys=variant_keys,
            n_queries=query_block.n,
            cap=ACC_CAP,
            accumulator_cls=Accumulator,
        )
        if restored is not None:
            accumulators = restored["accumulators"]
            timings.update(restored["timings"])
            done_chunks = {c for c in restored["done"] if c < n_groups}
            if progress and done_chunks:
                print(
                    f"  [checkpoint] resuming: {len(done_chunks)}/{n_groups} chunks already merged",
                    flush=True,
                )

    pending = [i for i in range(n_groups) if i not in done_chunks]
    t_start = time.time()
    done = len(done_chunks)
    since_ckpt = 0

    def _maybe_checkpoint(force: bool = False) -> None:
        nonlocal since_ckpt
        if checkpoint_path is None or checkpoint_fingerprint is None:
            return
        if not force and since_ckpt < checkpoint_every:
            return
        save_search_checkpoint(
            checkpoint_path,
            fingerprint=checkpoint_fingerprint,
            variant_keys=variant_keys,
            done_chunks=sorted(done_chunks),
            timings=timings,
            accumulators=accumulators,
        )
        since_ckpt = 0

    def _handle(result: dict, chunk_i: int) -> None:
        nonlocal done, since_ckpt
        seconds = _merge_chunk_result(accumulators, result)
        for key, sec in seconds.items():
            timings[key] = timings.get(key, 0.0) + sec
        done_chunks.add(int(chunk_i))
        done += 1
        since_ckpt += 1
        if progress and (done % 5 == 0 or done == n_groups):
            elapsed = time.time() - t_start
            print(f"  chunk {done}/{n_groups} done: elapsed {elapsed:.1f}s", flush=True)
        _maybe_checkpoint()

    if pending:
        if workers is None or workers <= 1:
            global _WORKER_STATE
            _WORKER_STATE = state
            for chunk_i in pending:
                _handle(_process_chunk(chunk_i), chunk_i)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(state,),
            ) as executor:
                futures = {executor.submit(_process_chunk, i): i for i in pending}
                for future in as_completed(futures):
                    chunk_i = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - surfaced to caller
                        raise RuntimeError(f"worker failed on chunk {chunk_i}") from exc
                    _handle(result, chunk_i)
    elif progress:
        print(f"  [checkpoint] all {n_groups} chunks already complete", flush=True)

    _maybe_checkpoint(force=True)
    return accumulators, timings


def protocol_b_ranks(
    meta: pd.DataFrame,
    queries: pd.DataFrame,
    exclude_molecules: set[str],
    ppm: float = 20.0,
) -> np.ndarray:
    """Structure-DB baseline: mass-filter molecules, rank by library popularity."""
    from src.core.adducts import neutral_mass_series

    mol_col = "inchikey"
    held_out = meta[mol_col].isin(exclude_molecules)
    counts = meta.loc[~held_out].groupby(mol_col).size()

    nm_all = neutral_mass_series(meta["precursor_mz"], meta["adduct"])
    df_tmp = pd.DataFrame(
        {
            mol_col: meta[mol_col].to_numpy(),
            "nm": nm_all,
            "pol": meta["ionization_mode"].astype(str).to_numpy(),
        }
    )
    mol_table = (
        df_tmp.groupby(mol_col, sort=True)
        .agg(nm=("nm", "median"), pol=("pol", lambda s: s.mode().iloc[0] if len(s.mode()) else "positive"))
        .reset_index()
    )
    mol_table["count"] = mol_table[mol_col].map(counts).fillna(0).astype(np.int64)
    mol_table["pol_pos"] = np.where(mol_table["pol"].str.contains("neg"), -1, 1).astype(np.int8)

    q_nm = neutral_mass_series(queries["precursor_mz"], queries["adduct"])
    q_pol = np.where(queries["ionization_mode"].astype(str).str.contains("neg"), -1, 1)
    true_keys = queries[mol_col].astype(str).to_numpy()

    table_nm = mol_table["nm"].to_numpy(dtype=np.float64)
    order = np.argsort(table_nm, kind="stable")
    table_nm = table_nm[order]
    table_pol = mol_table["pol_pos"].to_numpy(dtype=np.int8)[order]
    table_keys = mol_table[mol_col].to_numpy()[order]
    table_counts = mol_table["count"].to_numpy(dtype=np.int64)[order]

    ranks = np.zeros(len(queries), dtype=np.int64)
    for i in range(len(queries)):
        nm_q = q_nm[i]
        if not np.isfinite(nm_q):
            continue
        tol = nm_q * ppm / 1e6
        lo = np.searchsorted(table_nm, nm_q - tol, side="left")
        hi = np.searchsorted(table_nm, nm_q + tol, side="right")
        cand = np.arange(lo, hi)
        cand = cand[table_pol[cand] == q_pol[i]]
        if cand.size == 0:
            continue
        keys = table_keys[cand]
        counts_c = table_counts[cand]
        sort_order = np.lexsort((keys, -counts_c))
        ranked_keys = keys[sort_order]
        hit = np.nonzero(ranked_keys == true_keys[i])[0]
        ranks[i] = int(hit[0]) + 1 if hit.size else 0
    return ranks


def protocol_c_ranks(
    meta: pd.DataFrame,
    queries: pd.DataFrame,
    acc: Accumulator,
    true_mol: np.ndarray,
    exclude_molecules: set[str],
    ppm: float = 20.0,
    ppm_fallback: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Protocol C / Class-2-like ranks: molecule-disjoint library.

    The spectral search ran with 0 library spectra of every val molecule, so
    the true molecule never receives a cosine score. Candidates are still the
    full structure DB (val molecules included) inside a mass window (primary
    ppm, then M±1 isotope / expanded ppm fallback). Score =
      - accumulator cosine if that molecule has non-val library spectra, else
      - 0.0 (mass-compatible structure-only candidate).
    Tie-break: higher score, then non-val popularity, then mol_id.

    Returns (ranks, pure_spectral_ranks). pure_spectral_ranks only counts
    hits that came from the accumulator (always 0 for val molecules — shows
    why Protocol A cannot be the Stage-2 target).
    """
    from src.core.adducts import neutral_mass_series
    from src.search.candidate_filter import ISOTOPE_DELTA

    mol_col = "inchikey"
    held_out = meta[mol_col].isin(exclude_molecules)
    counts = meta.loc[~held_out].groupby(mol_col).size()

    nm_all = neutral_mass_series(meta["precursor_mz"], meta["adduct"])
    df_tmp = pd.DataFrame(
        {
            "mol_id": meta["mol_id"].to_numpy(),
            "nm": nm_all,
            "pol": meta["ionization_mode"].astype(str).to_numpy(),
        }
    )
    mol_table = (
        df_tmp.groupby("mol_id", sort=True)
        .agg(nm=("nm", "median"), pol=("pol", lambda s: s.mode().iloc[0] if len(s.mode()) else "positive"))
        .reset_index()
    )
    # Popularity from non-val spectra only (val spectra are not in the library).
    inchikey_by_mol = meta.groupby("mol_id", sort=True)[mol_col].first()
    mol_table[mol_col] = mol_table["mol_id"].map(inchikey_by_mol)
    mol_table["count"] = mol_table[mol_col].map(counts).fillna(0).astype(np.int64)
    mol_table["pol_pos"] = np.where(mol_table["pol"].str.contains("neg"), -1, 1).astype(np.int8)

    q_nm = neutral_mass_series(queries["precursor_mz"], queries["adduct"])
    q_pol = np.where(queries["ionization_mode"].astype(str).str.contains("neg"), -1, 1)

    table_nm = mol_table["nm"].to_numpy(dtype=np.float64)
    order = np.argsort(table_nm, kind="stable")
    table_nm = table_nm[order]
    table_pol = mol_table["pol_pos"].to_numpy(dtype=np.int8)[order]
    table_mol = mol_table["mol_id"].to_numpy(dtype=np.int64)[order]
    table_count = mol_table["count"].to_numpy(dtype=np.int64)[order]

    n_q = len(queries)
    ranks = np.zeros(n_q, dtype=np.int64)
    spectral_only = np.zeros(n_q, dtype=np.int64)

    for i in range(n_q):
        nm_q = q_nm[i]
        if not np.isfinite(nm_q):
            continue

        # Mass windows: primary, isotope (primary ppm), then expanded fallback
        # around M and around M±1 isotope centers (50 ppm).
        hit_idx: list[int] = []
        centers_p = [nm_q, nm_q - ISOTOPE_DELTA, nm_q + ISOTOPE_DELTA]
        for center in centers_p:
            tol = center * ppm / 1e6
            lo = np.searchsorted(table_nm, center - tol, side="left")
            hi = np.searchsorted(table_nm, center + tol, side="right")
            hit_idx.extend(range(lo, hi))
        if not hit_idx:
            for center in centers_p:
                tol = center * ppm_fallback / 1e6
                lo = np.searchsorted(table_nm, center - tol, side="left")
                hi = np.searchsorted(table_nm, center + tol, side="right")
                hit_idx.extend(range(lo, hi))
        if not hit_idx:
            continue
        cand = np.unique(np.asarray(hit_idx, dtype=np.int64))
        cand = cand[table_pol[cand] == q_pol[i]]
        if cand.size == 0:
            continue

        # Spectral scores from the molecule-disjoint search (val mols => none).
        acc_mols = acc.mols[i]
        acc_sc = acc.scores[i]
        valid = acc_mols >= 0
        score_map = {int(m): float(s) for m, s in zip(acc_mols[valid], acc_sc[valid])}

        mols_c = table_mol[cand]
        scores = np.fromiter((score_map.get(int(m), 0.0) for m in mols_c), dtype=np.float64, count=mols_c.size)
        counts_c = table_count[cand]
        # Higher score, then popularity, then stable mol_id.
        sort_order = np.lexsort((mols_c, -counts_c, -scores))
        ranked_mols = mols_c[sort_order]
        ranked_scores = scores[sort_order]

        true_id = int(true_mol[i])
        hit = np.nonzero(ranked_mols == true_id)[0]
        if hit.size:
            ranks[i] = int(hit[0]) + 1
            # Pure spectral: only credit if the hit had a positive cosine
            # (i.e. the molecule actually had library spectra in Protocol C).
            if ranked_scores[hit[0]] > 0:
                spectral_only[i] = ranks[i]

    return ranks, spectral_only


def aggregate_test_ranking(
    acc: Accumulator,
    molecule_ids: np.ndarray,
    top_k: int,
) -> dict[str, list[int]]:
    """Combine per-spectrum accumulator rows into per-molecule ranked lists."""
    out: dict[str, list[int]] = {}
    unique_mols = pd.unique(molecule_ids)
    for mol in unique_mols:
        rows = np.nonzero(molecule_ids == mol)[0]
        mols_cat = np.concatenate([acc.mols[r] for r in rows])
        sc_cat = np.concatenate([acc.scores[r] for r in rows])
        mask = mols_cat >= 0
        mols_cat = mols_cat[mask]
        sc_cat = sc_cat[mask]
        if mols_cat.size == 0:
            out[str(mol)] = []
            continue
        # Deduplicate keeping highest-scoring unique molecules
        order = np.argsort(-sc_cat)
        mols_cat = mols_cat[order]
        sc_cat = sc_cat[order]
        _, first_idx = np.unique(mols_cat, return_index=True)
        first_idx = np.sort(first_idx)
        mols_cat = mols_cat[first_idx]
        out[str(mol)] = [int(x) for x in mols_cat[:top_k]]
    return out
