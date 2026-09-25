"""Diagnose (1) Protocol C identical ranks across variants and (2) pair-filter noise."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.adducts import neutral_mass_series
from src.core.config import DATA_DIR
from src.core.preprocessing import default_variants
from src.core.split import molecule_disjoint_exclude_rows, select_queries, select_val_molecules
from src.search.spectral_search import (
    build_eligible_mask,
    extract_query_peaks,
    prepare_query_block,
    protocol_c_ranks,
    run_search_pass,
)
from src.train_pipeline import _load_train_meta


def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    meta = _load_train_meta(train_path)
    valid = (meta["precursor_mz"] >= 30) & (meta["precursor_mz"] <= 2000)
    meta_valid = meta[valid]
    val_molecules = select_val_molecules(meta_valid, n_molecules=300, seed=42)
    queries = select_queries(meta_valid, val_molecules, seed=42)
    print(f"val_mols={len(val_molecules)} queries={len(queries)}", flush=True)

    all_variants = default_variants()
    # Compare one plain (1A) and one mass-filtered+modcos (1F) variant.
    variants = {k: all_variants[k] for k in ("1A", "1F")}
    raw_peaks = extract_query_peaks(train_path, queries["row_id"].to_numpy())
    mol_lookup = {str(k): int(v) for k, v in zip(meta["inchikey"], meta["mol_id"])}
    query_block = prepare_query_block("val", queries, raw_peaks, variants, mol_lookup)

    eligible = build_eligible_mask(
        meta["neutral_mass"].to_numpy(),
        meta["polarity"].to_numpy(),
        query_block.neutral,
        query_block.polarity,
        20.0,
    )
    print(f"eligible rows: {int(eligible.sum()):,}", flush=True)

    exclude_c = molecule_disjoint_exclude_rows(meta, val_molecules)
    print(f"exclude_c rows: {len(exclude_c):,}", flush=True)

    acc_c, timings_c = run_search_pass(
        train_path,
        variants,
        query_block,
        meta,
        exclude_rows=exclude_c,
        eligible=eligible,
        ppm=20.0,
        batch_size=256,
        max_chunks=8,
        progress=True,
        workers=2,
        checkpoint_path=None,
        resume=False,
    )

    for key in variants:
        acc = acc_c[key]
        n_nonempty = int(sum(1 for i in range(acc.mols.shape[0]) if np.any(acc.mols[i] >= 0)))
        print(f"{key}: nonempty_acc={n_nonempty}/{acc.mols.shape[0]} timings={timings_c[key]:.1f}s", flush=True)
        # Score distribution for first 20 queries
        for i in range(min(5, acc.mols.shape[0])):
            m = acc.mols[i]
            s = acc.scores[i]
            mask = m >= 0
            print(f"  q{i}: n_hits={int(mask.sum())} score[min,max]=({s[mask].min() if mask.any() else 'NA'},"
                  f"{s[mask].max() if mask.any() else 'NA'}) top5={list(zip(m[mask][:5], np.round(s[mask][:5], 4)))}",
                  flush=True)

    # Protocol C ranks comparison
    exclude_molecules = set(val_molecules)
    ranks_by_key = {}
    for key in variants:
        ranks_c, ranks_spec = protocol_c_ranks(
            meta,
            queries,
            acc_c[key],
            query_block.true_mol,
            exclude_molecules=exclude_molecules,
            ppm=20.0,
        )
        ranks_by_key[key] = ranks_c
        mrr = float(np.mean([1.0 / r if 1 <= r <= 25 else 0.0 for r in ranks_c]))
        print(f"Protocol C {key}: mrr={mrr:.6f} pure_spec_nonzero={int((ranks_spec > 0).sum())}", flush=True)

    r0, r1 = ranks_by_key["1A"], ranks_by_key["1F"]
    same = int((r0 == r1).sum())
    print(f"Protocol C rank agreement 1A vs 1F: {same}/{len(r0)} identical", flush=True)
    diff_idx = np.nonzero(r0 != r1)[0][:10]
    print(f"first differing query idx: {diff_idx.tolist()}", flush=True)

    # For a differing (or first) query: inspect mass-window candidates vs acc scores
    i = int(diff_idx[0]) if diff_idx.size else 0
    print(f"\n--- deep dive query {i} ---", flush=True)
    q = queries.iloc[i]
    print(f"true_mol={query_block.true_mol[i]} inchikey={q['inchikey']}", flush=True)

    nm_all = neutral_mass_series(meta["precursor_mz"], meta["adduct"])
    q_nm = float(neutral_mass_series(pd.Series([q["precursor_mz"]]), pd.Series([str(q["adduct"])]))[0])
    q_pol = -1 if "neg" in str(q["ionization_mode"]) else 1
    print(f"q_nm={q_nm} q_pol={q_pol}", flush=True)

    # Build mass-window candidates like protocol_c_ranks
    from src.search.candidate_filter import ISOTOPE_DELTA

    df_tmp = pd.DataFrame({"mol_id": meta["mol_id"].to_numpy(), "nm": nm_all,
                           "pol": meta["ionization_mode"].astype(str).to_numpy()})
    mol_table = df_tmp.groupby("mol_id", sort=True).agg(
        nm=("nm", "median"), pol=("pol", lambda s: s.mode().iloc[0] if len(s.mode()) else "positive")
    ).reset_index()
    mol_table["pol_pos"] = np.where(mol_table["pol"].str.contains("neg"), -1, 1).astype(np.int8)
    held = meta["inchikey"].isin(exclude_molecules)
    counts = meta.loc[~held].groupby("inchikey").size()
    inchi_by_mol = meta.groupby("mol_id", sort=True)["inchikey"].first()
    mol_table["inchikey"] = mol_table["mol_id"].map(inchi_by_mol)
    mol_table["count"] = mol_table["inchikey"].map(counts).fillna(0).astype(np.int64)

    table_nm = mol_table["nm"].to_numpy(dtype=np.float64)
    order = np.argsort(table_nm, kind="stable")
    table_nm = table_nm[order]
    table_pol = mol_table["pol_pos"].to_numpy(dtype=np.int8)[order]
    table_mol = mol_table["mol_id"].to_numpy(dtype=np.int64)[order]
    table_count = mol_table["count"].to_numpy(dtype=np.int64)[order]
    table_inchi = mol_table["inchikey"].to_numpy()[order]

    hit_idx = []
    for center in (q_nm, q_nm - ISOTOPE_DELTA, q_nm + ISOTOPE_DELTA):
        tol = center * 20.0 / 1e6
        lo = np.searchsorted(table_nm, center - tol, side="left")
        hi = np.searchsorted(table_nm, center + tol, side="right")
        hit_idx.extend(range(lo, hi))
    if not hit_idx:
        for center in (q_nm, q_nm - ISOTOPE_DELTA, q_nm + ISOTOPE_DELTA):
            tol = center * 50.0 / 1e6
            lo = np.searchsorted(table_nm, center - tol, side="left")
            hi = np.searchsorted(table_nm, center + tol, side="right")
            hit_idx.extend(range(lo, hi))
    cand = np.unique(np.asarray(hit_idx, dtype=np.int64)) if hit_idx else np.empty(0, dtype=np.int64)
    if cand.size:
        cand = cand[table_pol[cand] == q_pol]
    print(f"mass-window candidates: {cand.size}", flush=True)

    true_id = int(query_block.true_mol[i])
    print(f"true_id in candidates: {true_id in set(int(table_mol[c]) for c in cand)}", flush=True)

    for key in variants:
        acc = acc_c[key]
        acc_mols = acc.mols[i]
        acc_sc = acc.scores[i]
        mask = acc_mols >= 0
        score_map = {int(m): float(s) for m, s in zip(acc_mols[mask], acc_sc[mask])}
        mols_c = table_mol[cand]
        scores = np.array([score_map.get(int(m), 0.0) for m in mols_c])
        n_pos = int((scores > 0).sum())
        in_acc = int(sum(1 for m in mols_c if int(m) in score_map))
        print(f"{key}: mass-cand in acc={in_acc}/{cand.size} with_score>0={n_pos}", flush=True)
        if cand.size:
            sort_order = np.lexsort((mols_c, -table_count[cand], -scores))
            ranked = mols_c[sort_order]
            hit = np.nonzero(ranked == true_id)[0]
            rank = int(hit[0]) + 1 if hit.size else 0
            print(f"  rank_of_true={rank} top5_mols={ranked[:5].tolist()} top5_scores={scores[sort_order][:5].tolist()}",
                  flush=True)

    # --- Protocol A miss diagnosis for 1F ---
    print("\n=== Protocol A 1F miss diagnosis (max_chunks=8) ===", flush=True)
    acc_a, _ = run_search_pass(
        train_path,
        variants,
        query_block,
        meta,
        exclude_rows=queries["row_id"].to_numpy(),
        eligible=eligible,
        ppm=20.0,
        batch_size=256,
        max_chunks=8,
        progress=True,
        workers=2,
        checkpoint_path=None,
        resume=False,
    )
    for key in variants:
        acc = acc_a[key]
        has_true = acc.has_true(query_block.true_mol)
        print(f"{key}: frac_true_scored={has_true.mean():.4f} n_miss={int((~has_true).sum())}", flush=True)
        if key == "1F":
            miss_idx = np.nonzero(~has_true)[0]
            print(f"miss count={miss_idx.size}", flush=True)
            for i in miss_idx[:15]:
                q = queries.iloc[i]
                true_id = int(query_block.true_mol[i])
                # siblings in library?
                sib = int(((meta["inchikey"] == q["inchikey"]) & (~meta.index.isin(queries["row_id"]))).sum())
                q_nm = query_block.neutral[i]
                # true mol masses
                true_rows = meta[meta["mol_id"] == true_id]
                true_nm = neutral_mass_series(true_rows["precursor_mz"], true_rows["adduct"])
                pol_ok = (true_rows["ionization_mode"].astype(str).str.contains("neg").to_numpy() ==
                          (query_block.polarity[i] == -1))
                finite_true = np.isfinite(true_nm)
                ppm_err = []
                if np.isfinite(q_nm) and finite_true.any():
                    ppm_err = list(np.abs(true_nm[finite_true] - q_nm) / q_nm * 1e6)
                print(
                    f"  miss q{i}: true_id={true_id} siblings={sib} q_nm={q_nm} "
                    f"q_pol={query_block.polarity[i]} n_true_rows={len(true_rows)} "
                    f"true_nm_finite={int(finite_true.sum())} pol_match={int(pol_ok.sum())} "
                    f"ppm_err_min={min(ppm_err) if ppm_err else 'NA'} "
                    f"adduct_q={q['adduct']} adduct_true_sample={list(true_rows['adduct'].head(3))}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
