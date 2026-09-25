from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.checkpoint import (
    atomic_write_json,
    atomic_write_text,
    clear_checkpoints,
    config_fingerprint,
)
from src.core.config import (
    BASELINE_DIR,
    DATA_DIR,
    DEFAULT_TOP_K,
    N_VAL_MOLECULES,
    PRECURSOR_MZ_MAX,
    PRECURSOR_MZ_MIN,
    SPLIT_SEED,
    STAGE01_RESULTS_PATH,
    STAGE01_SUBMISSION_PATH,
    SUBMISSION_PATH,
    TEST_PATH,
    TRAIN_PATH,
)
from src.core.adducts import neutral_mass_series
from src.core.data_loader import preview_dataframe
from src.core.evaluation import summarize_ranks, rank_candidates
from src.core.preprocessing import default_variants, preprocess_spectrum
from src.core.split import (
    molecule_disjoint_exclude_rows,
    select_queries,
    select_val_molecules,
)
from src.search.spectral_search import (
    _read_full_peak_file,
    aggregate_test_ranking,
    build_eligible_mask,
    extract_query_peaks,
    prepare_query_block,
    protocol_b_ranks,
    protocol_c_ranks,
    run_search_pass,
)

ALL_EXPERIMENTS = ("1A", "1B", "1C", "1D", "1E", "1F")
ALL_PROTOCOLS = ("A", "B", "C")


def run_baseline_pipeline(data_dir: str | Path | None = None, limit: int = 5) -> dict:
    """Minimal pipeline runner for development and debugging (quick smoke test)."""
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    train_path = data_dir / "train.parquet"

    if not train_path.exists():
        return {
            "status": "missing_data",
            "message": f"Train data not found at {train_path}. Add the Kaggle dataset to continue.",
            "rows": 0,
        }

    from src.core.data_loader import load_train_data

    df = load_train_data(data_dir)
    preview_dataframe(df, n=min(limit, len(df)))

    first_row = df.iloc[0]
    mz = first_row.get("ms2_mzs")
    intensities = first_row.get("ms2_normalized_intensities")
    precursor_mz = first_row.get("precursor_mz")

    if mz is None or intensities is None:
        return {
            "status": "schema_error",
            "message": "The dataset schema does not include required MS/MS fields.",
            "rows": len(df),
        }

    processed_mz, processed_intensities = preprocess_spectrum(
        mz=mz,
        intensities=intensities,
        precursor_mz=precursor_mz,
        min_rel=0.01,
        max_peaks=100,
        remove_precursor=True,
    )

    synthetic_reference = [
        processed_intensities,
        [0.5 * x for x in processed_intensities],
        [0.1 * x for x in processed_intensities],
    ]
    ranked = rank_candidates(processed_intensities, synthetic_reference)

    return {
        "status": "ok",
        "message": "Baseline preprocessing and ranking were executed successfully.",
        "rows": len(df),
        "processed_peaks": len(processed_mz),
        "top_ranked": ranked[:5],
    }


def _load_train_meta(train_path: Path) -> pd.DataFrame:
    cols = [
        "precursor_mz",
        "adduct",
        "ionization_mode",
        "inchikey",
        "normalized_smiles",
        "ingest_lib",
    ]
    meta = pd.read_parquet(train_path, columns=cols)
    meta["mol_id"] = pd.factorize(meta["inchikey"])[0].astype(np.int32)
    meta["neutral_mass"] = neutral_mass_series(meta["precursor_mz"], meta["adduct"])
    meta["polarity"] = np.where(
        meta["ionization_mode"].astype(str).str.contains("neg"), -1, 1
    ).astype(np.int8)
    return meta


def _molecule_table(meta: pd.DataFrame) -> pd.DataFrame:
    """Per-molecule neutral mass, polarity, popularity, and canonical SMILES."""
    table = (
        meta.groupby("mol_id", sort=True)
        .agg(
            inchikey=("inchikey", "first"),
            smiles=("normalized_smiles", "first"),
            nm=("neutral_mass", "median"),
            pol=("polarity", lambda s: (int(np.sign(s.mode().iloc[0])) or 1) if len(s.mode()) else 1),
            count=("mol_id", "size"),
        )
        .reset_index()
    )
    return table


def _mrr_table(ranks: np.ndarray, queries: pd.DataFrame, top_k: int) -> dict:
    summary = summarize_ranks(ranks, k=top_k)
    by_lib: dict[str, float] = {}
    libs = queries["ingest_lib"].astype(str).to_numpy()
    for lib in pd.unique(libs):
        mask = libs == lib
        sub = summarize_ranks(ranks[mask], k=top_k)
        by_lib[str(lib)] = round(float(sub["mrr"]), 4)
    summary["by_ingest_lib_mrr"] = by_lib
    return summary


def run_stage01(
    data_dir: str | Path | None = None,
    n_val_molecules: int = N_VAL_MOLECULES,
    seed: int = SPLIT_SEED,
    experiments: tuple[str, ...] = ALL_EXPERIMENTS,
    make_submission: bool = True,
    ppm: float = 20.0,
    top_k: int = DEFAULT_TOP_K,
    max_chunks: int | None = None,
    batch_size: int = 256,
    progress: bool = True,
    workers: int = 4,
    fresh: bool = False,
    protocols: tuple[str, ...] = ALL_PROTOCOLS,
) -> dict:
    """Stage 0/1: classical spectral retrieval baseline with MRR evaluation.

    Protocols:
      A — Class-1-like (siblings in library; near-duplicate retrieval).
      B — structure-DB popularity baseline (no spectra used for ranking).
      C — molecule-disjoint (0 library spectra of val molecules; honest
          Class-2-like target Stage 2+ must beat).

    Crash-safe: search state is checkpointed every few chunks, results JSON
    is written atomically as soon as evaluation finishes (before the
    submission pass), and a re-run with the same config resumes automatically.
    Pass fresh=True to discard checkpoints and start over.
    """
    data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"
    t_start = time.time()
    protocols = tuple(p.upper() for p in protocols)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = BASELINE_DIR / "checkpoints"
    if fresh:
        removed = clear_checkpoints(ckpt_dir, prefix="search_")
        if progress and removed:
            print(f"[stage01] --fresh: removed {removed} checkpoint file(s)", flush=True)

    eval_fp = config_fingerprint(
        {
            "kind": "stage01_eval",
            "seed": seed,
            "n_val_molecules": n_val_molecules,
            "experiments": list(experiments),
            "protocols": list(protocols),
            "ppm": ppm,
            "top_k": top_k,
            "max_chunks": max_chunks,
            "batch_size": batch_size,
            "train": str(train_path),
        }
    )
    eval_ckpt = ckpt_dir / f"search_{eval_fp}.npz"
    eval_c_fp = config_fingerprint(
        {
            "kind": "stage01_eval_c",
            "seed": seed,
            "n_val_molecules": n_val_molecules,
            "experiments": list(experiments),
            "ppm": ppm,
            "top_k": top_k,
            "max_chunks": max_chunks,
            "batch_size": batch_size,
            "train": str(train_path),
        }
    )
    eval_c_ckpt = ckpt_dir / f"search_{eval_c_fp}.npz"

    print("[stage01] loading train metadata...", flush=True)
    meta = _load_train_meta(train_path)
    if progress:
        print(f"  rows={len(meta):,} molecules={meta['mol_id'].nunique():,}", flush=True)

    valid_prec = (
        (meta["precursor_mz"] >= PRECURSOR_MZ_MIN)
        & (meta["precursor_mz"] <= PRECURSOR_MZ_MAX)
    )
    meta_valid = meta[valid_prec]

    val_molecules = select_val_molecules(
        meta_valid, n_molecules=n_val_molecules, seed=seed
    )
    queries = select_queries(meta_valid, val_molecules, seed=seed)
    if progress:
        print(
            f"[stage01] {len(val_molecules)} val molecules, "
            f"{len(queries)} held-out query spectra",
            flush=True,
        )

    all_variants = default_variants()
    variants = {k: all_variants[k] for k in experiments if k in all_variants}

    raw_peaks = extract_query_peaks(train_path, queries["row_id"].to_numpy())
    mol_lookup = {str(k): int(v) for k, v in zip(meta["inchikey"], meta["mol_id"])}
    query_block = prepare_query_block("val", queries, raw_peaks, variants, mol_lookup)

    eligible = None
    if any(cfg.mass_filter for cfg in variants.values()):
        eligible = build_eligible_mask(
            meta["neutral_mass"].to_numpy(),
            meta["polarity"].to_numpy(),
            query_block.neutral,
            query_block.polarity,
            ppm,
        )
        if progress:
            print(f"[stage01] mass-filter eligible rows: {int(eligible.sum()):,}", flush=True)

    if "A" in protocols:
        print("[stage01] running Protocol A library search (siblings in library)...", flush=True)
        acc, timings = run_search_pass(
            train_path,
            variants,
            query_block,
            meta,
            exclude_rows=queries["row_id"].to_numpy(),
            eligible=eligible,
            ppm=ppm,
            batch_size=batch_size,
            max_chunks=max_chunks,
            progress=progress,
            workers=workers,
            checkpoint_path=eval_ckpt,
            checkpoint_fingerprint=eval_fp,
            resume=not fresh,
        )

        protocol_a: dict[str, dict] = {}
        for key in variants:
            ranks = acc[key].ranks_for(query_block.true_mol)
            has_true = acc[key].has_true(query_block.true_mol)
            summary = _mrr_table(ranks, queries, top_k)
            summary["frac_true_scored"] = round(float(has_true.mean()), 4)
            summary["seconds"] = round(timings[key], 1)
            protocol_a[key] = summary
            print(
                f"  {key} ({variants[key].name}): MRR@{top_k}={summary['mrr']:.4f} "
                f"hit@1={summary['hit@1']:.3f} scored={summary['frac_true_scored']:.3f} "
                f"({timings[key]:.1f}s)",
                flush=True,
            )
    else:
        protocol_a = {}
        acc = None
        timings = {k: 0.0 for k in variants}

    protocol_b = None
    if "B" in protocols:
        print("[stage01] protocol B (structure-DB popularity baseline)...", flush=True)
        pB_ranks = protocol_b_ranks(meta, queries, set(val_molecules), ppm=ppm)
        protocol_b = _mrr_table(pB_ranks, queries, top_k)
        print(f"  protocol B: MRR@{top_k}={protocol_b['mrr']:.4f}", flush=True)

    protocol_c: dict[str, dict] = {}
    if "C" in protocols:
        print("[stage01] Protocol C search (molecule-disjoint: 0 val spectra in library)...", flush=True)
        exclude_c = molecule_disjoint_exclude_rows(meta, val_molecules)
        if progress:
            print(f"  excluding {len(exclude_c):,} val-molecule spectra from library", flush=True)
        acc_c, timings_c = run_search_pass(
            train_path,
            variants,
            query_block,
            meta,
            exclude_rows=exclude_c,
            eligible=eligible,
            ppm=ppm,
            batch_size=batch_size,
            max_chunks=max_chunks,
            progress=progress,
            workers=workers,
            checkpoint_path=eval_c_ckpt,
            checkpoint_fingerprint=eval_c_fp,
            resume=not fresh,
        )
        for key in variants:
            ranks_c, ranks_spec = protocol_c_ranks(
                meta,
                queries,
                acc_c[key],
                query_block.true_mol,
                exclude_molecules=set(val_molecules),
                ppm=ppm,
            )
            summary = _mrr_table(ranks_c, queries, top_k)
            summary["pure_spectral_mrr"] = round(
                float(summarize_ranks(ranks_spec, k=top_k)["mrr"]), 4
            )
            summary["frac_true_scored"] = round(
                float(acc_c[key].has_true(query_block.true_mol).mean()), 4
            )
            summary["seconds"] = round(timings_c[key], 1)
            protocol_c[key] = summary
            print(
                f"  {key} ({variants[key].name}): Protocol C MRR@{top_k}={summary['mrr']:.4f} "
                f"hit@1={summary['hit@1']:.3f} "
                f"pure_spectral={summary['pure_spectral_mrr']:.4f} "
                f"scored={summary['frac_true_scored']:.3f} "
                f"({timings_c[key]:.1f}s)",
                flush=True,
            )

    winner = max(protocol_a, key=lambda k: protocol_a[k]["mrr"]) if protocol_a else None
    if protocol_c and winner is None:
        winner = max(protocol_c, key=lambda k: protocol_c[k]["mrr"])

    results = {
        "config": {
            "n_val_molecules": len(val_molecules),
            "n_queries": int(len(queries)),
            "seed": seed,
            "ppm": ppm,
            "top_k": top_k,
            "experiments": list(experiments),
            "protocols": list(protocols),
            "max_chunks": max_chunks,
            "fingerprint": eval_fp,
            "protocol_a_note": (
                "Class-1-like: one spectrum per val molecule is held out as query; "
                "the molecule's remaining spectra stay in the library."
            ),
            "protocol_c_note": (
                "Molecule-disjoint: 0 library spectra of any val molecule. "
                "Ranks come from mass-window structure candidates scored by "
                "non-val cosine when available, else popularity."
            ),
            "mass_filter": f"strict {ppm} ppm neutral mass + polarity matching",
        },
        "protocol_a": protocol_a,
        "protocol_b": protocol_b,
        "protocol_c": protocol_c,
        "winner": winner,
        "total_seconds": round(time.time() - t_start, 1),
    }

    # Persist evaluation results BEFORE the (long) submission pass so a crash
    # mid-submission never loses the metrics.
    atomic_write_json(STAGE01_RESULTS_PATH, results)
    print(f"[stage01] evaluation results saved to {STAGE01_RESULTS_PATH}", flush=True)

    submission_path = None
    if make_submission and winner is not None and test_path.exists():
        print(f"[stage01] submission pass with winner={winner}...", flush=True)
        sub_fp = config_fingerprint(
            {
                "kind": "stage01_submission",
                "winner": winner,
                "ppm": ppm,
                "top_k": top_k,
                "max_chunks": max_chunks,
                "batch_size": batch_size,
                "train": str(train_path),
                "test": str(test_path),
            }
        )
        submission_path = _run_submission_pass(
            train_path=train_path,
            test_path=test_path,
            data_dir=data_dir,
            meta=meta,
            variant_key=winner,
            ppm=ppm,
            top_k=top_k,
            batch_size=batch_size,
            max_chunks=max_chunks,
            progress=progress,
            workers=workers,
            checkpoint_dir=ckpt_dir,
            checkpoint_fingerprint=sub_fp,
            resume=not fresh,
        )
        results["submission_path"] = str(submission_path)
        results["total_seconds"] = round(time.time() - t_start, 1)
        atomic_write_json(STAGE01_RESULTS_PATH, results)

    # Clean eval/submission search checkpoints only after full success.
    clear_checkpoints(ckpt_dir, prefix="search_")
    print(f"[stage01] results saved to {STAGE01_RESULTS_PATH}", flush=True)
    print(f"[stage01] total time {results['total_seconds']}s", flush=True)
    return results


def _run_submission_pass(
    train_path: Path,
    test_path: Path,
    data_dir: Path,
    meta: pd.DataFrame,
    variant_key: str,
    ppm: float,
    top_k: int,
    batch_size: int,
    max_chunks: int | None,
    progress: bool,
    workers: int = 4,
    checkpoint_dir: Path | None = None,
    checkpoint_fingerprint: str | None = None,
    resume: bool = True,
) -> Path:
    all_variants = default_variants()
    variants = {variant_key: all_variants[variant_key]}

    test_df = pd.read_parquet(test_path)
    test_df = test_df.reset_index(drop=True)
    test_df["row_id"] = np.arange(len(test_df), dtype=np.int64)

    test_raw = _read_full_peak_file(test_path)
    test_block = prepare_query_block("test", test_df, test_raw, variants, mol_lookup=None)

    eligible = None
    if variants[variant_key].mass_filter:
        eligible = build_eligible_mask(
            meta["neutral_mass"].to_numpy(),
            meta["polarity"].to_numpy(),
            test_block.neutral,
            test_block.polarity,
            ppm,
        )

    checkpoint_path = None
    if checkpoint_dir is not None and checkpoint_fingerprint is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"search_{checkpoint_fingerprint}.npz"

    acc, timings = run_search_pass(
        train_path,
        variants,
        test_block,
        meta,
        exclude_rows=None,
        eligible=eligible,
        ppm=ppm,
        batch_size=batch_size,
        max_chunks=max_chunks,
        progress=progress,
        workers=workers,
        checkpoint_path=checkpoint_path,
        checkpoint_fingerprint=checkpoint_fingerprint,
        resume=resume,
    )
    if progress:
        print(f"  submission search took {timings[variant_key]:.1f}s", flush=True)

    ranked = aggregate_test_ranking(
        acc[variant_key],
        test_df["molecule_id"].astype(str).to_numpy(),
        top_k,
    )

    mol_table = _molecule_table(meta)
    smiles_by_mol = dict(zip(mol_table["mol_id"], mol_table["smiles"]))
    # Global popularity fallback pool.
    popular = mol_table.sort_values(["count", "mol_id"], ascending=[False, True])
    global_fallback = popular["smiles"].tolist()

    # Per test molecule metadata (use first spectrum's adduct/polarity).
    test_meta = (
        test_df.assign(row_id=test_df["row_id"])
        .groupby("molecule_id", sort=False)
        .agg(
            prec=("precursor_mz", "median"),
            adduct=("adduct", "first"),
            pol=("ionization_mode", "first"),
        )
        .reset_index()
    )
    test_meta["nm"] = neutral_mass_series(test_meta["prec"], test_meta["adduct"])
    test_meta["pol_i"] = np.where(test_meta["pol"].astype(str).str.contains("neg"), -1, 1)

    sub_order = pd.read_csv(data_dir / "sample_submission.csv")["molecule_id"].tolist()

    # Molecule-level mass candidates for fallback padding.
    mol_nm = mol_table["nm"].to_numpy(dtype=np.float64)
    mol_pol = mol_table["pol"].to_numpy(dtype=np.int64)
    mol_count = mol_table["count"].to_numpy(dtype=np.int64)
    mol_smiles = mol_table["smiles"].to_numpy()
    mol_order = np.argsort(mol_nm, kind="stable")
    mol_nm = mol_nm[mol_order]
    mol_pol = mol_pol[mol_order]
    mol_count = mol_count[mol_order]
    mol_smiles = mol_smiles[mol_order]

    rows = []
    for molecule_id in sub_order:
        mols = ranked.get(str(molecule_id), [])
        smiles_list: list[str] = []
        seen: set[str] = set()
        for mid in mols:
            smi = smiles_by_mol.get(int(mid))
            if smi and smi not in seen:
                smiles_list.append(smi)
                seen.add(smi)
            if len(smiles_list) >= top_k:
                break

        if len(smiles_list) < top_k:
            trow = test_meta[test_meta["molecule_id"] == molecule_id]
            if len(trow) and np.isfinite(trow["nm"].iloc[0]):
                nm_q = float(trow["nm"].iloc[0])
                pol_q = int(trow["pol_i"].iloc[0])
                tol = nm_q * ppm / 1e6
                lo = int(np.searchsorted(mol_nm, nm_q - tol, side="left"))
                hi = int(np.searchsorted(mol_nm, nm_q + tol, side="right"))
                cand_idx = np.arange(lo, hi)
                cand_idx = cand_idx[mol_pol[cand_idx] == pol_q]
                if cand_idx.size:
                    sort_local = np.lexsort((cand_idx, -mol_count[cand_idx]))
                    for ci in cand_idx[sort_local]:
                        smi = str(mol_smiles[ci])
                        if smi not in seen:
                            smiles_list.append(smi)
                            seen.add(smi)
                        if len(smiles_list) >= top_k:
                            break

        for smi in global_fallback:
            if len(smiles_list) >= top_k:
                break
            if smi not in seen:
                smiles_list.append(str(smi))
                seen.add(str(smi))

        while len(smiles_list) < top_k:
            smiles_list.append("CCO")

        rows.append({"molecule_id": molecule_id, "smiles": ";".join(smiles_list[:top_k])})

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STAGE01_SUBMISSION_PATH
    # Atomic CSV write: power loss mid-write leaves the previous file intact.
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["molecule_id", "smiles"])
    for r in rows:
        writer.writerow([r["molecule_id"], r["smiles"]])
    atomic_write_text(out_path, buf.getvalue())
    if progress:
        print(f"  wrote {out_path} ({len(rows)} molecules)", flush=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="CASMI26 training pipeline")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Path to the data directory")
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to preview (quick mode)")
    parser.add_argument(
        "--mode",
        choices=["quick", "stage01", "full"],
        default="stage01",
        help="Pipeline mode",
    )
    parser.add_argument("--val-molecules", type=int, default=N_VAL_MOLECULES)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--experiments",
        type=str,
        default=",".join(ALL_EXPERIMENTS),
        help="Comma-separated experiment keys, e.g. 1A,1B,1E",
    )
    parser.add_argument("--no-submission", action="store_true")
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-chunks", type=int, default=None, help="Debug: limit row groups")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes for library search")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard existing search checkpoints and start from scratch",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        default=",".join(ALL_PROTOCOLS),
        help="Comma-separated protocols to run, e.g. A,B,C",
    )
    args = parser.parse_args()

    if args.mode == "quick":
        run_baseline_pipeline(args.data_dir, limit=args.limit)
    elif args.mode == "stage01":
        experiments = tuple(("1D" if e.strip() == "1" else e.strip()) for e in args.experiments.split(",") if e.strip())
        run_stage01(
            data_dir=args.data_dir,
            n_val_molecules=args.val_molecules,
            seed=args.seed,
            experiments=experiments,
            make_submission=not args.no_submission,
            ppm=args.ppm,
            top_k=args.top_k,
            max_chunks=args.max_chunks,
            batch_size=args.batch_size,
            workers=args.workers,
            fresh=args.fresh,
            protocols=tuple(p.strip().upper() for p in args.protocols.split(",") if p.strip()),
        )
    else:
        print("Full pipeline mode is not yet implemented; use --mode stage01.")


if __name__ == "__main__":
    main()
