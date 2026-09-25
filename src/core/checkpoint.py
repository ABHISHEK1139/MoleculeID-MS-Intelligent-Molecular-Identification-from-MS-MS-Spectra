"""Crash-safe checkpointing and atomic artifact writes.

Guarantees:
- Atomic JSON/bytes writes (temp file + os.replace) so a power cut never
  leaves a truncated stage01_results.json / submission CSV.
- Search-pass checkpoints (numpy npz) that record which train row-groups
  have been merged, so a resumed run skips completed chunks.
- Config fingerprints so a checkpoint from a different seed / experiment
  set / ppm is never silently reused.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# ── Atomic writes ──────────────────────────────────────────────────────────


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write bytes to path atomically (crash leaves old file or new file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2))


def atomic_write_csv(path: str | Path, text: str) -> None:
    atomic_write_text(path, text)


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


# ── Fingerprints ───────────────────────────────────────────────────────────


def config_fingerprint(cfg: dict) -> str:
    """Stable short hash of a config dict (sorted keys)."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ── Search-pass checkpoints ────────────────────────────────────────────────


def save_search_checkpoint(
    path: str | Path,
    fingerprint: str,
    variant_keys: list[str],
    done_chunks: list[int],
    timings: dict[str, float],
    accumulators: dict[str, Any],
) -> None:
    """Persist accumulator state + completed chunk ids into one npz file.

    accumulators[key] must expose .mols and .scores as (n_queries, cap) arrays.
    """
    path = Path(path)
    meta = {
        "fingerprint": fingerprint,
        "variants": list(variant_keys),
        "done": sorted(int(c) for c in done_chunks),
        "timings": {k: float(timings.get(k, 0.0)) for k in variant_keys},
    }
    arrays: dict[str, np.ndarray] = {
        "__meta__": np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
    }
    for key in variant_keys:
        acc = accumulators[key]
        arrays[f"{key}__mols"] = np.asarray(acc.mols)
        arrays[f"{key}__scores"] = np.asarray(acc.scores)

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_search_checkpoint(
    path: str | Path,
    fingerprint: str,
    variant_keys: list[str],
    n_queries: int,
    cap: int,
    accumulator_cls: Any,
) -> dict | None:
    """Load a search checkpoint; return None if missing/corrupt/mismatched.

    On success returns {"done": set[int], "timings": dict, "accumulators": dict}.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if "__meta__" not in data.files:
                return None
            meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))
            if meta.get("fingerprint") != fingerprint:
                return None
            if list(meta.get("variants", [])) != list(variant_keys):
                return None
            accumulators = {}
            for key in variant_keys:
                mols_key = f"{key}__mols"
                scores_key = f"{key}__scores"
                if mols_key not in data.files or scores_key not in data.files:
                    return None
                mols = np.array(data[mols_key], copy=True)
                scores = np.array(data[scores_key], copy=True)
                if mols.shape != (n_queries, cap) or scores.shape != (n_queries, cap):
                    return None
                acc = accumulator_cls(n_queries, cap=cap)
                acc.mols = mols
                acc.scores = scores
                accumulators[key] = acc
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return None
    return {
        "done": {int(c) for c in meta.get("done", [])},
        "timings": {k: float(v) for k, v in meta.get("timings", {}).items()},
        "accumulators": accumulators,
    }


def clear_checkpoints(directory: str | Path, prefix: str = "search_") -> int:
    """Delete matching checkpoint files; returns count removed."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    n = 0
    for path in directory.glob(f"{prefix}*.npz"):
        try:
            path.unlink()
            n += 1
        except OSError:
            pass
    return n
