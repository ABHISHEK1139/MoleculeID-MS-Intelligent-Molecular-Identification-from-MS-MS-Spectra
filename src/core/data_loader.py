"""Parquet I/O and streaming data access.

Moved from src/data_loader.py to src/core/data_loader.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq

from src.core.config import TRAIN_PATH


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a parquet dataset if it exists."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_parquet(path)


def load_train_data(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Return the train dataset from the configured path."""
    if data_dir is None:
        data_path = TRAIN_PATH
    else:
        data_path = Path(data_dir) / "train.parquet"

    return load_parquet(data_path)


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def preview_dataframe(df: pd.DataFrame, n: int = 5) -> None:
    print(df.head(n).to_string(index=False))


def iter_row_groups(path: str | Path, columns: list[str] | None = None):
    """Yield (row_group_index, global_row_offset, table) for streaming reads."""
    path = Path(path)
    parquet = pq.ParquetFile(path)
    offset = 0
    for i in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(i, columns=columns)
        yield i, offset, table
        offset += table.num_rows
