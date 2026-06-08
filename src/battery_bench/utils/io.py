"""Parquet I/O and split-manifest persistence for reproducibility."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_manifest(obj: dict, path: str | Path) -> None:
    """Persist a split manifest (lists of cell_ids / row indices) as JSON so a
    run is fully reproducible from disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def load_manifest(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
