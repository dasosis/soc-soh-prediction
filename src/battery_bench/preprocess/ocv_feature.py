"""OCV-informed SOC feature for Stage 3.

Maps terminal voltage to an OCV-implied SOC via a monotone inverse of each
chemistry's measured 25 degC C/20 OCV(SOC) table (built by
scripts/build_ocv_tables.py into data/ocv/). The feature injects chemistry
knowledge a plain (V, I, T) model lacks: the SAME voltage means different SOC on
NMC vs NCA, which is exactly the cross-dataset gap Stage 3 targets.

Used at TRAIN with the SOURCE chemistry's table and at TEST with the TARGET
chemistry's table (label-free: one C/20 curve per chemistry, no target labels).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_DIRS = [Path(__file__).resolve().parents[3] / "data" / "ocv", Path("data/ocv")]
_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _norm_chem(chem: str) -> str:
    c = str(chem).lower()
    if "lg" in c or c == "nmc":
        return "lg"
    if "pan" in c or "nca" in c:
        return "pan"
    raise ValueError(f"unknown chemistry '{chem}' (expected LG_HG2/NMC or PANASONIC_18650PF/NCA)")


def _resolve(chem: str):
    """Parse 'CHEM[:variant]'. variant 'dischg' -> discharge-leg-only table,
    otherwise the hysteresis-averaged 'mean' table. Returns (cache_key, filename)."""
    base, _, variant = str(chem).partition(":")
    key = _norm_chem(base)
    variant = variant.strip().lower()
    suffix = "_dischg" if variant == "dischg" else ""
    return f"{key}:{variant or 'mean'}", f"ocv_{key}_25C{suffix}.csv"


def load_ocv_table(chem: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (soc_grid, ocv_grid); ocv_grid is monotone-increasing. ``chem`` may
    carry a ':dischg' suffix to select the discharge-leg-only table."""
    cache_key, fname = _resolve(chem)
    if cache_key not in _CACHE:
        path = next((d / fname for d in _DIRS if (d / fname).exists()), None)
        if path is None:
            raise FileNotFoundError(f"OCV table {fname} not found; run scripts/build_ocv_tables.py")
        df = pd.read_csv(path)
        soc = df["soc"].to_numpy(dtype=float)
        ocv = df["ocv_V"].to_numpy(dtype=float)
        if not np.all(np.diff(ocv) > 0):
            raise ValueError(f"OCV table {fname} is not strictly increasing")
        _CACHE[cache_key] = (soc, ocv)
    return _CACHE[cache_key]


def soc_from_voltage(V, chem: str):
    """Map voltage -> OCV-implied SOC via the monotone inverse of ocv(soc).

    Returns ``(soc_ocv, clamped_mask)``. Voltages outside the table's measured
    range are CLAMPED to [soc_grid.min(), 1.0] (the table tops out at SOC 1.0),
    and ``clamped_mask`` marks those samples. Vectorized; preserves input shape.
    """
    soc_grid, ocv_grid = load_ocv_table(chem)
    V = np.asarray(V, dtype=float)
    # ocv_grid is strictly increasing -> use it as xp for a monotone inverse.
    soc = np.interp(V, ocv_grid, soc_grid)               # clamps to fp endpoints outside range
    soc = np.clip(soc, float(soc_grid.min()), 1.0)
    clamped = (V < ocv_grid[0]) | (V > ocv_grid[-1])
    return soc, clamped
