"""Shared regression metrics. Same units in -> same units out.

All functions take 1-D array-likes. Report these for BOTH tasks so cross-method
comparison is apples-to-apples. For SOC report rmse/mae/max_error/mape; for SOH
add r2 and (in the pipeline) RUL absolute error from an EOL threshold.
"""
from __future__ import annotations

import numpy as np


def _prep(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    return yt, yp


def rmse(y_true, y_pred) -> float:
    yt, yp = _prep(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _prep(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def max_error(y_true, y_pred) -> float:
    yt, yp = _prep(y_true, y_pred)
    return float(np.max(np.abs(yt - yp)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Mean absolute percentage error (%). Guarded against divide-by-zero;
    near-zero targets (SOC~0) inflate MAPE, so prefer rmse/mae for SOC."""
    yt, yp = _prep(y_true, y_pred)
    return float(np.mean(np.abs((yt - yp) / (np.abs(yt) + eps))) * 100.0)


def r2(y_true, y_pred) -> float:
    yt, yp = _prep(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def soc_report(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }


def soh_report(y_true, y_pred) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
