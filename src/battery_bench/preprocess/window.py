"""Sliding-window construction for sequence SOC models.

Windows NEVER cross a group boundary (default: profile_id). Mixing two drive
cycles inside one window is a subtle form of leakage / nonsense input, so the
grouping is enforced here, not left to the caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    length: int,
    stride: int = 1,
    group_col: str = "profile_id",
    add_ocv_channel: bool = False,
    ocv_chem: str | None = None,
    voltage_col: str = "voltage_V",
):
    """Return (X, y, groups):
      X      : (N, length, F) float32
      y      : (N,)           float32, label at the LAST step of each window
      groups : (N,)           the group_col value each window came from

    Stage-3 OCV feature (OFF by default — default path is byte-identical to
    before): with ``add_ocv_channel=True`` and ``ocv_chem`` set, an OCV-implied
    SOC channel (from the chemistry's OCV table, see preprocess.ocv_feature) is
    computed elementwise from ``voltage_col`` and appended as the last channel,
    making X (N, length, F+1). NOTE: ``voltage_col`` must hold RAW volts for the
    OCV lookup to be meaningful (the SOC pipeline computes it before scaling).
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    feature_cols = list(feature_cols)
    if add_ocv_channel:
        from .ocv_feature import soc_from_voltage
        soc_ocv, _ = soc_from_voltage(df[voltage_col].to_numpy(), ocv_chem)
        df = df.assign(__soc_ocv=np.asarray(soc_ocv, dtype=float))
        feature_cols = feature_cols + ["__soc_ocv"]
    X_list, y_list, g_list = [], [], []
    for gval, grp in df.groupby(group_col, sort=False):
        feats = grp[feature_cols].to_numpy(dtype=np.float32)
        labels = grp[label_col].to_numpy(dtype=np.float32)
        n = len(grp)
        for start in range(0, n - length + 1, stride):
            end = start + length
            X_list.append(feats[start:end])
            y_list.append(labels[end - 1])
            g_list.append(gval)
    if not X_list:
        return (
            np.empty((0, length, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=object),
        )
    return (
        np.stack(X_list).astype(np.float32),
        np.asarray(y_list, dtype=np.float32),
        np.asarray(g_list, dtype=object),
    )
