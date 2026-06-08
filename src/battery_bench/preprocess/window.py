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
):
    """Return (X, y, groups):
      X      : (N, length, F) float32
      y      : (N,)           float32, label at the LAST step of each window
      groups : (N,)           the group_col value each window came from
    """
    if length < 1:
        raise ValueError("length must be >= 1")
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
