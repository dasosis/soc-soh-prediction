"""Leakage-safe splitting. This is the correctness-critical module: verify it
yourself, do not trust generated edits blindly.

Two families of split:
  1. GROUP HOLDOUT (cell-wise / chemistry-wise / temperature-wise):
     no group value ever appears in both train and test. Use for cross-dataset,
     cross-chemistry, cross-temperature generalization (leave-one-*-out).
  2. FORWARD-IN-TIME (SOH within a cell): train on early cycles, test on later
     cycles; guarantees max(train cycle) < min(test cycle) per cell.

NEVER random-shuffle timesteps or cycles. That leaks the future into the past
and is the #1 cause of fake sub-0.05% errors in the literature.
"""
from __future__ import annotations

import pandas as pd


def _ids_for_groups(meta_df: pd.DataFrame, group_col: str, values) -> list[str]:
    mask = meta_df[group_col].isin(values)
    return sorted(meta_df.loc[mask, "cell_id"].unique().tolist())


def group_holdout(meta_df: pd.DataFrame, group_col: str, holdout_values) -> dict:
    """Hold out all cells whose `group_col` is in `holdout_values` for test.

    `meta_df` must contain `cell_id` and `group_col` (e.g. the `cells` table, or
    a profile-level metadata frame for temperature splits).
    Returns {"train_cells": [...], "test_cells": [...]} and asserts disjointness.
    """
    if group_col not in meta_df.columns:
        raise ValueError(f"group_col '{group_col}' not in metadata columns")
    holdout_values = list(holdout_values)
    present = set(meta_df[group_col].unique())
    missing = [v for v in holdout_values if v not in present]
    if missing:
        raise ValueError(f"holdout values not present in '{group_col}': {missing}")

    test_cells = _ids_for_groups(meta_df, group_col, holdout_values)
    all_cells = set(meta_df["cell_id"].unique())
    train_cells = sorted(all_cells - set(test_cells))

    overlap = set(train_cells) & set(test_cells)
    assert not overlap, f"LEAKAGE: cells in both splits: {overlap}"
    if not test_cells:
        raise ValueError("empty test split")
    return {"train_cells": train_cells, "test_cells": test_cells}


def leave_one_group_out(meta_df: pd.DataFrame, group_col: str):
    """Yield (held_out_value, split_dict) for each unique value of group_col.

    e.g. leave-one-dataset-out:  leave_one_group_out(cells, "dataset")
         leave-one-chemistry-out: leave_one_group_out(cells, "chemistry")
    """
    for value in sorted(meta_df[group_col].unique()):
        yield value, group_holdout(meta_df, group_col, [value])


def forward_cycle_split(
    soh_df: pd.DataFrame, train_frac: float = 0.6, by: str = "cell_id"
) -> dict:
    """Within each cell, sort by cycle_index and take the first `train_frac`
    of cycles as train, the rest as test. Returns row-index lists into soh_df.

    Guarantees: for every shared cell, max(train cycle) < min(test cycle).
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, grp in soh_df.groupby(by, sort=False):
        grp_sorted = grp.sort_values("cycle_index")
        n = len(grp_sorted)
        k = max(1, int(round(n * train_frac)))
        k = min(k, n - 1)  # leave at least one test cycle
        train_idx.extend(grp_sorted.index[:k].tolist())
        test_idx.extend(grp_sorted.index[k:].tolist())

    # leakage assertion: per cell, last train cycle strictly before first test
    tr = soh_df.loc[train_idx]
    te = soh_df.loc[test_idx]
    for cell, g_tr in tr.groupby(by):
        g_te = te[te[by] == cell]
        if len(g_te):
            assert g_tr["cycle_index"].max() < g_te["cycle_index"].min(), (
                f"LEAKAGE in {cell}: train cycle overlaps/follows test cycle"
            )
    return {"train_idx": train_idx, "test_idx": test_idx}
