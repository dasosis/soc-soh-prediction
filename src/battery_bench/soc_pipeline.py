"""SOC experiment data plumbing: profile-level splits, source-only normalization,
and window construction. Reuses the harness (`preprocess.window`,
`preprocess.normalize`) WITHOUT modifying any core module.

Leakage rules enforced here:
  * splits are profile-level (a profile is wholly train, val, or test);
  * the Standardizer is fit on TRAIN profiles only and applied unchanged to val
    and test;
  * for cross-dataset experiments the train/val both come from the SOURCE
    dataset and the scaler is fit on SOURCE only, so the covariate shift the
    experiment measures is preserved on the TARGET test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocess.normalize import Standardizer
from .preprocess.window import make_windows


def profile_meta(soc: pd.DataFrame) -> pd.DataFrame:
    """One row per profile: dataset, nominal_temperature_C, drive_cycle."""
    return (soc.groupby("profile_id", as_index=False)
               .agg(dataset=("dataset", "first"),
                    nominal_temperature_C=("nominal_temperature_C", "first"),
                    drive_cycle=("drive_cycle", "first")))


def _sample_by_temperature(meta: pd.DataFrame, frac: float, rng) -> list[str]:
    """Pick ~frac of profiles, stratified by temperature, always leaving at least
    one profile per temperature behind (so train keeps every temperature)."""
    picked: list[str] = []
    for _, g in meta.groupby("nominal_temperature_C"):
        pids = sorted(g["profile_id"])
        n = len(pids)
        if n <= 1:
            continue
        k = min(max(1, int(round(n * frac))), n - 1)
        picked += rng.choice(pids, k, replace=False).tolist()
    return picked


def split_within(meta: pd.DataFrame, dataset: str, test_frac: float = 0.2,
                 val_frac: float = 0.2, seed: int = 0):
    """Within-dataset profile holdout (in-distribution sanity).

    Temperature-stratified so every nominal_temperature_C is represented in
    train; carves a val set out of the remaining train profiles. No profile
    appears in more than one split.
    """
    rng = np.random.default_rng(seed)
    sub = meta[meta["dataset"] == dataset]
    test = set(_sample_by_temperature(sub, test_frac, rng))
    remaining = sub[~sub["profile_id"].isin(test)]
    val = set(_sample_by_temperature(remaining, val_frac, rng))
    train = [p for p in sub["profile_id"] if p not in test and p not in val]
    return sorted(train), sorted(val), sorted(test)


def split_cross(manifest: dict, meta: pd.DataFrame, val_frac: float = 0.2, seed: int = 0):
    """Cross-dataset zero-shot split from a leave-one-dataset-out manifest.

    train/val are carved (temperature-stratified) from the SOURCE
    ``train_profiles``; test is the TARGET ``test_profiles`` untouched.
    """
    rng = np.random.default_rng(seed)
    src = manifest["train_profiles"]
    src_meta = meta[meta["profile_id"].isin(src)]
    val = set(_sample_by_temperature(src_meta, val_frac, rng))
    train = [p for p in src if p not in val]
    return sorted(train), sorted(val), list(manifest["test_profiles"])


def fit_source_scaler(soc: pd.DataFrame, train_pids, features) -> Standardizer:
    """Fit a Standardizer on TRAIN-profile rows only (the source in cross-dataset
    experiments). This is the single point where train statistics are learned."""
    train_rows = soc[soc["profile_id"].isin(set(train_pids))]
    return Standardizer(list(features)).fit(train_rows)


def make_split_windows(soc, pids, scaler, features, target, length, stride):
    """Normalize (with the already-fit scaler) then window one split. Windows
    never cross a profile_id (enforced by the harness windower)."""
    df = soc[soc["profile_id"].isin(set(pids))]
    df = scaler.transform(df)
    X, y, groups = make_windows(df, list(features), target, length=length,
                                stride=stride, group_col="profile_id")
    return X, y, groups


def subsample_windows(X, y, max_n, seed=0):
    """Deterministically keep at most ``max_n`` windows (for smoke mode)."""
    if max_n is None or len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X), max_n, replace=False))
    return X[idx], y[idx]
