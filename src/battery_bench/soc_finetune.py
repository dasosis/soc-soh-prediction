"""Stage 2 (target-domain adaptation) helpers: target TEST/VAL/POOL splitting,
profile-level fraction sampling from POOL, and affine/bias recalibration of
zero-shot predictions. Pure numpy/pandas — no torch, no training — so the
leakage guards are unit-testable without fitting anything.

Leakage contract (enforced + tested):
  * TEST, VAL, POOL are disjoint profile sets, fixed by a dedicated seed, and
    CONSTANT across all fractions / methods / models;
  * fine-tune / recalibration data is sampled only from POOL;
  * at fraction 0 recalibration is the identity (a=1, b=0) -> exactly zero-shot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_target(meta: pd.DataFrame, target_pids, test_frac: float,
                 val_frac: float, seed: int):
    """Partition TARGET profiles into (test, val, pool), stratified by
    nominal_temperature_C, disjoint, deterministic in ``seed``. TEST and VAL are
    meant to be held constant across the whole ladder."""
    rng = np.random.default_rng(seed)
    sub = meta[meta["profile_id"].isin(set(target_pids))]
    test, val, pool = [], [], []
    for _, g in sub.groupby("nominal_temperature_C"):
        pids = sorted(g["profile_id"])
        pids = [pids[i] for i in rng.permutation(len(pids))]
        n = len(pids)
        nt = max(int(round(n * test_frac)), 1) if n >= 2 else 0
        nv = max(int(round(n * val_frac)), 1) if (n - nt) >= 2 else 0
        # keep a non-empty pool whenever the temperature has enough profiles
        while n - nt - nv < 1 and n >= 3:
            if nv > 0:
                nv -= 1
            elif nt > 1:
                nt -= 1
            else:
                break
        test += pids[:nt]
        val += pids[nt:nt + nv]
        pool += pids[nt + nv:]
    assert not (set(test) & set(val)), "TEST/VAL overlap"
    assert not (set(test) & set(pool)), "TEST/POOL overlap"
    assert not (set(val) & set(pool)), "VAL/POOL overlap"
    return sorted(test), sorted(val), sorted(pool)


def sample_pool(pool_pids, fraction: float, seed: int):
    """Profile-level sample of ~``fraction`` of POOL. Deterministic in
    (seed, fraction); fraction<=0 -> [], fraction>=1 -> all of POOL."""
    pool = sorted(pool_pids)
    if fraction <= 0 or not pool:
        return []
    if fraction >= 1:
        return pool
    rng = np.random.default_rng([int(seed), int(round(fraction * 1000))])
    k = min(max(1, int(round(len(pool) * fraction))), len(pool))
    return sorted(rng.choice(pool, k, replace=False).tolist())


def affine_fit(y_pred, y_true):
    """Least-squares a, b for SOC' = a*y_pred + b."""
    yp = np.asarray(y_pred, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    A = np.vstack([yp, np.ones_like(yp)]).T
    sol, *_ = np.linalg.lstsq(A, yt, rcond=None)
    return float(sol[0]), float(sol[1])


def bias_fit(y_pred, y_true):
    """Constant-offset only (a=1): b = mean(true - pred)."""
    return float(np.mean(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))


def recalibrate_test(pool_pred, pool_true, test_zeroshot, mode="affine"):
    """Fit the recalibration on POOL zero-shot predictions, apply to TEST
    zero-shot predictions. Empty pool (fraction 0) -> identity -> zero-shot.
    Returns (recalibrated_test_pred, (a, b))."""
    test_zeroshot = np.asarray(test_zeroshot, dtype=float)
    if len(np.asarray(pool_pred)) == 0:
        return test_zeroshot, (1.0, 0.0)
    if mode == "affine":
        a, b = affine_fit(pool_pred, pool_true)
    elif mode == "bias":
        a, b = 1.0, bias_fit(pool_pred, pool_true)
    else:
        raise ValueError(f"unknown recalibration mode '{mode}'")
    return a * test_zeroshot + b, (a, b)
