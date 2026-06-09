"""Stage 2 (target-domain adaptation) helpers: target TEST/POOL splitting,
profile-level fraction sampling from POOL, an internal fine-tune train/val carve,
and affine/bias recalibration of zero-shot predictions. Pure numpy/pandas — no
torch, no training — so the leakage + budget guards are unit-testable without
fitting anything.

Shared-budget contract (enforced + tested):
  * TEST and POOL are disjoint profile sets, fixed by a dedicated seed and
    CONSTANT across all fractions / methods / models (there is NO separate fixed
    VAL: it would give fine-tuning an off-budget target-label advantage);
  * at fraction f, sample f*POOL profiles -> this is the TOTAL target-label
    budget at f, shared identically by every method;
  * recalibration fits on the FULL sampled fraction; fine-tuning carves its
    early-stopping val from INSIDE that same sampled fraction (profile-level),
    so both methods consume the same budget, just partitioned differently;
  * fine-tune internal train/val are disjoint and never overlap TEST;
  * at fraction 0 recalibration is the identity (a=1, b=0) -> exactly zero-shot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_target(meta: pd.DataFrame, target_pids, test_frac: float, seed: int):
    """Partition TARGET profiles into (test, pool), stratified by
    nominal_temperature_C, disjoint, deterministic in ``seed``.

    There is NO separate fixed VAL — fine-tuning carves its early-stopping val
    from inside the sampled fraction (see ``split_finetune_train_val``) so every
    method shares the same per-fraction target-label budget. The TEST assignment
    is byte-identical to the earlier TEST/VAL/POOL split (same seed, same per-temp
    permutation, same ``nt``); the former VAL profiles simply fold back into POOL,
    so fraction-0 zero-shot stays comparable to prior runs."""
    rng = np.random.default_rng(seed)
    sub = meta[meta["profile_id"].isin(set(target_pids))]
    test, pool = [], []
    for _, g in sub.groupby("nominal_temperature_C"):
        pids = sorted(g["profile_id"])
        pids = [pids[i] for i in rng.permutation(len(pids))]
        n = len(pids)
        nt = max(int(round(n * test_frac)), 1) if n >= 2 else 0
        test += pids[:nt]
        pool += pids[nt:]          # former VAL + POOL
    assert not (set(test) & set(pool)), "TEST/POOL overlap"
    return sorted(test), sorted(pool)


def split_finetune_train_val(sampled_pids, seed: int, fraction: float,
                             val_frac: float = 0.25, min_profiles: int = 3):
    """Carve a fine-tune early-stopping val from INSIDE the sampled fraction,
    profile-level. Returns ``(train_pids, val_pids, early_stop)``.

    If the sample has fewer than ``min_profiles`` profiles it cannot be split
    fairly -> returns ``(all, [], False)`` and the caller trains a fixed number
    of epochs without early stopping. Otherwise holds out ~``val_frac`` of the
    sampled profiles (>=1) for val; train and val are disjoint and (being a
    subset of the sample, which is a subset of POOL) never touch TEST.
    Deterministic in (seed, fraction)."""
    pids = sorted(sampled_pids)
    if len(pids) < int(min_profiles):
        return pids, [], False
    rng = np.random.default_rng([int(seed), int(round(fraction * 1000)), 17])
    pids = [pids[i] for i in rng.permutation(len(pids))]
    nv = min(max(1, int(round(len(pids) * val_frac))), len(pids) - 1)
    return sorted(pids[nv:]), sorted(pids[:nv]), True


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
