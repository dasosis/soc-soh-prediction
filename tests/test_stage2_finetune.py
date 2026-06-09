"""Stage 2 leakage + recalibration tests (fast, no training)."""
import numpy as np
import pandas as pd

from battery_bench import soc_finetune as FT

FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]


def _toy_meta(n_per_temp=8, temps=(0.0, 25.0, 40.0), dataset="LG_HG2"):
    rows = []
    for temp in temps:
        for p in range(n_per_temp):
            rows.append(dict(profile_id=f"{dataset}__{int(temp)}__{p}", dataset=dataset,
                             nominal_temperature_C=temp, drive_cycle="US06"))
    return pd.DataFrame(rows)


def test_target_split_disjoint_and_covers():
    meta = _toy_meta()
    target = list(meta["profile_id"])
    test, val, pool = FT.split_target(meta, target, test_frac=0.4, val_frac=0.15, seed=0)
    assert test and val and pool
    assert not (set(test) & set(val)) and not (set(test) & set(pool)) and not (set(val) & set(pool))
    assert set(test) | set(val) | set(pool) == set(target)
    # deterministic: same seed -> identical TEST (constant across the ladder)
    test2, _, _ = FT.split_target(meta, target, 0.4, 0.15, seed=0)
    assert test == test2


def test_sampled_pool_subset_and_test_never_leaks():
    meta = _toy_meta()
    target = list(meta["profile_id"])
    test, val, pool = FT.split_target(meta, target, 0.4, 0.15, seed=0)
    test_set, val_set, pool_set = set(test), set(val), set(pool)
    for f in FRACTIONS:
        for seed in range(3):
            sampled = FT.sample_pool(pool, f, seed)
            assert set(sampled) <= pool_set            # only from POOL
            assert not (set(sampled) & test_set)       # never TEST
            assert not (set(sampled) & val_set)        # never VAL
            if f == 0.0:
                assert sampled == []
            if f >= 1.0:
                assert set(sampled) == pool_set


def test_recalibrate_identity_at_fraction_zero():
    test_zs = np.array([0.2, 0.5, 0.9])
    for mode in ("affine", "bias"):
        out, (a, b) = FT.recalibrate_test(np.array([]), np.array([]), test_zs, mode)
        assert np.allclose(out, test_zs)
        assert (a, b) == (1.0, 0.0)


def test_affine_and_bias_recover_known_transform():
    pool_pred = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    pool_true = 2.0 * pool_pred + 0.1          # a=2, b=0.1
    test_zs = np.array([0.3, 0.6])
    out, (a, b) = FT.recalibrate_test(pool_pred, pool_true, test_zs, "affine")
    assert abs(a - 2.0) < 1e-6 and abs(b - 0.1) < 1e-6
    assert np.allclose(out, 2.0 * test_zs + 0.1)
    # bias-only keeps slope 1, shifts by mean residual
    out_b, (ab, bb) = FT.recalibrate_test(pool_pred, pool_true, test_zs, "bias")
    assert ab == 1.0
    assert abs(bb - float(np.mean(pool_true - pool_pred))) < 1e-9
