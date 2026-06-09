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
    test, pool = FT.split_target(meta, target, test_frac=0.4, seed=0)   # TEST + POOL only
    assert test and pool
    assert not (set(test) & set(pool))
    assert set(test) | set(pool) == set(target)
    # deterministic: same seed -> identical TEST (constant across the ladder)
    test2, _ = FT.split_target(meta, target, 0.4, seed=0)
    assert test == test2


def test_sampled_pool_subset_and_test_never_leaks():
    meta = _toy_meta()
    target = list(meta["profile_id"])
    test, pool = FT.split_target(meta, target, 0.4, seed=0)
    test_set, pool_set = set(test), set(pool)
    for f in FRACTIONS:
        for seed in range(3):
            sampled = FT.sample_pool(pool, f, seed)
            assert set(sampled) <= pool_set            # only from POOL
            assert not (set(sampled) & test_set)       # never TEST
            if f == 0.0:
                assert sampled == []
            if f >= 1.0:
                assert set(sampled) == pool_set


def test_finetune_internal_split_shares_budget_and_avoids_test():
    meta = _toy_meta()
    target = list(meta["profile_id"])
    test, pool = FT.split_target(meta, target, 0.4, seed=0)
    test_set = set(test)

    # a healthy sample splits into disjoint train/val that together == the sample
    sampled = FT.sample_pool(pool, 0.5, seed=0)
    tr, va, early = FT.split_finetune_train_val(sampled, seed=0, fraction=0.5,
                                                val_frac=0.25, min_profiles=3)
    assert early is True
    assert tr and va
    assert not (set(tr) & set(va))                      # internal train/val disjoint
    assert set(tr) | set(va) == set(sampled)            # consumes the full shared budget
    assert not ((set(tr) | set(va)) & test_set)         # never overlaps TEST

    # too few profiles -> fixed-epoch path (no early stopping, no val held out)
    tiny = sampled[:2]
    tr2, va2, early2 = FT.split_finetune_train_val(tiny, seed=0, fraction=0.05, min_profiles=3)
    assert early2 is False and va2 == [] and sorted(tr2) == sorted(tiny)


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
