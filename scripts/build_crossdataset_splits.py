#!/usr/bin/env python
"""Build the leave-one-dataset-out SOC cross-dataset split manifests (the
headline generalization experiment) for LG 18650HG2 vs Panasonic 18650PF.

The splitting UNIT is the profile (one drive cycle). Each dataset here is a
single physical cell, so leave-one-dataset-out is equivalently cell-wise and
profile-wise -- no profile ever appears in both train and test, and no
timesteps are shuffled. Two folds are produced:

    fold A: train = Panasonic, test = LG   (hold out LG_HG2)
    fold B: train = LG,        test = Panasonic (hold out PANASONIC_18650PF)

Manifests are written under runs/soc_cross_dataset/ via utils.io.save_manifest.

Usage: python scripts/build_crossdataset_splits.py
"""
from __future__ import annotations

from pathlib import Path

from battery_bench.splits import leave_one_group_out
from battery_bench.utils import read_table, save_manifest

PROC = Path("data/processed")
RUNS = Path("runs/soc_cross_dataset")
# Fold name keyed by which dataset is the TEST set:
#   fold A: train = Panasonic, test = LG
#   fold B: train = LG,        test = Panasonic
FOLD_NAMES = {"LG_HG2": "A", "PANASONIC_18650PF": "B"}


def main() -> int:
    soc = read_table(PROC / "soc_timeseries.parquet")

    # Profile-level metadata frame: one row per profile_id. nominal_temperature_C
    # and drive_cycle are constant within a profile (folded in by the loaders).
    prof_meta = (
        soc.groupby("profile_id", as_index=False)
        .agg(dataset=("dataset", "first"),
             nominal_temperature_C=("nominal_temperature_C", "first"),
             drive_cycle=("drive_cycle", "first"))
    )
    # The generic splitter keys on a "cell_id" column; the splitting unit for
    # this SOC experiment is the profile, so cell_id := profile_id.
    split_input = prof_meta.assign(cell_id=prof_meta["profile_id"])

    print(f"profiles: {len(prof_meta)}  "
          f"({(prof_meta['dataset'] == 'LG_HG2').sum()} LG / "
          f"{(prof_meta['dataset'] == 'PANASONIC_18650PF').sum()} Panasonic)\n")

    for test_dataset, split in leave_one_group_out(split_input, "dataset"):
        train_profiles = split["train_cells"]
        test_profiles = split["test_cells"]
        train_dataset = next(d for d in FOLD_NAMES if d != test_dataset)
        fold = FOLD_NAMES[test_dataset]

        assert not (set(train_profiles) & set(test_profiles)), "LEAKAGE: profile in both splits"
        assert set(prof_meta.loc[prof_meta["profile_id"].isin(test_profiles), "dataset"]) == {test_dataset}

        manifest = {
            "experiment": "soc_cross_dataset",
            "fold": fold,
            "group_by": "dataset",
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            "n_train_profiles": len(train_profiles),
            "n_test_profiles": len(test_profiles),
            "train_profiles": train_profiles,
            "test_profiles": test_profiles,
        }
        out = RUNS / f"fold_{fold}_test_{test_dataset}.json"
        save_manifest(manifest, out)
        print(f"fold {fold}: train={train_dataset} ({len(train_profiles)} profiles) | "
              f"test={test_dataset} ({len(test_profiles)} profiles)  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
