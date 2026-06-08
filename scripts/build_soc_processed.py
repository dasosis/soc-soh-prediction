#!/usr/bin/env python
"""Build the canonical SOC processed tables for the cross-dataset experiment
(LG 18650HG2 + Panasonic 18650PF only) and print a per-dataset summary.

Usage:
    python scripts/build_soc_processed.py
Writes:
    data/processed/cells.parquet
    data/processed/soc_timeseries.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from battery_bench import schema
from battery_bench.loaders.lg import LgHg2Loader
from battery_bench.loaders.panasonic import Panasonic18650PFLoader
from battery_bench.utils import write_table

RAW = Path("data/raw")
OUT = Path("data/processed")


def main() -> int:
    loaders = [
        LgHg2Loader(RAW / "LG18650HG2"),
        Panasonic18650PFLoader(RAW / "Panasonic-18650PF"),
    ]

    cells_frames, soc_frames = [], []
    for ld in loaders:
        t0 = time.time()
        cells_frames.append(ld.cells())          # validated
        soc = ld.soc()                            # validated (schema range/monotonic)
        soc_frames.append(soc)
        print(f"[{ld.dataset_name}] {soc['profile_id'].nunique()} profiles, "
              f"{len(soc):,} timesteps in {time.time()-t0:.1f}s")
        if ld.soc_overshoots:
            print(f"  note: {len(ld.soc_overshoots)} profile(s) had a sub-0.1% full-charge "
                  f"regen overshoot (anchored to SOC=1.0, not clipped):")
            for pid, raw_max in ld.soc_overshoots:
                print(f"    {pid}: unanchored max SOC = {raw_max:.5f}")

    cells = pd.concat(cells_frames, ignore_index=True)
    soc = pd.concat(soc_frames, ignore_index=True)
    schema.validate_cells(cells)
    schema.validate_soc_timeseries(soc)

    write_table(cells, OUT / "cells.parquet")
    write_table(soc, OUT / "soc_timeseries.parquet")
    print(f"\nwrote {OUT/'cells.parquet'} ({len(cells)} rows) and "
          f"{OUT/'soc_timeseries.parquet'} ({len(soc):,} rows)")

    # ---- summary table ----
    print("\n================ SOC PROCESSED SUMMARY ================")
    for ds, g in soc.groupby("dataset", sort=True):
        temps = sorted(g["nominal_temperature_C"].unique())
        dcs = sorted(g["drive_cycle"].unique())
        med_I = g["current_A"].median()
        print(f"\n{ds}")
        print(f"  profiles      : {g['profile_id'].nunique()}")
        print(f"  timesteps(1Hz): {len(g):,}")
        print(f"  temperatures  : {temps}")
        print(f"  drive cycles  : {dcs}")
        print(f"  SOC range     : {g['soc'].min():.4f} .. {g['soc'].max():.4f}")
        print(f"  voltage range : {g['voltage_V'].min():.3f} .. {g['voltage_V'].max():.3f} V")
        print(f"  current sanity: median I = {med_I:.3f} A "
              f"({'discharge-negative OK' if med_I < 0 else 'WARN: not net-discharge'})")
    print(f"\ncells:\n{cells.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
