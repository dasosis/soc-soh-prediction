#!/usr/bin/env python
"""Validate processed parquet tables against the schema + sanity checks.
Run after a loader writes data/processed/* to catch contract violations early.

Usage: python scripts/validate_processed.py data/processed
"""
import sys
from pathlib import Path

from battery_bench import schema
from battery_bench.utils import read_table


def main(proc_dir: str) -> int:
    proc = Path(proc_dir)
    checks = {
        "cells.parquet": schema.validate_cells,
        "soc_timeseries.parquet": schema.validate_soc_timeseries,
        "soh_cycles.parquet": schema.validate_soh_cycles,
    }
    ok = True
    for fname, validator in checks.items():
        fp = proc / fname
        if not fp.exists():
            print(f"skip   {fname} (not present)")
            continue
        try:
            df = read_table(fp)
            validator(df)
            print(f"PASS   {fname}  ({len(df):,} rows)")
        except Exception as e:  # noqa
            ok = False
            print(f"FAIL   {fname}: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data/processed"))
