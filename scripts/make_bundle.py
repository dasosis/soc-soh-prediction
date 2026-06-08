#!/usr/bin/env python
"""Bundle the processed SOC tables + split manifests + SCHEMA.md into a single
portable zip for Colab / JarvisLabs. Raw data is NOT included.

Usage: python scripts/make_bundle.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

OUT = Path("artifacts/soc_crossdataset_bundle.zip")


def main() -> int:
    members = [
        (Path("data/processed/cells.parquet"), "processed/cells.parquet"),
        (Path("data/processed/soc_timeseries.parquet"), "processed/soc_timeseries.parquet"),
        (Path("SCHEMA.md"), "SCHEMA.md"),
    ]
    members += [(p, f"runs/soc_cross_dataset/{p.name}")
                for p in sorted(Path("runs/soc_cross_dataset").glob("*.json"))]

    missing = [str(src) for src, _ in members if not src.exists()]
    if missing:
        raise FileNotFoundError(f"cannot bundle, missing: {missing}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in members:
            z.write(src, arc)

    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  ({size_mb:.2f} MB, {len(members)} files)")
    for _, arc in members:
        print(f"  {arc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
