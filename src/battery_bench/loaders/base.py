"""Loader interface. Every dataset loader subclasses this and returns canonical
dataframes validated against schema.py. Implement the three methods; the
harness handles everything downstream.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from battery_bench import schema


class BaseLoader(ABC):
    dataset_name: str = "OVERRIDE_ME"

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        if not self.raw_dir.exists():
            raise FileNotFoundError(self.raw_dir)

    @abstractmethod
    def load_cells(self) -> pd.DataFrame:
        """Return the `cells` metadata table (one row per cell)."""

    def load_soc_timeseries(self) -> pd.DataFrame:
        """Return the SOC timeseries table, or raise NotImplementedError if this
        dataset is SOH-only (e.g. NASA/CALCE/Oxford cycle-aging)."""
        raise NotImplementedError(f"{self.dataset_name} has no SOC timeseries")

    def load_soh_cycles(self) -> pd.DataFrame:
        """Return the SOH per-cycle table, or raise NotImplementedError if this
        dataset is SOC-only (e.g. LG HG2 / Panasonic drive cycles)."""
        raise NotImplementedError(f"{self.dataset_name} has no SOH cycles")

    # -- validated wrappers the pipeline actually calls --
    def cells(self) -> pd.DataFrame:
        return schema.validate_cells(self.load_cells())

    def soc(self) -> pd.DataFrame:
        return schema.validate_soc_timeseries(self.load_soc_timeseries())

    def soh(self) -> pd.DataFrame:
        return schema.validate_soh_cycles(self.load_soh_cycles())
