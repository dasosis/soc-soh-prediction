"""Copy-me template for a new dataset loader. Rename the class & dataset_name,
then implement whichever of load_cells / load_soc_timeseries / load_soh_cycles
this dataset supports. Delete the methods that don't apply.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseLoader


class TemplateLoader(BaseLoader):
    dataset_name = "RENAME_ME"

    def load_cells(self) -> pd.DataFrame:
        raise NotImplementedError

    # SOC datasets (LG HG2, Panasonic) implement this:
    # def load_soc_timeseries(self) -> pd.DataFrame: ...

    # SOH datasets (CALCE, Oxford, Sandia) implement this:
    # def load_soh_cycles(self) -> pd.DataFrame: ...
