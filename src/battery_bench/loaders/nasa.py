"""NASA PCoE loader (SOH/RUL).  *** STUB — implement against real .mat files ***

This is the canonical example to hand to Claude Code together with one real
.mat file. The NASA PCoE .mat files are deeply nested MATLAB structs
(B0005.cycle(i).type / .data.Capacity etc.), so the field paths below are a
STARTING GUESS and MUST be verified against an actual file. A good ground-truth
check: B0005 initial capacity is ~1.85 Ah and EOL (~70%) is reached near
cycle ~120. If your loader disagrees, the field mapping is wrong.

Recommended workflow:
  1. Load one file with scipy.io.loadmat(squeeze_me=True, struct_as_record=False)
  2. Print the structure; confirm where capacity per discharge cycle lives.
  3. Fill in _parse_one_cell so it emits the SOH_CYCLES_COLUMNS contract.
  4. Run scripts/validate_processed.py to confirm schema + sanity checks pass.
"""
from __future__ import annotations

import pandas as pd

from .base import BaseLoader


class NasaPcoeLoader(BaseLoader):
    dataset_name = "NASA_PCOE"

    # Map filename stem -> (chemistry, nominal_capacity_Ah). NASA 18650 are LCO.
    CELL_SPECS = {
        "B0005": ("LCO", 2.0),
        "B0006": ("LCO", 2.0),
        "B0007": ("LCO", 2.0),
        "B0018": ("LCO", 2.0),
    }

    def load_cells(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Implement: for each .mat file, compute initial_capacity_Ah from the "
            "first valid discharge cycle and emit one row per cell with columns "
            f"matching schema.CELLS_COLUMNS."
        )

    def load_soh_cycles(self) -> pd.DataFrame:
        raise NotImplementedError(
            "Implement _parse_one_cell(file) -> DataFrame with columns: "
            "dataset, cell_id, cycle_index, capacity_Ah, soh. "
            "soh = capacity_Ah / initial_capacity_Ah (initial from load_cells)."
        )

    # def _parse_one_cell(self, mat_path) -> pd.DataFrame:
    #     ...  # Claude Code fills this in against the real struct layout
