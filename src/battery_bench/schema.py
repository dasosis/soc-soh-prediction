"""Canonical schema contract for the battery SOC/SOH benchmark.

Every per-dataset loader MUST emit dataframes conforming to these column
contracts. Models, splitters, and metrics only ever see the canonical form,
so they are dataset-agnostic.

UNIT & SIGN CONVENTIONS (fixed once, here, forever):
  - voltage_V         : volts
  - current_A         : amperes, DISCHARGE NEGATIVE (current leaving the cell
                        into the load is negative; charge is positive)
  - temperature_C     : degrees Celsius (measured cell/surface temperature)
  - nominal_temperature_C : the nominal ambient setpoint of the test run
  - capacity_Ah       : amp-hours
  - soc               : fraction in [0, 1]
  - soh               : fraction, capacity_Ah / initial_capacity_Ah  (NOT nominal)
                        Switch the reference in config if you prefer nominal.

LABEL DEFINITIONS:
  - SOC label is computed by the loader (typically coulomb counting with a
    known initial SOC). The loader is responsible for it; the harness trusts it.
  - SOH label is capacity_Ah / initial_capacity_Ah, where initial_capacity_Ah
    is the first valid measured capacity of that cell (stored in `cells`).
  - RUL is NOT stored. It is derived in the pipeline from an EOL threshold
    given in config (e.g. EOL = 0.80), so the threshold is never baked into data.
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

CELLS_COLUMNS = {
    "dataset": "string",            # e.g. "NASA_PCOE", "LG_HG2"
    "cell_id": "string",            # globally unique, e.g. "NASA_B0005"
    "chemistry": "string",          # e.g. "NMC", "NCA", "LFP", "LCO"
    "manufacturer": "string",
    "form_factor": "string",        # e.g. "18650", "pouch", "prismatic"
    "nominal_capacity_Ah": "float64",
    "initial_capacity_Ah": "float64",
}

SOC_TIMESERIES_COLUMNS = {
    "dataset": "string",
    "cell_id": "string",
    "profile_id": "string",         # one drive cycle / discharge run, unique
    "nominal_temperature_C": "float64",
    "drive_cycle": "string",        # e.g. "US06", "UDDS", "MIXED"
    "timestep": "int64",            # 0..N-1 within profile, monotonic
    "time_s": "float64",
    "voltage_V": "float64",
    "current_A": "float64",         # discharge negative
    "temperature_C": "float64",
    "soc": "float64",               # label, [0, 1]
}

SOH_CYCLES_COLUMNS = {
    "dataset": "string",
    "cell_id": "string",
    "cycle_index": "int64",         # 1..M, monotonic within cell
    "capacity_Ah": "float64",
    "soh": "float64",               # capacity_Ah / initial_capacity_Ah
    # Optional health-indicator features may be added with an "hi_" prefix,
    # e.g. hi_cc_time_s, hi_dv_dq_peak. They are ignored by validation.
}

RAW_SOC_FEATURES = ["voltage_V", "current_A", "temperature_C"]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, contract: dict, name: str) -> None:
    missing = [c for c in contract if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}")


def validate_cells(df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(df, CELLS_COLUMNS, "cells")
    if df["cell_id"].duplicated().any():
        dups = df.loc[df["cell_id"].duplicated(), "cell_id"].tolist()
        raise ValueError(f"[cells] duplicate cell_id: {dups}")
    if (df["initial_capacity_Ah"] <= 0).any():
        raise ValueError("[cells] initial_capacity_Ah must be > 0")
    return df


def validate_soc_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(df, SOC_TIMESERIES_COLUMNS, "soc_timeseries")
    if not df["soc"].between(0.0, 1.0).all():
        bad = df.loc[~df["soc"].between(0.0, 1.0), "soc"]
        raise ValueError(
            f"[soc_timeseries] soc out of [0,1]; "
            f"min={bad.min()}, max={bad.max()}. Did the loader use % not fraction?"
        )
    # timestep must be monotonic non-decreasing within each profile
    g = df.groupby("profile_id")["timestep"]
    if not g.apply(lambda s: s.is_monotonic_increasing).all():
        raise ValueError("[soc_timeseries] timestep not monotonic within a profile")
    return df


def validate_soh_cycles(df: pd.DataFrame) -> pd.DataFrame:
    _check_columns(df, SOH_CYCLES_COLUMNS, "soh_cycles")
    if (df["soh"] <= 0).any() or (df["soh"] > 1.2).any():
        raise ValueError(
            "[soh_cycles] soh outside plausible (0, 1.2]; check the reference capacity."
        )
    g = df.groupby("cell_id")["cycle_index"]
    if not g.apply(lambda s: s.is_monotonic_increasing).all():
        raise ValueError("[soh_cycles] cycle_index not monotonic within a cell")
    return df
