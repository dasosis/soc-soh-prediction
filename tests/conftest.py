"""Synthetic canonical-format fixtures so the harness is testable without any
real dataset downloaded. These mimic the schema, not real battery physics.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cells_df():
    return pd.DataFrame({
        "dataset": ["NASA_PCOE", "NASA_PCOE", "CALCE", "OXFORD"],
        "cell_id": ["NASA_B0005", "NASA_B0006", "CALCE_CS2_35", "OX_Cell1"],
        "chemistry": ["LCO", "LCO", "LCO", "LCO"],
        "manufacturer": ["LG", "LG", "Maryland", "Kokam"],
        "form_factor": ["18650", "18650", "prismatic", "pouch"],
        "nominal_capacity_Ah": [2.0, 2.0, 1.1, 0.74],
        "initial_capacity_Ah": [1.86, 1.85, 1.10, 0.74],
    }).astype({"dataset": "string", "cell_id": "string", "chemistry": "string",
               "manufacturer": "string", "form_factor": "string"})


@pytest.fixture
def soh_df():
    rows = []
    for cell, cap0 in [("NASA_B0005", 1.86), ("NASA_B0006", 1.85)]:
        for cyc in range(1, 21):
            cap = cap0 * (1.0 - 0.01 * cyc)  # smooth fade
            rows.append(["NASA_PCOE", cell, cyc, cap, cap / cap0])
    return pd.DataFrame(rows, columns=[
        "dataset", "cell_id", "cycle_index", "capacity_Ah", "soh"
    ]).astype({"dataset": "string", "cell_id": "string"})


@pytest.fixture
def soc_df():
    rows = []
    rng = np.random.default_rng(0)
    for prof, dc in [("P1", "US06"), ("P2", "UDDS")]:
        soc = 1.0
        for t in range(50):
            soc = max(0.0, soc - 0.02)
            rows.append(["LG_HG2", "LGHG2_C1", prof, 25.0, dc, t,
                         float(t), 3.7 + rng.normal(0, 0.01),
                         -1.5, 25.0 + rng.normal(0, 0.1), soc])
    return pd.DataFrame(rows, columns=[
        "dataset", "cell_id", "profile_id", "nominal_temperature_C", "drive_cycle",
        "timestep", "time_s", "voltage_V", "current_A", "temperature_C", "soc",
    ]).astype({"dataset": "string", "cell_id": "string", "profile_id": "string",
               "drive_cycle": "string"})
