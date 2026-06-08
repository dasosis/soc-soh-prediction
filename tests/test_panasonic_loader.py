"""Panasonic 18650PF loader test on tiny synthetic .mat files in the real
``meas`` struct layout (written with scipy.io.savemat)."""
import numpy as np
import pytest
from scipy.io import savemat

from battery_bench.loaders.panasonic import Panasonic18650PFLoader


def _write_pan_mat(path, n=120, dt=0.5, ah_end=-0.15, current=-3.0):
    t = np.arange(n) * dt
    frac = np.arange(n) / (n - 1)
    meas = {
        "Time": t,
        "Voltage": 4.2 + (3.9 - 4.2) * frac,
        "Current": np.full(n, current),
        "Battery_Temp_degC": np.full(n, 25.0),
        "Chamber_Temp_degC": np.full(n, 25.0),
        "Ah": ah_end * frac,           # 0 -> ah_end (negative)
        "Wh": np.zeros(n),
        "Power": np.full(n, current * 4.0),
    }
    savemat(path, {"meas": meas})


@pytest.fixture
def pan_dir(tmp_path):
    folder = tmp_path / "25degC" / "Drive cycles"
    folder.mkdir(parents=True)
    _write_pan_mat(folder / "01-01-20_00.00 25degC_US06_Pan18650PF.mat")
    _write_pan_mat(folder / "01-01-20_01.00 25degC_Cycle_1_Pan18650PF.mat")  # -> MIXED
    return tmp_path


def test_pan_emits_valid_schema(pan_dir):
    ld = Panasonic18650PFLoader(pan_dir)
    cells = ld.cells()   # validates schema internally
    soc = ld.soc()       # validates schema internally
    assert list(cells["cell_id"]) == ["PANASONIC_18650PF_C1"]
    assert (cells["initial_capacity_Ah"] > 0).all()
    assert set(soc["drive_cycle"]) == {"US06", "MIXED"}


def test_pan_soc_in_range_and_starts_full(pan_dir):
    soc = Panasonic18650PFLoader(pan_dir).soc()
    assert soc["soc"].between(0.0, 1.0).all()
    first = soc.sort_values("timestep").groupby("profile_id").first()
    assert np.allclose(first["soc"], 1.0, atol=1e-6)


def test_pan_current_discharge_negative(pan_dir):
    soc = Panasonic18650PFLoader(pan_dir).soc()
    assert (soc["current_A"] <= 0).all()
    assert soc["current_A"].mean() < 0


def test_pan_profile_boundaries_intact(pan_dir):
    soc = Panasonic18650PFLoader(pan_dir).soc()
    assert soc["profile_id"].nunique() == 2
    for _, g in soc.groupby("profile_id"):
        assert g["timestep"].iloc[0] == 0
        assert g["timestep"].is_monotonic_increasing
        assert g["nominal_temperature_C"].nunique() == 1
