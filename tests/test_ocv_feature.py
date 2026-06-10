"""Stage-3 OCV feature validation: recovery, monotone inverse + clamping,
chemistry separation, and window-constructor wiring."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from battery_bench.preprocess.ocv_feature import load_ocv_table, soc_from_voltage
from battery_bench.preprocess.window import make_windows

FEATS = ["voltage_V", "current_A", "temperature_C"]
_RAW = Path("data/raw")


def _cc_soc(t, I, rated):
    q = np.concatenate(([0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t)))) / 3600.0
    return 1.0 + (q - np.nanmax(q)) / rated


# ---- (a) recovery on each chemistry's own C/20 file -----------------------
# NCA (Panasonic) has ~100 mV charge/discharge hysteresis at C/20, so its
# recovery floor is higher than NMC's; thresholds reflect that physics.
@pytest.mark.parametrize("chem,glob,rated,thresh", [
    ("LG_HG2", "LG18650HG2/25degC/*C20DisCh*.csv", 3.0, 0.03),
    ("PANASONIC_18650PF", "Panasonic-18650PF/**/*C20 OCV*25dC*.mat", 2.9, 0.06),
])
def test_recovery_matches_cc_soc(chem, glob, rated, thresh):
    files = list(_RAW.glob(glob))
    if not files:
        pytest.skip(f"raw file for {chem} not present (gitignored)")
    if chem == "LG_HG2":
        from battery_bench.loaders.lg import LgHg2Loader
        d = LgHg2Loader(_RAW / "LG18650HG2")._read_csv(files[0])
    else:
        from battery_bench.loaders.panasonic import Panasonic18650PFLoader
        d = Panasonic18650PFLoader._read_mat(files[0])
    cc = _cc_soc(d["time_s"], d["current_A"], rated)
    rec, _ = soc_from_voltage(d["voltage_V"], chem)
    mae = float(np.mean(np.abs(rec - cc)))
    assert mae < thresh, f"{chem} recovery MAE {mae:.4f} >= {thresh}"


# ---- (b) monotone inverse + clamping --------------------------------------
@pytest.mark.parametrize("chem", ["LG_HG2", "PANASONIC_18650PF"])
def test_inverse_monotone_and_clamped(chem):
    soc_grid, ocv_grid = load_ocv_table(chem)
    V = np.linspace(ocv_grid[0], ocv_grid[-1], 400)
    soc, clamped = soc_from_voltage(V, chem)
    assert np.all(np.diff(soc) >= -1e-9)                       # monotone non-decreasing
    assert not clamped.any()                                  # in-range -> no clamp

    lo, clo = soc_from_voltage([ocv_grid[0] - 0.3], chem)
    hi, chi = soc_from_voltage([ocv_grid[-1] + 0.3], chem)
    assert clo[0] and chi[0]                                  # both flagged clamped
    assert abs(lo[0] - float(soc_grid.min())) < 1e-9          # below range -> soc floor
    assert abs(hi[0] - 1.0) < 1e-9                            # above range -> 1.0


# ---- (c) the feature actually encodes the chemistry difference ------------
def test_chemistry_separation():
    lg, _ = soc_from_voltage([3.40], "LG_HG2")
    pan, _ = soc_from_voltage([3.40], "PANASONIC_18650PF")
    assert abs(lg[0] - pan[0]) > 0.025                         # non-trivial at 3.40 V
    Vs = np.arange(3.3, 4.0, 0.05)
    sl, _ = soc_from_voltage(Vs, "LG_HG2")
    sp, _ = soc_from_voltage(Vs, "PANASONIC_18650PF")
    assert float(np.max(np.abs(sl - sp))) > 0.05              # clear divergence somewhere


# ---- (d) window-constructor wiring ----------------------------------------
def _toy_df(n=60):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "profile_id": "p1",
        "timestep": np.arange(n),
        "voltage_V": np.linspace(4.15, 3.05, n),              # raw volts, full sweep
        "current_A": -1.5 + rng.normal(0, 0.05, n),
        "temperature_C": 25.0 + rng.normal(0, 0.1, n),
        "soc": np.linspace(1.0, 0.1, n),
    })


def test_window_add_ocv_channel():
    df = _toy_df()
    L = 20
    X_off, y_off, g_off = make_windows(df, FEATS, "soc", length=L, stride=5)
    X_on, y_on, g_on = make_windows(df, FEATS, "soc", length=L, stride=5,
                                    add_ocv_channel=True, ocv_chem="LG_HG2")
    assert X_off.shape[-1] == 3 and X_on.shape[-1] == 4       # extra channel appended
    assert X_on.shape[:2] == X_off.shape[:2]
    # first 3 channels and labels/groups byte-identical to the flag-off call
    assert np.array_equal(X_on[..., :3], X_off)
    assert np.array_equal(y_on, y_off) and np.array_equal(g_on, g_off)
    # 4th channel == elementwise OCV lookup of the raw voltage windows
    expected, _ = soc_from_voltage(X_off[..., 0].ravel(), "LG_HG2")
    assert np.allclose(X_on[..., 3].ravel(), expected.astype(np.float32), atol=1e-5)


def test_window_default_is_unchanged():
    df = _toy_df()
    a = make_windows(df, FEATS, "soc", length=15, stride=3)[0]
    b = make_windows(df, FEATS, "soc", length=15, stride=3, add_ocv_channel=False)[0]
    assert np.array_equal(a, b) and a.shape[-1] == 3
