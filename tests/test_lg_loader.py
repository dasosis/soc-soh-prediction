"""LG 18650HG2 loader test on a tiny synthetic Digatron-format CSV."""
import numpy as np
import pandas as pd
import pytest

from battery_bench.loaders.lg import LgHg2Loader

_HEADER = (
    "Time Stamp,Step,Status,Prog Time,Step Time,Cycle,Cycle Level,Procedure,"
    "Voltage,Current,Temperature,Capacity,WhAccu,Cnt,"
)
_UNITS = ",,,,,,,,[V],[A],[C],[Ah],[Wh],[Cnt],"


def _fmt(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def _write_lg_csv(path, n=120, dt=0.5, cap_end=-0.15, current=-3.0):
    # 28 metadata lines, then header, units, then data (discharge from full).
    lines = [f"meta{i},x" for i in range(28)]
    lines += [_HEADER, _UNITS]
    for k in range(n):
        t = k * dt
        cap = cap_end * (k / (n - 1))          # 0 -> cap_end (negative)
        v = 4.2 + (3.9 - 4.2) * (k / (n - 1))  # 4.2 -> 3.9
        lines.append(
            f"1/1/2020 12:00:00 AM,25,TABLE,{_fmt(t)},{_fmt(t)},1,1,SYN,"
            f"{v:.5f},{current:.5f},25.0,{cap:.5f},0.0,3.0,"
        )
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def lg_dir(tmp_path):
    folder = tmp_path / "25degC"
    folder.mkdir()
    _write_lg_csv(folder / "001_UDDS.csv")
    _write_lg_csv(folder / "002_Mixed1.csv")  # -> drive_cycle MIXED
    return tmp_path


def test_lg_emits_valid_schema(lg_dir):
    ld = LgHg2Loader(lg_dir)
    cells = ld.cells()   # validates schema internally
    soc = ld.soc()       # validates schema internally
    assert list(cells["cell_id"]) == ["LG_HG2_C1"]
    assert (cells["initial_capacity_Ah"] > 0).all()
    assert set(soc["drive_cycle"]) == {"UDDS", "MIXED"}


def test_lg_soc_in_range_and_starts_full(lg_dir):
    soc = LgHg2Loader(lg_dir).soc()
    assert soc["soc"].between(0.0, 1.0).all()
    first = soc.sort_values("timestep").groupby("profile_id").first()
    assert np.allclose(first["soc"], 1.0, atol=1e-6)


def test_lg_current_discharge_negative(lg_dir):
    soc = LgHg2Loader(lg_dir).soc()
    assert (soc["current_A"] <= 0).all()
    assert soc["current_A"].mean() < 0


def test_lg_profile_boundaries_intact(lg_dir):
    soc = LgHg2Loader(lg_dir).soc()
    assert soc["profile_id"].nunique() == 2
    for _, g in soc.groupby("profile_id"):
        assert g["timestep"].iloc[0] == 0
        assert g["timestep"].is_monotonic_increasing
        assert g["nominal_temperature_C"].nunique() == 1
