"""LG 18650HG2 loader (SOC drive cycles).

McMaster University dataset (P. Kollmeyer): one brand-new 3.0 Ah LG HG2 NMC
18650 cell, tested at six ambient temperatures (40, 25, 10, 0, -10, -20 degC).
SOC-only -> emits ``cells`` + ``soc_timeseries`` (no soh_cycles).

RAW FORMAT (per-temperature CSV, Digatron export):
  * 28 metadata lines, then a header row, then a units row, then data.
  * columns used: Prog Time (HH:MM:SS.fff), Voltage[V], Current[A],
    Temperature[C], Capacity[Ah] (= accumulated amp-hours, reset to ~0 after
    each charge/test/drive cycle).

ASSUMPTIONS (verified against raw files in Phase 1; see _kollmeyer.py):
  * current is DISCHARGE-NEGATIVE already -> passed through unflipped.
  * each drive-cycle file's `Capacity` counter resets to ~0 at full charge, so
    every profile starts at SOC = 1.0.
  * SOC(t) = 1.0 + Capacity(t) / C_ref.  C_ref defaults to the rated 3.0 Ah
    (configurable via `soc_reference_Ah`); empirically this keeps SOC in [0,1]
    at every temperature. No clamping is performed.
  * native logging is ~10 Hz with variable rate -> linearly resampled to a 1 Hz
    integer-second grid so a fixed window = fixed physical duration across the
    LG and Panasonic datasets.

ONE PROFILE per (temperature folder x drive-cycle file). Drive cycles kept:
UDDS, US06, LA92, HWFET (mapped to themselves) and Mixed1..8 (mapped to MIXED,
original name retained in profile_id). Capacity/Charge/HPPC/C20/Dis_* files are
ignored. Cells below 10 degC (n10/n20) are kept as normal profiles with the
correct nominal_temperature_C; below 10 degC the tester uses a reduced regen
current limit.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import _kollmeyer as km
from .base import BaseLoader

_LG_HEADER_ROW = 28  # 0-based line index of the column-name row in the CSV
_USE_COLS = ["Prog Time", "Voltage", "Current", "Temperature", "Capacity"]


def _prog_time_to_seconds(values) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    for i, v in enumerate(values):
        h, m, s = str(v).split(":")
        out[i] = int(h) * 3600 + int(m) * 60 + float(s)
    return out


class LgHg2Loader(BaseLoader):
    dataset_name = "LG_HG2"

    TEMP_FOLDERS = ["0degC", "10degC", "25degC", "40degC", "n10degC", "n20degC"]
    CELL_ID = "LG_HG2_C1"

    def __init__(
        self,
        raw_dir,
        soc_reference_Ah: float | None = None,
        chemistry: str = "NMC",
        manufacturer: str = "LG",
        form_factor: str = "18650",
        nominal_capacity_Ah: float = 3.0,
        verbose: bool = False,
    ):
        super().__init__(raw_dir)
        self.nominal_capacity_Ah = float(nominal_capacity_Ah)
        # SOC normalization reference; default rated capacity.
        self.soc_reference_Ah = float(soc_reference_Ah) if soc_reference_Ah else self.nominal_capacity_Ah
        self.chemistry = chemistry
        self.manufacturer = manufacturer
        self.form_factor = form_factor
        self.verbose = verbose
        # (profile_id, unanchored_max_soc) for profiles whose Ah counter drifted
        # positive at the full-charge start (regen artifact, anchored away).
        self.soc_overshoots: list[tuple[str, float]] = []

    # -- raw file handling --
    def _read_csv(self, path: Path) -> dict[str, np.ndarray]:
        df = pd.read_csv(path, skiprows=_LG_HEADER_ROW, header=0, usecols=_USE_COLS, low_memory=False)
        df = df.iloc[1:].reset_index(drop=True)  # drop the units row ([V],[A],...)
        num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        cols = {
            "voltage_V": num("Voltage"),
            "current_A": num("Current"),
            "temperature_C": num("Temperature"),
            "ah": num("Capacity"),
        }
        t = _prog_time_to_seconds(df["Prog Time"].to_numpy())
        # Drop occasional logging dropouts (valid timestamp, empty channels).
        finite = np.isfinite(t)
        for v in cols.values():
            finite &= np.isfinite(v)
        t = t[finite]
        t = t - t[0]
        return {"time_s": t, **{k: v[finite] for k, v in cols.items()}}

    def _drive_cycle_files(self):
        """Yield (folder_name, label, drive_cycle, path) for every kept file."""
        for folder in self.TEMP_FOLDERS:
            fdir = self.raw_dir / folder
            if not fdir.is_dir():
                continue
            for path in sorted(fdir.glob("*.csv")):
                label = re.sub(r"^\d+_", "", path.stem)  # "551_UDDS" -> "UDDS"
                drive_cycle, reason = km.classify_drive_cycle(path.stem)
                if drive_cycle is None:
                    continue
                yield folder, label, drive_cycle, path

    def _measure_initial_capacity(self) -> float:
        """Measured 1C reference capacity from the 25degC Cap_1C file (Ah swing).
        Falls back to nominal if the file is absent or implausible (<50% nominal)."""
        cands = list((self.raw_dir / "25degC").glob("*Cap_1C*.csv")) if (self.raw_dir / "25degC").is_dir() else []
        if cands:
            try:
                cap = self._read_csv(cands[0])["ah"]
                measured = float(np.nanmax(cap) - np.nanmin(cap))
                if measured > 0.5 * self.nominal_capacity_Ah:
                    return measured
            except Exception:
                pass
        return self.nominal_capacity_Ah

    # -- BaseLoader interface --
    def load_cells(self) -> pd.DataFrame:
        init = self._measure_initial_capacity()
        return pd.DataFrame([{
            "dataset": self.dataset_name,
            "cell_id": self.CELL_ID,
            "chemistry": self.chemistry,
            "manufacturer": self.manufacturer,
            "form_factor": self.form_factor,
            "nominal_capacity_Ah": self.nominal_capacity_Ah,
            "initial_capacity_Ah": init,
        }]).astype({
            "dataset": "string", "cell_id": "string", "chemistry": "string",
            "manufacturer": "string", "form_factor": "string",
            "nominal_capacity_Ah": "float64", "initial_capacity_Ah": "float64",
        })

    def load_soc_timeseries(self) -> pd.DataFrame:
        frames = []
        for folder, label, drive_cycle, path in self._drive_cycle_files():
            raw = self._read_csv(path)
            grid, ch = km.resample_to_1hz(raw["time_s"], {
                "voltage_V": raw["voltage_V"], "current_A": raw["current_A"],
                "temperature_C": raw["temperature_C"], "ah": raw["ah"],
            })
            soc = km.coulomb_count_soc(ch["ah"], self.soc_reference_Ah)
            if soc.max() - soc.min() < km.MIN_SOC_SWING:
                if self.verbose:
                    print(f"  [LG] skip {folder}/{label}: flat SOC (rest/no-discharge file)")
                continue
            overshoot = float(np.nanmax(ch["ah"])) / self.soc_reference_Ah
            if overshoot > 1e-5:
                self.soc_overshoots.append((f"{self.dataset_name}__{folder}__{label}", 1.0 + overshoot))
            if np.nanmean(ch["current_A"]) >= 0:  # net discharge => mean current < 0
                raise ValueError(f"{path.name}: expected net-discharge (mean I<0); sign convention wrong")
            n = len(grid)
            profile_id = f"{self.dataset_name}__{folder}__{label}"
            frames.append(pd.DataFrame({
                "dataset": self.dataset_name,
                "cell_id": self.CELL_ID,
                "profile_id": profile_id,
                "nominal_temperature_C": km.parse_nominal_temperature_c(folder),
                "drive_cycle": drive_cycle,
                "timestep": np.arange(n, dtype=np.int64),
                "time_s": grid.astype(float),
                "voltage_V": ch["voltage_V"],
                "current_A": ch["current_A"],
                "temperature_C": ch["temperature_C"],
                "soc": soc,
            }))
            if self.verbose:
                print(f"  [LG] {profile_id}: {n} steps, SOC {soc.min():.3f}..{soc.max():.3f}")
        if not frames:
            raise FileNotFoundError(f"no LG drive-cycle CSVs found under {self.raw_dir}")
        out = pd.concat(frames, ignore_index=True)
        return out.astype({
            "dataset": "string", "cell_id": "string", "profile_id": "string",
            "drive_cycle": "string", "nominal_temperature_C": "float64",
            "timestep": "int64", "time_s": "float64", "voltage_V": "float64",
            "current_A": "float64", "temperature_C": "float64", "soc": "float64",
        })
