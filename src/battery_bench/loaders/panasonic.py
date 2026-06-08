"""Panasonic 18650PF loader (SOC drive cycles).

University of Wisconsin-Madison dataset (P. Kollmeyer): one brand-new 2.9 Ah
Panasonic 18650PF NCA cell, tested at 25, 10, 0, -10, -20 degC plus the
temperature-rise ("Trise") variants. SOC-only -> emits ``cells`` +
``soc_timeseries`` (no soh_cycles).

RAW FORMAT (one MATLAB .mat per drive cycle, struct ``meas``):
  meas.Time (s, from 0), meas.Voltage, meas.Current, meas.Battery_Temp_degC,
  meas.Ah (accumulated amp-hours, reset to ~0 before each test), plus Wh /
  Power / Chamber_Temp_degC / TimeStamp (unused).

ASSUMPTIONS (verified against raw files in Phase 1; see _kollmeyer.py):
  * current is DISCHARGE-NEGATIVE already -> passed through unflipped.
  * meas.Ah resets to ~0 at full charge, so each profile starts at SOC = 1.0.
  * SOC(t) = 1.0 + Ah(t) / C_ref.  C_ref defaults to the rated 2.9 Ah
    (configurable via `soc_reference_Ah`); empirically this keeps SOC in [0,1]
    at every temperature (cold cycles terminate at a fixed DOD). No clamping.
  * native logging ~10 Hz with variable rate -> linearly resampled to a 1 Hz
    integer-second grid (matches the LG loader so windows are comparable).

ONE PROFILE per drive-cycle .mat file. Files are discovered by the
``*Pan18650PF.mat`` suffix (charges, pauses, dis5_10p, EIS and the C20 test do
not carry that suffix and are skipped). HPPC "5pulse" files carry the suffix but
have no drive-cycle token and are skipped. Contiguous multi-cycle files (e.g.
``US06_HWFET_UDDS_LA92``) and the "with pause"/"1to4" files are skipped because
they duplicate the split single-cycle files and/or contain charge segments that
break the single-discharge SOC ramp (see _kollmeyer.classify_drive_cycle).

Cold cells (n10/n20/0 degC and the Trise variants) are KEPT with the correct
nominal_temperature_C taken from the temperature folder. Below 10 degC the test
protocol includes NO regen current.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from . import _kollmeyer as km
from .base import BaseLoader


def _label_from_stem(stem: str) -> str:
    """'06-07-17_08.39 n10degC_trise_Cycle_1_Pan18650PF' -> 'Cycle_1'."""
    core = stem.split(" ")[-1]
    core = re.sub(r"_?Pan18650PF$", "", core, flags=re.IGNORECASE)
    core = re.sub(r"^(n|-)?\d+degC_?", "", core, flags=re.IGNORECASE)
    core = re.sub(r"^trise_?", "", core, flags=re.IGNORECASE)
    return core or "drive"


class Panasonic18650PFLoader(BaseLoader):
    dataset_name = "PANASONIC_18650PF"
    CELL_ID = "PANASONIC_18650PF_C1"

    def __init__(
        self,
        raw_dir,
        soc_reference_Ah: float | None = None,
        chemistry: str = "NCA",
        manufacturer: str = "Panasonic",
        form_factor: str = "18650",
        nominal_capacity_Ah: float = 2.9,
        verbose: bool = False,
    ):
        super().__init__(raw_dir)
        self.nominal_capacity_Ah = float(nominal_capacity_Ah)
        self.soc_reference_Ah = float(soc_reference_Ah) if soc_reference_Ah else self.nominal_capacity_Ah
        self.chemistry = chemistry
        self.manufacturer = manufacturer
        self.form_factor = form_factor
        self.verbose = verbose
        # (profile_id, unanchored_max_soc) for profiles whose Ah counter drifted
        # positive at the full-charge start (regen artifact, anchored away).
        self.soc_overshoots: list[tuple[str, float]] = []

    # -- raw file handling --
    @staticmethod
    def _read_mat(path: Path) -> dict[str, np.ndarray]:
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        key = next(k for k in m if not k.startswith("__"))
        meas = m[key]
        t = np.asarray(meas.Time, dtype=float).ravel()
        t = t - t[0]
        return {
            "time_s": t,
            "voltage_V": np.asarray(meas.Voltage, dtype=float).ravel(),
            "current_A": np.asarray(meas.Current, dtype=float).ravel(),
            "temperature_C": np.asarray(meas.Battery_Temp_degC, dtype=float).ravel(),
            "ah": np.asarray(meas.Ah, dtype=float).ravel(),
        }

    def _temp_folder_for(self, path: Path) -> str:
        """Nearest ancestor folder name that encodes a temperature setpoint."""
        for parent in path.relative_to(self.raw_dir).parents:
            name = parent.name
            if re.search(r"(n|-)?\d+\s*degc", name.lower()):
                return name
        raise ValueError(f"no temperature folder in path {path}")

    def _drive_cycle_files(self):
        """Yield (folder_name, label, drive_cycle, path) for every kept file."""
        for path in sorted(self.raw_dir.rglob("*Pan18650PF.mat")):
            label = _label_from_stem(path.stem)
            drive_cycle, reason = km.classify_drive_cycle(label)
            if drive_cycle is None:
                continue
            yield self._temp_folder_for(path), label, drive_cycle, path

    def _measure_initial_capacity(self) -> float:
        """Measured 1C reference capacity from a 25degC start-of-tests 1C
        discharge (Ah swing). Falls back to nominal if absent/implausible."""
        cands = sorted(self.raw_dir.rglob("*Dis1C_1.mat"))
        cands = [c for c in cands if "start_of_tests" in str(c).lower()] or cands
        for c in cands:
            try:
                ah = self._read_mat(c)["ah"]
                measured = float(np.nanmax(ah) - np.nanmin(ah))
                if measured > 0.5 * self.nominal_capacity_Ah:
                    return measured
            except Exception:
                continue
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
            raw = self._read_mat(path)
            grid, ch = km.resample_to_1hz(raw["time_s"], {
                "voltage_V": raw["voltage_V"], "current_A": raw["current_A"],
                "temperature_C": raw["temperature_C"], "ah": raw["ah"],
            })
            soc = km.coulomb_count_soc(ch["ah"], self.soc_reference_Ah)
            if soc.max() - soc.min() < km.MIN_SOC_SWING:
                if self.verbose:
                    print(f"  [PAN] skip {folder}/{label}: flat SOC (rest/no-discharge file)")
                continue
            folder_label = folder.replace(" ", "_")
            profile_id = f"{self.dataset_name}__{folder_label}__{label}"
            overshoot = float(np.nanmax(ch["ah"])) / self.soc_reference_Ah
            if overshoot > 1e-5:
                self.soc_overshoots.append((profile_id, 1.0 + overshoot))
            if np.nanmean(ch["current_A"]) >= 0:  # net discharge => mean current < 0
                raise ValueError(f"{path.name}: expected net-discharge (mean I<0); sign convention wrong")
            n = len(grid)
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
                print(f"  [PAN] {profile_id}: {n} steps, SOC {soc.min():.3f}..{soc.max():.3f}")
        if not frames:
            raise FileNotFoundError(f"no Panasonic drive-cycle .mat files under {self.raw_dir}")
        out = pd.concat(frames, ignore_index=True)
        return out.astype({
            "dataset": "string", "cell_id": "string", "profile_id": "string",
            "drive_cycle": "string", "nominal_temperature_C": "float64",
            "timestep": "int64", "time_s": "float64", "voltage_V": "float64",
            "current_A": "float64", "temperature_C": "float64", "soc": "float64",
        })
