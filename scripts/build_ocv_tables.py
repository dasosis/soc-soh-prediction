#!/usr/bin/env python
"""Build per-chemistry 25 degC OCV(SOC) tables from the C/20 characterization
files located by scripts/inventory_raw_ocv.py. Stage-3 OCV feature source.

Reuses the loaders' raw field-reading (LgHg2Loader._read_csv /
Panasonic18650PFLoader._read_mat) WITHOUT applying the drive-cycle skip filter,
so the (normally skipped) C/20 OCV files are read directly.

Per file: coulomb-count SOC from full charge (SOC=1) using the file's own
current + time and rated capacity; split the discharge and charge legs; resample
each onto a common SOC grid; average the two legs (cancels hysteresis); lightly
smooth; assert OCV is monotone-increasing in SOC. Writes data/ocv/ocv_*_25C.csv.

Usage: python scripts/build_ocv_tables.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from battery_bench.loaders.lg import LgHg2Loader
from battery_bench.loaders.panasonic import Panasonic18650PFLoader

OUT = Path("data/ocv")
SOC_GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.005), 5)


def _coulomb_soc(t, I, rated):
    """SOC from cumulative-trapezoid of current; anchored so full charge = 1."""
    q = np.concatenate(([0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(t)))) / 3600.0
    return 1.0 + (q - np.nanmax(q)) / rated


def _leg_to_grid(soc_leg, v_leg):
    order = np.argsort(soc_leg)
    s, v = soc_leg[order], v_leg[order]
    keep = np.concatenate(([True], np.diff(s) > 1e-9))   # strictly increasing for interp
    s, v = s[keep], v[keep]
    out = np.full(len(SOC_GRID), np.nan)
    inside = (SOC_GRID >= s[0]) & (SOC_GRID <= s[-1])
    out[inside] = np.interp(SOC_GRID[inside], s, v)
    return out


def build_table(t, V, I, rated, name):
    soc = _coulomb_soc(t, I, rated)
    turn = int(np.argmin(soc))                           # discharge -> charge turning point
    dis = _leg_to_grid(soc[:turn + 1], V[:turn + 1])
    cha = _leg_to_grid(soc[turn:], V[turn:])

    # True OCV ~ midpoint of the two legs. Each leg only spans part of SOC (the
    # cell may not start/return to full), so shift EACH leg to the true-OCV midline
    # using the per-SOC hysteresis gap (filled across SOC: linear inside, held at
    # edges) and average over their UNION. Where both legs exist this is their mean;
    # where one exists it is that leg corrected by half the (held) hysteresis, so
    # coverage gaps join smoothly instead of stepping by the hysteresis offset.
    both = ~np.isnan(dis) & ~np.isnan(cha)
    gap = cha - dis
    gap_filled = pd.Series(gap).interpolate(limit_direction="both").to_numpy()
    dis_adj = dis + gap_filled / 2.0
    cha_adj = cha - gap_filled / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN columns -> dropped below
        ocv_grid = np.nanmean(np.vstack([dis_adj, cha_adj]), axis=0)
    half = (float(np.nanmedian(gap[both])) / 2.0) if both.any() else 0.0
    valid = ~np.isnan(ocv_grid)
    soc_t, ocv_t = SOC_GRID[valid], ocv_grid[valid]

    win = min(11, len(ocv_t) - (1 - len(ocv_t) % 2))     # odd window <= len
    if win >= 5:
        ocv_s = savgol_filter(ocv_t, win, 2)
    else:
        ocv_s = ocv_t

    d = np.diff(ocv_s)
    if np.any(d < -1e-3):
        i = int(np.argmin(d))
        raise ValueError(f"[{name}] OCV not monotone-increasing near SOC={soc_t[i]:.3f} "
                         f"(dV={d[i]:.4f}); legs/smoothing wrong")

    print(f"[{name}] SOC {soc_t[0]:.3f}..{soc_t[-1]:.3f} | OCV {ocv_s[0]:.3f}..{ocv_s[-1]:.3f} V | "
          f"{len(soc_t)} pts | half-hyst {half*1000:.1f} mV | turn SOC={soc[turn]:.3f} (n={len(t)})")
    return pd.DataFrame({"soc": np.round(soc_t, 5), "ocv_V": np.round(ocv_s, 5)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    lg_file = next(Path("data/raw/LG18650HG2/25degC").glob("*C20DisCh*.csv"))
    d = LgHg2Loader("data/raw/LG18650HG2")._read_csv(lg_file)
    lg = build_table(d["time_s"], d["voltage_V"], d["current_A"], 3.0, "LG_HG2")
    lg.to_csv(OUT / "ocv_lg_25C.csv", index=False)

    pan_file = next(Path("data/raw/Panasonic-18650PF").rglob("*C20 OCV*25dC*.mat"))
    d = Panasonic18650PFLoader._read_mat(pan_file)
    pan = build_table(d["time_s"], d["voltage_V"], d["current_A"], 2.9, "PANASONIC")
    pan.to_csv(OUT / "ocv_pan_25C.csv", index=False)

    print(f"\nwrote {OUT/'ocv_lg_25C.csv'} and {OUT/'ocv_pan_25C.csv'}")
    print("source files:")
    print(f"  LG  : {lg_file}")
    print(f"  PAN : {pan_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
