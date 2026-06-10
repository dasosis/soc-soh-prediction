#!/usr/bin/env python
"""READ-ONLY inventory of RAW LG + Panasonic files to locate low-rate
(C/20-style) OCV / characterization sweeps — the Stage-3 OCV source decision.

Standalone diagnostic: it copies the MINIMAL raw-field read logic from
loaders/_kollmeyer.py + lg.py + panasonic.py but deliberately does NOT import the
drive-cycle skip filter, so the files the loaders skip (C20DisCh, C20 OCV, Cap,
HPPC, ...) ARE included here. Touches nothing in core / loaders / data/processed;
the only output is a print table + the gitignored runs/inventory_raw_ocv.csv.

Usage: python scripts/inventory_raw_ocv.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

CHEMS = [
    {"name": "LG_HG2",    "raw": Path("data/raw/LG18650HG2"),       "rated": 3.0},
    {"name": "PANASONIC", "raw": Path("data/raw/Panasonic-18650PF"), "rated": 2.9},
]
KEYWORDS = ["OCV", "C20", "C/20", "Cap", "Char", "HPPC", "pulse", "dis5", "rest"]
_LG_COLS = ["Prog Time", "Voltage", "Current", "Temperature", "Capacity"]


# ---- minimal raw readers (copied from the loaders; no skip filter) ----
def _prog_time_to_seconds(values):
    out = np.empty(len(values), dtype=float)
    for i, v in enumerate(values):
        try:
            h, m, s = str(v).split(":")
            out[i] = int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            out[i] = np.nan
    return out


def read_lg_csv(path):
    df = pd.read_csv(path, skiprows=28, header=0, usecols=_LG_COLS, low_memory=False)
    df = df.iloc[1:].reset_index(drop=True)               # drop units row
    num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    V, I, T, Ah = num("Voltage"), num("Current"), num("Temperature"), num("Capacity")
    t = _prog_time_to_seconds(df["Prog Time"].to_numpy())
    ok = np.isfinite(t) & np.isfinite(V) & np.isfinite(I) & np.isfinite(Ah)
    t = t[ok]
    return t - t[0], V[ok], I[ok], Ah[ok], T[ok]


def read_mat(path):
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    meas = m[next(k for k in m if not k.startswith("__"))]
    t = np.asarray(meas.Time, dtype=float).ravel()
    V = np.asarray(meas.Voltage, dtype=float).ravel()
    I = np.asarray(meas.Current, dtype=float).ravel()
    Ah = np.asarray(meas.Ah, dtype=float).ravel()
    T = np.asarray(meas.Battery_Temp_degC, dtype=float).ravel() if hasattr(meas, "Battery_Temp_degC") else np.full_like(V, np.nan)
    return t - t[0], V, I, Ah, T


def read_file(path):
    if path.suffix.lower() == ".mat":
        return read_mat(path)
    return read_lg_csv(path)            # EIS/other CSVs lacking the cols will raise -> caught


# ---- temperature parse from path/filename ----
def parse_temp(path_str, T):
    m = re.search(r"(n|-)?(\d+)\s*d(?:eg)?c", path_str.lower())
    if m:
        return (-1 if m.group(1) in ("n", "-") else 1) * float(m.group(2))
    if np.isfinite(T).any():
        return round(float(np.nanmedian(T)), 1)
    return np.nan


def classify(cr, full_range, discharge, monotonic, span, rated, maxc, rest_frac, flips):
    """cr = mean|I|/rated (fraction). Thresholds per task spec; SLOW_OCV is the
    low-rate, full-range, substantially-discharging sweep (a C/20 OCV that also
    charges still qualifies — it is THE ocv source — but `monotonic`/sign are
    reported so a pure-discharge sweep is distinguishable)."""
    # OCV: very low MEAN and low PEAK current (excludes pulse tests), full-range,
    # large Ah span. NOT gated on net-discharge: a C/20 "discharge AND charge"
    # OCV sweep is balanced (net ~0), which is ideal (averages hysteresis). Peak
    # guard keeps HPPC/5pulse out; span guard keeps trickle/rest tails out.
    if cr < 1 / 15 and full_range and span > 0.6 * rated and maxc < 1.5:
        return "SLOW_OCV_CANDIDATE"
    if 0.35 < cr < 1.2 and full_range and discharge and rest_frac < 0.25 and monotonic:
        return "CAPACITY_TEST"
    if maxc > 1.5 and cr < 0.5 and (rest_frac > 0.3 or flips > 0.05):
        return "HPPC/PULSE"
    return "DRIVE_CYCLE"


def analyze(path, rated):
    t, V, I, Ah, T = read_file(path)
    if len(t) < 2 or not np.isfinite(t).any():
        raise ValueError("too few samples")
    absI = np.abs(I)
    dt = np.diff(t)
    dt = dt[dt > 0]
    rate = 1.0 / np.median(dt) if len(dt) else np.nan
    dur_h = (t[-1] - t[0]) / 3600.0
    mean_absI = float(np.nanmean(absI))
    cr = mean_absI / rated if mean_absI > 1e-9 else np.inf
    span = float(np.nanmax(Ah) - np.nanmin(Ah))
    vmin, vmax = float(np.nanmin(V)), float(np.nanmax(V))
    net = abs(float(Ah[-1] - Ah[0]))
    monotonic = span > 1e-6 and (net / span) > 0.7
    discharge = float(np.nanmean(I)) < 0
    rest_frac = float(np.nanmean(absI < rated / 10.0))
    sign = np.sign(I[absI > rated / 20.0]) if np.any(absI > rated / 20.0) else np.array([0])
    flips = float(np.mean(np.abs(np.diff(sign)) > 0)) if len(sign) > 1 else 0.0
    full_range = vmax >= 4.1 and vmin <= 2.85   # 2.85 covers the LG C/20 floor (~2.80 V)
    kws = [k for k in KEYWORDS if k.lower().replace("/", "") in path.name.lower().replace("/", "")]
    return {
        "filename": path.name,
        "class": classify(cr, full_range, discharge, monotonic, span, rated,
                          float(np.nanmax(absI)) / rated, rest_frac, flips),
        "c_rate": (f"C/{rated/mean_absI:.0f}" if mean_absI > 1e-9 else "C/inf"),
        "c_rate_frac": round(float(cr), 5),
        "duration_h": round(dur_h, 3),
        "sample_rate_hz": round(float(rate), 3),
        "mean_abs_I": round(mean_absI, 4),
        "median_abs_I": round(float(np.nanmedian(absI)), 4),
        "max_abs_I": round(float(np.nanmax(absI)), 4),
        "mean_I_sign": "discharge" if discharge else "charge",
        "monotonic": monotonic,
        "V_min": round(vmin, 3), "V_max": round(vmax, 3),
        "Ah_span": round(span, 4),
        "low_I_frac": round(rest_frac, 3),
        "temp": parse_temp(str(path), T),
        "keywords": ",".join(kws),
        "rel_path": str(path),
    }


def main():
    all_rows, could_not = [], []
    for chem in CHEMS:
        rows = []
        files = sorted(list(chem["raw"].rglob("*.mat")) + list(chem["raw"].rglob("*.csv")))
        for p in files:
            try:
                r = analyze(p, chem["rated"])
                r["chemistry"] = chem["name"]
                rows.append(r)
            except Exception as e:
                could_not.append((chem["name"], str(p), type(e).__name__))
        rows.sort(key=lambda r: r["c_rate_frac"])
        all_rows.extend(rows)

        print(f"\n{'='*120}\n{chem['name']}  (rated {chem['rated']} Ah, C/20 ~ {chem['rated']/20:.3f} A)"
              f"  — {len(rows)} files parsed, sorted slowest-first\n{'='*120}")
        hdr = f"{'filename':42} {'class':20} {'c_rate':7} {'dur_h':>7} {'mAbsI':>7} {'Vmin':>6} {'Vmax':>6} {'Ahspan':>7} {'temp':>6} {'loIfr':>6}"
        print(hdr)
        for r in rows:
            print(f"{r['filename'][:42]:42} {r['class']:20} {r['c_rate']:7} {r['duration_h']:>7.2f} "
                  f"{r['mean_abs_I']:>7.3f} {r['V_min']:>6.2f} {r['V_max']:>6.2f} {r['Ah_span']:>7.3f} "
                  f"{str(r['temp']):>6} {r['low_I_frac']:>6.2f}")

    out = Path("runs/inventory_raw_ocv.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    print(f"\nwrote {out} ({len(all_rows)} rows)")

    if could_not:
        print(f"\nCOULD NOT READ ({len(could_not)}):")
        for chem, p, err in could_not[:40]:
            print(f"  [{chem}] {err}: {p}")
        if len(could_not) > 40:
            print(f"  ... and {len(could_not)-40} more")

    # ---- VERDICT ----
    print(f"\n{'='*120}\nVERDICT\n{'='*120}")
    for chem in CHEMS:
        rows = [r for r in all_rows if r["chemistry"] == chem["name"]]
        cands = [r for r in rows if r["class"] == "SLOW_OCV_CANDIDATE"]
        # fallback only over real sweeps (exclude trickle/rest tails with no Ah span)
        below10 = [r for r in rows if r["c_rate_frac"] < 0.1 and r["Ah_span"] > 0.5 * chem["rated"]]
        if not cands:
            slow = min(below10, key=lambda r: r["c_rate_frac"]) if below10 else None
            if slow:
                print(f"VERDICT {chem['name']}: no strict SLOW_OCV_CANDIDATE; slowest <C/10 = "
                      f"{slow['filename']} ({slow['c_rate']}, {slow['duration_h']:.1f} h, "
                      f"V {slow['V_max']:.2f}->{slow['V_min']:.2f}, {slow['temp']}C, "
                      f"Ah_span {slow['Ah_span']:.2f}, {slow['mean_I_sign']}) — inspect.")
            else:
                print(f"VERDICT {chem['name']}: no file below C/10 found.")
            continue
        at25 = [r for r in cands if r["temp"] == 25.0]
        best = min(at25 or cands, key=lambda r: r["c_rate_frac"])
        full = best["V_max"] >= 4.1 and best["V_min"] <= 2.7
        if best["temp"] != 25.0:
            note = f"NO 25C candidate; nearest is {best['temp']}C — flag for Stage 3"
        elif full:
            note = "full-range 25C OCV available"
        else:
            note = f"near-full-range 25C OCV (V floor {best['V_min']:.2f} V, does not reach ~2.5 V)"
        print(f"VERDICT {chem['name']}: best OCV source = {best['filename']} "
              f"({best['c_rate']}, {best['duration_h']:.1f} h, V {best['V_max']:.2f}->{best['V_min']:.2f}, "
              f"{best['temp']}C, Ah_span {best['Ah_span']:.2f}, {best['mean_I_sign']}, "
              f"monotonic={best['monotonic']}) — {note}.")
        if at25 and len(cands) > 1:
            others = [f"{r['temp']}C" for r in cands if r is not best]
            print(f"                 ({len(cands)} candidates total; other temps: {sorted(set(others))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
