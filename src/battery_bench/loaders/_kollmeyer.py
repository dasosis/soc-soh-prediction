"""Shared helpers for the two Kollmeyer drive-cycle SOC datasets
(LG 18650HG2 and Panasonic 18650PF). Both were logged on Digatron testers by
P. Kollmeyer with the SAME sign and label conventions, so the resampling, SOC
coulomb-counting, and drive-cycle vocabulary live here and are imported by both
``lg.py`` and ``panasonic.py``. This is NOT a harness-core module; it only
serves the loaders.

Conventions established once here (verified against raw files in Phase 1):
  * current is DISCHARGE-NEGATIVE in both datasets -> passed through unflipped.
  * each drive-cycle file resets its accumulated-Ah counter to ~0 at full
    charge, so every profile starts at SOC = 1.0.
  * SOC(t) = 1.0 + Ah_accum(t) / C_ref ,  C_ref default = rated capacity.
  * native logging is ~10 Hz (variable) -> resampled to a 1 Hz integer-second
    grid so a fixed window length means the same physical duration across
    datasets (the survey's default for raw V/I/T SOC models).
"""
from __future__ import annotations

import re

import numpy as np

# Canonical drive-cycle vocabulary used by the schema.
SCHEMA_DRIVE_CYCLES = {"UDDS", "US06", "LA92", "HWFET", "MIXED"}

# Minimum SOC swing for a profile to count as a real discharge. Some raw files
# are short rest/idle recordings (zero current, flat voltage at full charge);
# they carry no SOC dynamics and are skipped rather than emitted.
MIN_SOC_SWING = 0.02


def parse_nominal_temperature_c(folder_name: str) -> float:
    """Map a temperature folder name to its nominal setpoint in degC.

    Handles both the ``n10degC`` (LG) and ``-10degC`` (Panasonic) spellings, and
    folder suffixes like ``-20degC Trise`` / ``10degC Trise with pause``.
    """
    m = re.search(r"(n|-)?(\d+)\s*degc", folder_name.lower())
    if not m:
        raise ValueError(f"cannot parse temperature from folder '{folder_name}'")
    sign = -1.0 if m.group(1) in ("n", "-") else 1.0
    return sign * float(m.group(2))


def _token_to_vocab(token: str) -> str | None:
    """Map one filename token to a schema drive-cycle term, or None if the token
    is not a drive cycle (temperature tag, cell id, 'Pan18650PF', etc.)."""
    t = token.lower()
    if t == "us06":
        return "US06"
    if t == "udds":
        return "UDDS"
    if t == "la92":
        return "LA92"
    if t.startswith("hwf"):          # HWFET, HWFT, HWFTa, HWFTb
        return "HWFET"
    if t == "nn":                    # neural-network composite cycle
        return "MIXED"
    if t.startswith("cycle"):        # Cycle_1..4 = random mix
        return "MIXED"
    if t.startswith("mixed"):        # LG Mixed1..8 = random mix
        return "MIXED"
    return None


def classify_drive_cycle(descriptor: str) -> tuple[str | None, str]:
    """Classify a file descriptor into a schema drive-cycle term.

    Returns ``(drive_cycle, reason)``. ``drive_cycle`` is None when the file
    should be SKIPPED, and ``reason`` explains why (for logging):
      * a contiguous file containing charges/pauses ('pause', '1to4') is skipped
        because its Ah counter is not a clean single discharge ramp;
      * a contiguous multi-cycle file (more than one named drive cycle in the
        descriptor, e.g. ``US06_HWFET_UDDS_LA92``) is skipped to avoid double
        counting the split single-cycle files that duplicate it;
      * a descriptor with no drive-cycle token (Charge, HPPC, Cap_1C, ...) is
        skipped as 'not a drive cycle'.
    """
    d = descriptor.lower()
    if "pause" in d or "1to4" in d:
        return None, "contiguous file with charges/pauses"
    tokens = re.split(r"[_\s]+", descriptor)
    vocabs = [v for v in (_token_to_vocab(t) for t in tokens) if v]
    distinct = set(vocabs)
    if not distinct:
        return None, "not a drive cycle"
    if len(distinct) > 1:
        return None, "contiguous multi-cycle file (duplicate of split files)"
    return distinct.pop(), "ok"


def resample_to_1hz(time_s: np.ndarray, channels: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Linearly interpolate ``channels`` from their native (variable, ~10 Hz)
    timestamps onto a 1 Hz integer-second grid 0,1,..,floor(t_end).

    ``time_s`` is elapsed seconds from profile start. Non-increasing duplicate
    timestamps are dropped (np.interp requires a strictly increasing x).
    Returns ``(grid_seconds, {name: values_on_grid})``.
    """
    t = np.asarray(time_s, dtype=float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t = t[keep]
    if t.size < 2:
        raise ValueError("profile has fewer than 2 distinct timestamps")
    grid = np.arange(0.0, np.floor(t[-1]) + 1.0, 1.0)
    out: dict[str, np.ndarray] = {}
    for name, arr in channels.items():
        a = np.asarray(arr, dtype=float)[order][keep]
        out[name] = np.interp(grid, t, a)
    return grid, out


def coulomb_count_soc(ah_accum_grid: np.ndarray, reference_capacity_Ah: float) -> np.ndarray:
    """SOC referenced to the fully-charged state = 1.0.

    SOC(t) = 1.0 + (Ah_accum(t) - max Ah_accum) / C_ref.

    The cycler's Ah counter is reset to ~0 at the 4.2 V CV full charge that
    precedes every drive cycle, so the profile's MAXIMUM accumulated charge is
    the fully-charged reference. Subtracting that per-profile max anchors the
    peak to exactly SOC=1.0 and removes the sub-0.1% coulomb-counter reset /
    early-regen offsets that would otherwise push SOC a hair above 1.0. This is
    a constant offset, NOT a clip: all discharge dynamics are preserved, and a
    genuine sign/reference error still blows past [0,1] and is caught by the
    schema validator. No np.clip is ever applied.
    """
    return 1.0 + (ah_accum_grid - np.nanmax(ah_accum_grid)) / float(reference_capacity_Ah)
