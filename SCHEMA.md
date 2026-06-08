# Schema Specification — Battery SOC/SOH Benchmark Harness

**Status:** Stage 0 draft, v0.1
**Purpose:** Define one canonical, dataset-agnostic data format so that every
public dataset (NASA, CALCE, Oxford, Sandia, LG HG2, Panasonic) flows through
the *same* splitting, normalization, windowing, metrics, and models. This is
what makes a *fair* cross-method, cross-dataset comparison possible — the single
most important design decision in the thesis.

The contract below is enforced in code by `src/battery_bench/schema.py`. If a
loader emits a table that violates it, validation raises before any model runs.

---

## 1. Design principles

1. **One canonical form, many loaders.** Each raw dataset has its own messy
   format. A per-dataset loader is the *only* code that knows about that mess;
   it converts raw files into the canonical tables defined here. Everything
   downstream is dataset-agnostic.
2. **Units and conventions fixed once.** Ambiguous units are the field's #1
   source of non-comparable numbers. They are pinned in §3 and never negotiated
   per-dataset.
3. **Labels are data; thresholds are config.** SOC and SOH labels are stored.
   The EOL threshold used to derive RUL is *not* stored — it lives in the
   experiment config, so a single processed dataset supports EOL=70% and 80%
   without reprocessing.
4. **Columnar + metadata sidecar.** Processed data is stored as Parquet
   (language-agnostic, compresses repeated profile-level constants well). A
   small `cells` metadata table carries per-cell attributes used for splitting.
5. **Leakage-safety is structural.** The schema carries the identifiers
   (`cell_id`, `profile_id`, `cycle_index`) that the splitter and windower need
   to guarantee no future/test information leaks into training.

---

## 2. The three canonical tables

The pipeline reads at most three Parquet files from `data/processed/`:

| Table | Granularity | Task | Required for |
|---|---|---|---|
| `cells.parquet` | one row per cell | both | all splits (group metadata) |
| `soc_timeseries.parquet` | one row per timestep | SOC | SOC experiments |
| `soh_cycles.parquet` | one row per cycle | SOH | SOH/RUL experiments |

A SOC-only dataset (LG HG2, Panasonic) ships `cells` + `soc_timeseries`.
A SOH-only dataset (NASA, CALCE, Oxford, Sandia) ships `cells` + `soh_cycles`.

---

## 3. Units, signs, and label definitions (non-negotiable)

| Quantity | Unit | Convention |
|---|---|---|
| `voltage_V` | volts | — |
| `current_A` | amperes | **discharge negative** (current into the load); charge positive |
| `temperature_C` | °C | measured cell/surface temperature |
| `nominal_temperature_C` | °C | nominal ambient setpoint of the run |
| `capacity_Ah` | amp-hours | per-cycle discharge capacity |
| `soc` | fraction | in **[0, 1]** — never percent |
| `soh` | fraction | `capacity_Ah / initial_capacity_Ah` |

**SOC label.** Computed by the loader, typically coulomb counting from a known
initial SOC. The harness trusts the loader's `soc` column but validates the
range.

**SOH label.** `soh = capacity_Ah / initial_capacity_Ah`, where
`initial_capacity_Ah` is the **first valid measured capacity** of that cell
(stored in `cells`). Both `initial_capacity_Ah` and `nominal_capacity_Ah` are
stored so a config flag can switch the reference if desired.

**RUL.** Not stored. Derived in the pipeline as
`RUL(cycle) = (first cycle where soh <= eol_threshold) - cycle`, with
`eol_threshold` from config (default 0.80).

---

## 4. Column contracts

### 4.1 `cells` (one row per cell)

| Column | Type | Notes |
|---|---|---|
| `dataset` | string | e.g. `NASA_PCOE`, `LG_HG2`, `SANDIA_SNL` |
| `cell_id` | string | **globally unique**, e.g. `NASA_B0005`; prefix with dataset |
| `chemistry` | string | `NMC`, `NCA`, `LFP`, `LCO`, … (drives cross-chemistry splits) |
| `manufacturer` | string | |
| `form_factor` | string | `18650`, `pouch`, `prismatic` |
| `nominal_capacity_Ah` | float64 | rated capacity |
| `initial_capacity_Ah` | float64 | first valid measured capacity; **> 0** |

Validation: unique `cell_id`; `initial_capacity_Ah > 0`.

### 4.2 `soc_timeseries` (one row per timestep)

| Column | Type | Notes |
|---|---|---|
| `dataset` | string | |
| `cell_id` | string | FK to `cells` |
| `profile_id` | string | **one drive cycle / discharge run**, globally unique |
| `nominal_temperature_C` | float64 | constant within a profile |
| `drive_cycle` | string | `US06`, `UDDS`, `LA92`, `HWFET`, `MIXED`, … |
| `timestep` | int64 | `0..N-1` within profile, monotonic |
| `time_s` | float64 | seconds since profile start |
| `voltage_V` | float64 | |
| `current_A` | float64 | discharge negative |
| `temperature_C` | float64 | |
| `soc` | float64 | label, **[0, 1]** |

Profile-level constants (`nominal_temperature_C`, `drive_cycle`) are folded into
the timeseries to avoid join bugs; Parquet compresses the repetition. Validation:
`soc ∈ [0,1]`; `timestep` monotonic within each `profile_id`.

### 4.3 `soh_cycles` (one row per cycle)

| Column | Type | Notes |
|---|---|---|
| `dataset` | string | |
| `cell_id` | string | FK to `cells` |
| `cycle_index` | int64 | `1..M`, monotonic within cell |
| `capacity_Ah` | float64 | discharge capacity that cycle |
| `soh` | float64 | `capacity_Ah / initial_capacity_Ah` |
| `hi_*` (optional) | float64 | health-indicator features, e.g. `hi_cc_time_s`, `hi_dvdq_peak` |

Validation: `soh ∈ (0, 1.2]`; `cycle_index` monotonic within each `cell_id`.
`hi_*` feature columns are passed through untouched by validation.

---

## 5. How the splitter uses the schema

- **Cross-dataset (leave-one-dataset-out):** group `cells` by `dataset`.
- **Cross-chemistry (leave-one-chemistry-out):** group `cells` by `chemistry`
  — Sandia/SNL is ideal because LFP/NCA/NMC share one test protocol.
- **Cross-temperature (leave-one-temperature-out):** group by
  `nominal_temperature_C` (SOC) — pass a profile-level metadata frame to the
  splitter.
- **Forward-in-time (SOH within a cell):** sort by `cycle_index`, early→train,
  late→test; the splitter asserts `max(train cycle) < min(test cycle)` per cell.

No split ever shuffles timesteps or cycles. Windows never cross a `profile_id`.

---

## 6. On-disk layout

```
data/
  raw/        # untouched downloads, one subfolder per dataset (git-ignored)
    nasa_pcoe/ calce/ oxford/ sandia_snl/ lg_hg2/ panasonic_18650pf/
  processed/  # canonical parquet, written by loaders (git-ignored)
    cells.parquet
    soc_timeseries.parquet
    soh_cycles.parquet
runs/         # split manifests, scaler params, metrics, per experiment
```

After any loader writes `processed/`, run
`python scripts/validate_processed.py data/processed` to confirm the contract
holds before training anything.

---

## 7. Extending the schema

- New health indicators → add `hi_*` columns to `soh_cycles` (no validator
  change needed).
- A new dataset → subclass `BaseLoader`, emit the contracted columns, done.
- A new task (e.g. temperature estimation) → add a new canonical table + a
  validator rather than overloading an existing one.
