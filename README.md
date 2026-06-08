# battery-bench

A leakage-safe, dataset-agnostic harness for **comparing SOC/SOH estimation
methods** across public Li-ion battery datasets. Built for an M.Tech thesis
whose contribution is *rigorous, reproducible, generalization-first comparison*
rather than a marginal accuracy tweak on one dataset.

Read **[`SCHEMA.md`](SCHEMA.md) first** — it is the canonical data contract that
every loader, splitter, normalizer, and model obeys.

## Status

| Area | State |
|---|---|
| Harness core (`schema`, `splits`, `preprocess`, `metrics`, `utils`) | done |
| **SOC loaders: LG 18650HG2 + Panasonic 18650PF** | **done** |
| SOH loaders (NASA / CALCE / Oxford) | not in this line of work |
| Cross-dataset SOC split manifests | done |
| Models (Stage 1) | not started |

The current focus is the **SOC cross-dataset generalization experiment**:
train on one chemistry/manufacturer, test on the other.

## Layout

```
src/battery_bench/
  schema.py            canonical column contracts + validators
  splits/splitter.py   leakage-safe group-holdout / forward-in-time
  preprocess/          fit-on-train-only standardizer + profile-safe windows
  metrics/regression.py RMSE/MAE/MAX/MAPE/R2 + SOC/SOH reports
  utils/io.py          parquet I/O + split-manifest persistence
  loaders/
    base.py            BaseLoader interface (emits validated canonical tables)
    lg.py              LG 18650HG2  (NMC, SOC drive cycles)   -> cells + soc_timeseries
    panasonic.py       Panasonic 18650PF (NCA, SOC drive cycles) -> cells + soc_timeseries
    _kollmeyer.py      shared helpers for the two Kollmeyer datasets
scripts/
  build_soc_processed.py        raw -> data/processed/*.parquet (+ summary)
  validate_processed.py         assert processed tables obey SCHEMA.md
  build_crossdataset_splits.py  leave-one-dataset-out manifests -> runs/
  make_bundle.py                portable zip for Colab / JarvisLabs
tests/                          pytest suite (harness + both SOC loaders)
configs/                        experiment configs
data/raw/                       untouched downloads (git-ignored)
data/processed/                 canonical parquet (git-ignored, regenerable)
runs/ , artifacts/              manifests + bundles (git-ignored)
```

## Quickstart

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows; use source on *nix
pip install -e .
pip install pytest
pytest -q                                        # 21 passing tests, no data needed
```

## SOC cross-dataset pipeline

With the raw datasets under `data/raw/LG18650HG2/` and
`data/raw/Panasonic-18650PF/`:

```bash
python scripts/build_soc_processed.py        # -> data/processed/{cells,soc_timeseries}.parquet
python scripts/validate_processed.py data/processed
python scripts/build_crossdataset_splits.py  # -> runs/soc_cross_dataset/fold_{A,B}_*.json
python scripts/make_bundle.py                # -> artifacts/soc_crossdataset_bundle.zip
```

Current output: **124 profiles / ~981k timesteps at 1 Hz** (LG 69, Panasonic 55).
Two leave-one-dataset-out folds:

- **fold A** — train = Panasonic, test = LG
- **fold B** — train = LG, test = Panasonic

## Conventions you must not break

- `current_A` is **discharge-negative**; `soc` is a **fraction in [0, 1]**;
  `soh = capacity / initial_capacity`.
- Splits are **cell-wise / dataset-wise or forward-in-time only** — never
  shuffle timesteps. Windows never cross a `profile_id`.
- Report **mean ± std over ≥5 seeds**, not the best run.

### SOC loader specifics (LG / Panasonic)

Both are McMaster/Wisconsin Digatron datasets (P. Kollmeyer), one physical cell
each, logged at ~10 Hz across several ambient temperatures.

- SOC is coulomb-counted from the cycler's accumulated-Ah column, anchored so
  each fully-charged profile peaks at exactly **SOC = 1.0**
  (`SOC = 1 + (Ah - max Ah)/C_ref`) — an offset, never a clip; `C_ref` defaults
  to rated capacity (3.0 Ah LG, 2.9 Ah Panasonic).
- Native ~10 Hz is resampled to a **1 Hz** integer-second grid so a fixed window
  length means the same physical duration across datasets.
- Drive cycles normalize to `{UDDS, US06, LA92, HWFET, MIXED}`
  (random / NN / Cycle_N -> MIXED, original name kept in `profile_id`).
- Cold cells (n10/n20/0 degC and Trise) are kept with the correct
  `nominal_temperature_C`; below 10 degC the protocol uses reduced/no regen.

## Next: Stage 1 (model wiring)

Wire models (`mlp` first, then `lstm`/`gru`/`bilstm`, `cnn_bilstm_attn`,
`transformer`, `patchtst`, a `pinn`) behind one `fit/predict` interface and run
the in-distribution comparison, then the headline cross-dataset generalization
runs against the manifests in `runs/`.
