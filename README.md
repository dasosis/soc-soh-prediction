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

## Stage 1 — SOC model comparison

Roster behind one `fit/predict` interface (`src/battery_bench/models/`,
registered by string): `linear` (ridge floor), `mlp`, `lstm`, `gru`, `bilstm`,
`cnn_bilstm_attn`, `tcn`, `transformer`, `patchtst`. Input window
`(N, window_len, 3)` for (V, I, T); target = SOC at the last timestep. Windowing
and normalization reuse the harness; the scaler is **fit on train (source) only**
and applied unchanged to test, preserving the measured covariate shift.

Two experiment families (`configs/soc_stage1.yaml`):
- **within-dataset** sanity (LG, Panasonic) — temperature-stratified profile holdout;
- **cross-dataset zero-shot** (headline) — leave-one-dataset-out via `runs/soc_cross_dataset/`.

```bash
# CPU smoke (1 seed, 2 epochs, tiny window subset) — verify the pipeline (~3 min):
# writes to its OWN dir (runs/soc_stage1_smoke/) so it never contaminates the full sweep.
python scripts/run_soc_stage1.py --config configs/soc_stage1.yaml --smoke
# full sweep (5 seeds): writes runs/soc_stage1/{results.csv,summary.csv,diagnostics/}
python scripts/run_soc_stage1.py --config configs/soc_stage1.yaml
```

Each run persists immediately under
`runs/soc_stage1/<experiment>/<model>/seed_<k>/` (config/scaler/metrics, plus
`preds.npz` for cross runs); `summary.csv` reports mean ± std over seeds (never
best-run). Cross-dataset diagnostics (error binned by true SOC, signed bias,
within-vs-cross gap, LG→Pan vs Pan→LG asymmetry) land in
`runs/soc_stage1/diagnostics/`.

### Running the full sweep on Colab / JarvisLabs (GPU)

```bash
git clone https://github.com/dasosis/soc-soh-prediction.git && cd soc-soh-prediction
pip install -e . && pip install -r requirements.txt        # torch picks up the GPU build
# bring processed tables + fold manifests (no raw data needed):
unzip artifacts/soc_crossdataset_bundle.zip -d _bundle
mkdir -p data/processed runs/soc_cross_dataset
cp _bundle/processed/*.parquet data/processed/
cp _bundle/runs/soc_cross_dataset/*.json runs/soc_cross_dataset/
python scripts/run_soc_stage1.py --smoke           # ~3 min sanity on the GPU box
python scripts/run_soc_stage1.py                   # full sweep
```

The full sweep is ~(#models × #seeds × #experiments) ≈ **180 runs**. Results
persist incrementally, so a disconnect loses nothing — re-running resumes
(completed `metrics.json` are skipped). Run the cheap models first, e.g.
`--models linear,mlp,lstm,gru,bilstm,cnn_bilstm_attn,tcn` then
`--models transformer,patchtst`, since the transformer family dominates runtime.

> **PINN is intentionally excluded** from Stage 1: for SOC it is a proposed-method
> / Stage-3 contribution, not a fair baseline for "compare existing methods."

## Stage 2 — fine-tuning data-efficiency ladder

For each cross-dataset direction, measures how target-TEST error falls as the
model is given a growing fraction of labeled **target** data, comparing network
**fine-tuning** (continue-train the source net, neural models only) vs a cheap
**affine recalibration** of the zero-shot predictions (`SOC' = a·ŷ + b`, plus a
bias-only `a=1` variant) — decomposing the Stage-1 transfer gap into removable
systematic bias vs genuine adaptation.

**Shared label budget.** The target is split once into disjoint **TEST + POOL**
(fixed seed, constant across all fractions/methods/models). At each fraction the
sampled `f·POOL` profiles are the *total* target-label budget at `f`, shared
identically by every method — recalibration fits on the full sample, fine-tuning
carves its early-stopping val from *inside* that same sample (fixed-epoch
fallback when the sample is too small). There is no off-budget external VAL, so
the x-axis is the honest budget (sampled profiles + windows are reported per
fraction). TEST is never used for fitting, recalibration, or early stopping. The
source scaler is fit on source-train only and held fixed across fractions, so the
curve isolates adaptation. Reuses the Stage-1 pipeline/models/metrics and
`resolve_out_dir` unchanged.

```bash
python scripts/run_soc_stage2.py --config configs/soc_stage2.yaml --smoke   # -> runs/soc_stage2_smoke/
python scripts/run_soc_stage2.py --config configs/soc_stage2.yaml           # full -> runs/soc_stage2/
```

Outputs (per direction): `summary.csv` (mean ± std over seeds per
model/method/fraction) and `diagnostics/` — data-efficiency curves (target-TEST
MAE vs fraction, fine-tune vs recalibration, within-target oracle as the dotted
reference, fraction 0 = zero-shot anchor), a headline 80%-gap-closure table, and
per-fraction gap-share. Same Colab recipe as Stage 1 (the bundle already carries
the processed tables + manifests; `git pull`, then run).
