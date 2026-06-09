#!/usr/bin/env python
"""Stage 1 SOC sweep runner — config-driven, one command = full sweep.

Experiments:
  * WITHIN-DATASET sanity (LG, Panasonic): temperature-stratified profile holdout.
  * CROSS-DATASET zero-shot (headline): leave-one-dataset-out manifests, scaler
    and val carved from the SOURCE only.

Every (experiment x model x seed) run trains, predicts on test, computes metrics,
and persists immediately under out_dir/<experiment>/<model>/seed_<k>/
(config.json, scaler.json, metrics.json, and preds.npz for cross runs). Each run
is also appended to out_dir/results.csv as it finishes, so a Colab disconnect
never loses completed work; summary.csv + diagnostics are recomputed from the
per-run files at the end.

--smoke writes to a SEPARATE output dir ('<out_dir>_smoke', i.e.
runs/soc_stage1_smoke/) so undertrained smoke runs never collide with — or get
resumed into — the full sweep under cfg['out_dir'] (runs/soc_stage1/).

Usage:
  python scripts/run_soc_stage1.py --config configs/soc_stage1.yaml --smoke   # -> runs/soc_stage1_smoke/
  python scripts/run_soc_stage1.py --config configs/soc_stage1.yaml           # full -> runs/soc_stage1/
  python scripts/run_soc_stage1.py --config ... --models linear,lstm --experiments within_LG
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from battery_bench import soc_pipeline as P
from battery_bench.metrics.regression import soc_report, r2
from battery_bench.models import build_model
from battery_bench.utils import read_table

RESULTS_COLUMNS = ["experiment", "model", "seed", "rmse", "mae", "max_error",
                   "mape_pct", "r2", "n_params", "train_time_s", "n_train",
                   "n_test", "device", "timestamp"]


# --------------------------------------------------------------------------
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_out_dir(cfg, smoke: bool) -> Path:
    """Output directory for this run. The full sweep writes to cfg['out_dir'];
    smoke runs write to a SEPARATE sibling dir so undertrained smoke runs can
    never be picked up by the full sweep's resume logic. Honors an optional
    cfg['smoke']['out_dir'] override, else uses '<out_dir>_smoke'."""
    base = Path(cfg["out_dir"])
    if not smoke:
        return base
    override = (cfg.get("smoke") or {}).get("out_dir")
    return Path(override) if override else base.with_name(base.name + "_smoke")


def build_run_cfg(cfg, model_name, seed, smoke):
    run = dict(cfg.get("deep_defaults", {}))
    run.update(cfg.get("model_overrides", {}).get(model_name, {}))
    run["seed"] = int(seed)
    if smoke:
        run["max_epochs"] = int(cfg["smoke"].get("max_epochs", 2))
    return run


def prepare_experiment(exp, cfg, soc, meta, smoke):
    """Return (Xtr, ytr, Xval, yval, Xte, yte, scaler, info) for one experiment."""
    feats, target = cfg["features"], cfg["target"]
    length, stride = cfg["window"]["length"], cfg["window"]["stride"]

    if exp["type"] == "within":
        wd = cfg["within_dataset"]
        train, val, test = P.split_within(meta, exp["dataset"], wd["test_frac"],
                                          wd["val_frac"], wd["split_seed"])
    elif exp["type"] == "cross":
        cd = cfg["cross_dataset"]
        manifest = json.loads(Path(exp["manifest"]).read_text())
        train, val, test = P.split_cross(manifest, meta, cd["val_frac"], cd["split_seed"])
    else:
        raise ValueError(f"unknown experiment type {exp['type']}")

    scaler = P.fit_source_scaler(soc, train, feats)
    Xtr, ytr, _ = P.make_split_windows(soc, train, scaler, feats, target, length, stride)
    Xval, yval, _ = P.make_split_windows(soc, val, scaler, feats, target, length, stride)
    Xte, yte, _ = P.make_split_windows(soc, test, scaler, feats, target, length, stride)

    if smoke:
        Xtr, ytr = P.subsample_windows(Xtr, ytr, cfg["smoke"]["max_windows_train"], seed=0)
        Xval, yval = P.subsample_windows(Xval, yval, cfg["smoke"]["max_windows_eval"], seed=1)
        Xte, yte = P.subsample_windows(Xte, yte, cfg["smoke"]["max_windows_eval"], seed=2)

    info = {"train_profiles": train, "val_profiles": val, "test_profiles": test,
            "n_train_windows": int(len(Xtr)), "n_val_windows": int(len(Xval)),
            "n_test_windows": int(len(Xte))}
    return Xtr, ytr, Xval, yval, Xte, yte, scaler, info


def append_result(results_csv: Path, row: dict):
    new = not results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------
def run_sweep(cfg, smoke, model_filter, exp_filter, force):
    out_dir = resolve_out_dir(cfg, smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"

    soc = read_table(Path(cfg["processed_dir"]) / "soc_timeseries.parquet")
    meta = P.profile_meta(soc)

    experiments = [e for e in cfg["experiments"] if not exp_filter or e["name"] in exp_filter]
    models = [m for m in cfg["models"] if not model_filter or m in model_filter]
    seeds = cfg["smoke"]["seeds"] if smoke else cfg["seeds"]

    print(f"{'SMOKE' if smoke else 'FULL'} sweep | experiments={[e['name'] for e in experiments]} "
          f"| models={models} | seeds={seeds}")

    for exp in experiments:
        t0 = time.time()
        Xtr, ytr, Xval, yval, Xte, yte, scaler, info = prepare_experiment(exp, cfg, soc, meta, smoke)
        print(f"\n=== {exp['name']} ({exp['type']}) | windows train/val/test = "
              f"{len(Xtr)}/{len(Xval)}/{len(Xte)} | prep {time.time()-t0:.1f}s ===")

        for model_name in models:
            for seed in seeds:
                seed_dir = out_dir / exp["name"] / model_name / f"seed_{seed}"
                metrics_path = seed_dir / "metrics.json"
                if metrics_path.exists() and not force:
                    print(f"  skip (done) {exp['name']}/{model_name}/seed_{seed}")
                    continue
                seed_dir.mkdir(parents=True, exist_ok=True)

                run_cfg = build_run_cfg(cfg, model_name, seed, smoke)
                model = build_model(model_name)
                t1 = time.time()
                model.fit(Xtr, ytr, Xval, yval, run_cfg)
                train_time = time.time() - t1
                y_hat = model.predict(Xte)

                rep = soc_report(yte, y_hat)
                rep["r2"] = r2(yte, y_hat)
                device = getattr(getattr(model, "device", None), "type", "cpu")

                # persist
                (seed_dir / "config.json").write_text(json.dumps(
                    {"experiment": exp, "model": model_name, "run_cfg": run_cfg,
                     "window": cfg["window"], "features": cfg["features"],
                     "n_windows": {k: info[k] for k in info if k.startswith("n_")}},
                    indent=2))
                scaler.save(seed_dir / "scaler.json")
                (seed_dir / "metrics.json").write_text(json.dumps(
                    {"experiment": exp["name"], "type": exp["type"], "model": model_name,
                     "seed": int(seed), "n_params": model.n_params,
                     "train_time_s": train_time, "device": device,
                     "smoke": bool(smoke), "max_epochs": int(run_cfg.get("max_epochs", 0)),
                     "n_train": int(len(Xtr)), "n_test": int(len(Xte)), **rep}, indent=2))
                if exp["type"] == "cross":
                    np.savez_compressed(seed_dir / "preds.npz",
                                        y_true=np.asarray(yte), y_pred=np.asarray(y_hat))

                append_result(results_csv, {
                    "experiment": exp["name"], "model": model_name, "seed": int(seed),
                    "rmse": rep["rmse"], "mae": rep["mae"], "max_error": rep["max_error"],
                    "mape_pct": rep["mape_pct"], "r2": rep["r2"], "n_params": model.n_params,
                    "train_time_s": round(train_time, 3), "n_train": int(len(Xtr)),
                    "n_test": int(len(Xte)), "device": device,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                print(f"  {model_name:16} seed {seed} | MAE {rep['mae']:.4f} RMSE {rep['rmse']:.4f} "
                      f"R2 {rep['r2']:.3f} | {train_time:.1f}s | {model.n_params:,} params")

    return out_dir


# --------------------------------------------------------------------------
def aggregate_summary(out_dir: Path) -> pd.DataFrame:
    """Recompute summary.csv from authoritative per-run metrics.json files
    (robust to appended duplicates in results.csv)."""
    rows = []
    for mp in out_dir.glob("*/*/seed_*/metrics.json"):
        rows.append(json.loads(mp.read_text()))
    if not rows:
        print("no completed runs to summarize")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["experiment", "model"], as_index=False)
    summary = g.agg(
        n_seeds=("seed", "nunique"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        max_mean=("max_error", "mean"),
        r2_mean=("r2", "mean"),
        n_params=("n_params", "first"),
        train_time_mean=("train_time_s", "mean"),
    ).round(5)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nwrote {out_dir/'summary.csv'} ({len(summary)} experiment×model rows)")
    return summary


def cross_diagnostics(cfg, out_dir: Path):
    """Cross-dataset diagnostics: error-by-SOC, signed bias, within-vs-cross gap,
    and LG->Pan vs Pan->LG asymmetry. (Does not read THESIS_LOG.)"""
    diag = out_dir / "diagnostics"
    diag.mkdir(exist_ok=True)
    cross = [e for e in cfg["experiments"] if e["type"] == "cross"]

    # 1) error binned by true SOC + 2) signed bias per fold
    bins = np.linspace(0, 1, 21)
    mid = 0.5 * (bins[1:] + bins[:-1])
    by_soc_rows, bias_rows = [], []
    for exp in cross:
        exp_dir = out_dir / exp["name"]
        if not exp_dir.exists():
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for model_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            preds = list(model_dir.glob("seed_*/preds.npz"))
            if not preds:
                continue
            yt = np.concatenate([np.load(p)["y_true"] for p in preds])
            yp = np.concatenate([np.load(p)["y_pred"] for p in preds])
            signed, absol = yp - yt, np.abs(yp - yt)
            idx = np.digitize(yt, bins) - 1
            idx = np.clip(idx, 0, len(mid) - 1)
            ms = np.array([signed[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(mid))])
            ma = np.array([absol[idx == b].mean() if np.any(idx == b) else np.nan for b in range(len(mid))])
            for b in range(len(mid)):
                by_soc_rows.append({"experiment": exp["name"], "model": model_dir.name,
                                    "soc_bin": round(float(mid[b]), 3),
                                    "mean_signed_error": ms[b], "mean_abs_error": ma[b]})
            axes[0].plot(mid, ms, marker=".", label=model_dir.name)
            axes[1].plot(mid, ma, marker=".", label=model_dir.name)
            # per-seed signed bias
            for p in preds:
                d = np.load(p)
                bias_rows.append({"experiment": exp["name"], "model": model_dir.name,
                                  "mean_signed_error": float((d["y_pred"] - d["y_true"]).mean())})
        axes[0].axhline(0, color="k", lw=0.8, ls=":")
        axes[0].set(title=f"{exp['name']}\nsigned error vs true SOC", xlabel="true SOC",
                    ylabel="mean signed error (pred - true)")
        axes[1].set(title="absolute error vs true SOC", xlabel="true SOC", ylabel="mean |error|")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(diag / f"error_by_soc_{exp['name']}.png", dpi=120)
        plt.close(fig)

    if by_soc_rows:
        pd.DataFrame(by_soc_rows).to_csv(diag / "error_by_soc.csv", index=False)
    if bias_rows:
        bdf = pd.DataFrame(bias_rows)
        (bdf.groupby(["experiment", "model"], as_index=False)["mean_signed_error"]
            .agg(bias_mean="mean", bias_std="std").round(5)
            .to_csv(diag / "bias_summary.csv", index=False))

    # 3) within-vs-cross gap + asymmetry, from summary.csv
    spath = out_dir / "summary.csv"
    if spath.exists():
        s = pd.read_csv(spath).set_index(["experiment", "model"])["rmse_mean"]
        names = {e["type"] + "_" + e.get("target", ""): e["name"] for e in cfg["experiments"]}
        # map specific experiment names
        emap = {e["name"]: e for e in cfg["experiments"]}
        cross_by_target = {e["target"]: e["name"] for e in cross}
        rows = []
        models = sorted({m for (_, m) in s.index})
        for m in models:
            def rmse(exp_name):
                return float(s.get((exp_name, m), np.nan))
            within_lg = rmse("within_LG")
            within_pan = rmse("within_PAN")
            cross_lg = rmse(cross_by_target.get("LG_HG2", ""))      # train Pan -> test LG
            cross_pan = rmse(cross_by_target.get("PANASONIC_18650PF", ""))  # train LG -> test Pan
            rows.append({
                "model": m,
                "within_LG_rmse": within_lg, "cross_test_LG_rmse": cross_lg,
                "gap_LG": cross_lg - within_lg,
                "within_PAN_rmse": within_pan, "cross_test_PAN_rmse": cross_pan,
                "gap_PAN": cross_pan - within_pan,
                "LGtoPAN_rmse": cross_pan, "PANtoLG_rmse": cross_lg,
                "asymmetry_LGtoPAN_over_PANtoLG": (cross_pan / cross_lg) if cross_lg else np.nan,
            })
        pd.DataFrame(rows).round(5).to_csv(diag / "within_vs_cross_gap.csv", index=False)
    print(f"wrote diagnostics to {diag}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/soc_stage1.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--models", default="", help="comma-separated subset")
    ap.add_argument("--experiments", default="", help="comma-separated subset")
    ap.add_argument("--force", action="store_true", help="recompute even if metrics.json exists")
    ap.add_argument("--diagnostics-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_filter = [s for s in args.models.split(",") if s]
    exp_filter = [s for s in args.experiments.split(",") if s]
    out_dir = resolve_out_dir(cfg, args.smoke)

    if not args.diagnostics_only:
        out_dir = run_sweep(cfg, args.smoke, model_filter, exp_filter, args.force)
    aggregate_summary(out_dir)
    cross_diagnostics(cfg, out_dir)

    spath = out_dir / "summary.csv"
    if spath.exists():
        print("\n================ SUMMARY ================")
        print(pd.read_csv(spath).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
