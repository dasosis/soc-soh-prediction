#!/usr/bin/env python
"""Stage 3 — OCV-informed feature full sweep (label-free domain adaptation).

Reuses the S1/S2 pipeline/models/metrics + resolve_out_dir. Test split = the WHOLE
target (same as Stage 1) so the C1 baseline reproduces S1 exactly. Scaler is fit on
SOURCE train only; the OCV channel is appended UNSCALED (it is the domain-shifting
signal we want preserved).

Conditions (all label-free — no labeled target drive cycles):
  C1_baseline  : (V, I, T)                                  [reproduce S1; asserted]
  C2_ocv_mean  : (V, I, T, SOC_ocv), source OCV curve@train, TARGET OCV curve@test  [HEADLINE]
  C3_ocv_dischg: C2 with discharge-leg-only OCV tables      [NCA-hysteresis ablation]
  C4_coral     : (V, I, T) + CORAL aligning source-train features to UNLABELED
                 target test inputs (mean + covariance)     [marginal-alignment FOIL]

Usage:
  python scripts/run_soc_stage3.py --config configs/soc_stage3.yaml --smoke   # -> runs/soc_stage3_smoke/
  python scripts/run_soc_stage3.py --config configs/soc_stage3.yaml           # full -> runs/soc_stage3/
  python scripts/run_soc_stage3.py --config ... --verify-c1                    # C1==S1 (linear, full)
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_soc_stage1 import resolve_out_dir  # noqa: E402

from battery_bench import soc_pipeline as P  # noqa: E402
from battery_bench.metrics.regression import soc_report, r2  # noqa: E402
from battery_bench.models import build_model  # noqa: E402
from battery_bench.preprocess.ocv_feature import soc_from_voltage  # noqa: E402
from battery_bench.utils import read_table  # noqa: E402

RESULTS_COLUMNS = ["direction", "condition", "model", "seed", "mae", "rmse",
                   "max_error", "mape_pct", "r2", "signed_bias", "n_train",
                   "n_test", "device", "smoke", "timestamp"]
SOC_BINS = np.round(np.arange(0.0, 1.0001, 0.1), 3)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_cfg_for(cfg, model, seed, smoke):
    rc = dict(cfg.get("deep_defaults", {}))
    rc.update(cfg.get("model_overrides", {}).get(model, {}))
    rc["seed"] = int(seed)
    if smoke:
        rc["max_epochs"] = int(cfg["smoke"].get("max_epochs", 2))
    return rc


# ---- CORAL (mean + covariance alignment of source-train -> target inputs) ----
def _sym_sqrt(M, inv=False):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    s = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return (V * s) @ V.T


def coral_align(Xtr, Xva, Xte, eps):
    C = Xtr.shape[-1]
    s = Xtr.reshape(-1, C).astype(np.float64)
    t = Xte.reshape(-1, C).astype(np.float64)
    ms, mt = s.mean(0), t.mean(0)
    A = _sym_sqrt(np.cov(s, rowvar=False) + eps * np.eye(C), inv=True) @ \
        _sym_sqrt(np.cov(t, rowvar=False) + eps * np.eye(C))

    def tf(X):
        flat = (X.reshape(-1, C).astype(np.float64) - ms) @ A + mt
        return flat.reshape(X.shape).astype(np.float32)
    return tf(Xtr), tf(Xva)


# ---- per-direction base windows -------------------------------------------
def prepare_base(direction, cfg, soc, meta):
    feats, target = cfg["features"], cfg["target"]
    L, S = cfg["window"]["length"], cfg["window"]["stride"]
    manifest = json.loads(Path(direction["manifest"]).read_text())
    st = cfg["source_train"]
    tr, va, te = P.split_cross(manifest, meta, st["val_frac"], st["split_seed"])  # te = whole target
    scaler = P.fit_source_scaler(soc, tr, feats)
    Xtr, ytr, _ = P.make_split_windows(soc, tr, scaler, feats, target, L, S)
    Xva, yva, _ = P.make_split_windows(soc, va, scaler, feats, target, L, S)
    Xte, yte, _ = P.make_split_windows(soc, te, scaler, feats, target, L, S)
    return dict(scaler=scaler, tr=tr, va=va, te=te, L=L, S=S, feats=feats, target=target,
                Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)


def condition_windows(cond, direction, cfg, soc, base):
    src, tgt = direction["source_chem"], direction["target_chem"]
    f, t, L, S, sc = base["feats"], base["target"], base["L"], base["S"], base["scaler"]
    if cond == "C1_baseline":
        return base["Xtr"], base["Xva"], base["Xte"]
    if cond == "C4_coral":
        Xtr, Xva = coral_align(base["Xtr"], base["Xva"], base["Xte"], cfg["coral"]["eps"])
        return Xtr, Xva, base["Xte"]
    suffix = ":dischg" if cond == "C3_ocv_dischg" else ""        # C2_ocv_mean -> ""
    Xtr, _, _ = P.make_split_windows(soc, base["tr"], sc, f, t, L, S, add_ocv=True, ocv_chem=src + suffix)
    Xva, _, _ = P.make_split_windows(soc, base["va"], sc, f, t, L, S, add_ocv=True, ocv_chem=src + suffix)
    Xte, _, _ = P.make_split_windows(soc, base["te"], sc, f, t, L, S, add_ocv=True, ocv_chem=tgt + suffix)
    return Xtr, Xva, Xte


def append_result(results_csv, row):
    new = not results_csv.exists()
    with open(results_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


# ---- sweep ----------------------------------------------------------------
def run_sweep(cfg, smoke, dir_filter, model_filter, cond_filter, force):
    out_dir = resolve_out_dir(cfg, smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    soc = read_table(Path(cfg["processed_dir"]) / "soc_timeseries.parquet")
    meta = P.profile_meta(soc)

    directions = [d for d in cfg["directions"] if not dir_filter or d["name"] in dir_filter]
    conditions = [c for c in cfg["conditions"] if not cond_filter or c in cond_filter]
    models = [m for m in cfg["models"] if not model_filter or m in model_filter]
    seeds = cfg["smoke"]["seeds"] if smoke else cfg["seeds"]
    mw_tr = cfg["smoke"]["max_windows_train"] if smoke else None
    mw_ev = cfg["smoke"]["max_windows_eval"] if smoke else None
    print(f"{'SMOKE' if smoke else 'FULL'} stage-3 | dirs={[d['name'] for d in directions]} "
          f"| conditions={conditions} | models={models} | seeds={seeds}")

    for direction in directions:
        dn = direction["name"]
        t0 = time.time()
        base = prepare_base(direction, cfg, soc, meta)
        print(f"\n=== {dn} | src tr/val={len(base['Xtr'])}/{len(base['Xva'])} "
              f"target test={len(base['Xte'])} windows | prep {time.time()-t0:.1f}s ===")

        for cond in conditions:
            Xtr, Xva, Xte = condition_windows(cond, direction, cfg, soc, base)
            ytr, yva, yte = base["ytr"], base["yva"], base["yte"]
            if smoke:
                Xtr, ytr = P.subsample_windows(Xtr, ytr, mw_tr, seed=0)
                Xva, yva = P.subsample_windows(Xva, yva, mw_ev, seed=1)
                Xte, yte = P.subsample_windows(Xte, yte, mw_ev, seed=2)

            for model_name in models:
                for seed in seeds:
                    sd = out_dir / dn / cond / model_name / f"seed_{seed}"
                    if (sd / "metrics.json").exists() and not force:
                        continue
                    sd.mkdir(parents=True, exist_ok=True)
                    rc = run_cfg_for(cfg, model_name, seed, smoke)
                    t1 = time.time()
                    m = build_model(model_name).fit(Xtr, ytr, Xva, yva, rc)
                    yhat = m.predict(Xte)
                    rep = soc_report(yte, yhat)
                    rep["r2"] = r2(yte, yhat)
                    bias = float(np.mean(np.asarray(yhat) - np.asarray(yte)))
                    device = getattr(getattr(m, "device", None), "type", "cpu")
                    rec = {"direction": dn, "condition": cond, "model": model_name,
                           "seed": int(seed), "signed_bias": bias, "n_train": int(len(Xtr)),
                           "n_test": int(len(Xte)), "device": device, "smoke": bool(smoke), **rep}
                    (sd / "metrics.json").write_text(json.dumps(rec, indent=2))
                    np.savez_compressed(sd / "preds.npz", y_true=np.asarray(yte), y_pred=np.asarray(yhat))
                    append_result(results_csv, {**{k: rec.get(k) for k in RESULTS_COLUMNS if k != "timestamp"},
                                                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                    print(f"  {cond:13} {model_name:9} seed {seed} | MAE {rep['mae']:.4f} "
                          f"bias {bias:+.4f} | {time.time()-t1:.1f}s")
            del Xtr, Xva, Xte
            gc.collect()
    return out_dir


# ---- diagnostics ----------------------------------------------------------
def diagnostics(cfg, out_dir):
    diag = out_dir / "diagnostics"
    diag.mkdir(exist_ok=True)
    rows = [json.loads(p.read_text()) for p in out_dir.glob("*/*/*/seed_*/metrics.json")]
    if not rows:
        print("no runs to diagnose")
        return
    df = pd.DataFrame(rows)

    # summary.csv (per direction/model/condition: mean +/- std over seeds)
    summ = (df.groupby(["direction", "model", "condition"], as_index=False)
              .agg(n_seeds=("seed", "nunique"), mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                   rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                   bias_mean=("signed_bias", "mean"), bias_std=("signed_bias", "std")).round(5))
    summ.to_csv(diag / "summary.csv", index=False)

    # bias_by_condition.csv (S3-2: C2 bias -> ~0)
    (df.groupby(["direction", "condition"], as_index=False)
       .agg(bias_mean=("signed_bias", "mean"), bias_std=("signed_bias", "std"),
            mae_mean=("mae", "mean")).round(5)).to_csv(diag / "bias_by_condition.csv", index=False)

    # error_by_soc.csv (pool seeds per direction/condition/model)
    eb = []
    for p in out_dir.glob("*/*/*/seed_*/preds.npz"):
        dn, cond, model = p.parts[-5], p.parts[-4], p.parts[-3]
        d = np.load(p)
        yt, yp = d["y_true"], d["y_pred"]
        idx = np.clip(np.digitize(yt, SOC_BINS) - 1, 0, len(SOC_BINS) - 2)
        for b in range(len(SOC_BINS) - 1):
            m = idx == b
            if m.any():
                eb.append({"direction": dn, "condition": cond, "model": model,
                           "soc_bin": f"{SOC_BINS[b]:.1f}-{SOC_BINS[b+1]:.1f}",
                           "mae": float(np.mean(np.abs(yp[m] - yt[m]))), "count": int(m.sum())})
    if eb:
        (pd.DataFrame(eb).groupby(["direction", "condition", "model", "soc_bin"], as_index=False)
           .agg(mae=("mae", "mean"), count=("count", "sum")).round(5)).to_csv(diag / "error_by_soc.csv", index=False)

    # clamp_rate.csv (per direction, C2 mean table + C3 dischg table on target test voltage)
    soc = read_table(Path(cfg["processed_dir"]) / "soc_timeseries.parquet")
    cr = []
    for d in cfg["directions"]:
        man = json.loads(Path(d["manifest"]).read_text())
        v = soc[soc["profile_id"].isin(set(man["test_profiles"]))]["voltage_V"].to_numpy()
        for cond, suf in [("C2_ocv_mean", ""), ("C3_ocv_dischg", ":dischg")]:
            _, clamped = soc_from_voltage(v, d["target_chem"] + suf)
            cr.append({"direction": d["name"], "condition": cond, "target_chem": d["target_chem"],
                       "clamp_rate": round(float(clamped.mean()), 4)})
    pd.DataFrame(cr).to_csv(diag / "clamp_rate.csv", index=False)

    # gapclose.csv (the money table; pulls S2 recal plateau + finetune oracle)
    s2 = pd.read_csv(cfg["stage2_summary"]) if Path(cfg["stage2_summary"]).exists() else None
    gc_rows = []
    cond_col = {"C1_baseline": "zero_shot", "C2_ocv_mean": "ocv_mean",
                "C3_ocv_dischg": "ocv_dischg", "C4_coral": "coral"}
    for (dn, model), g in summ.groupby(["direction", "model"]):
        vals = {cond_col[c]: float(g[g.condition == c]["mae_mean"].iloc[0])
                for c in cond_col if (g.condition == c).any()}
        recal_plateau = oracle = np.nan
        if s2 is not None:
            sub = s2[s2.direction == dn]
            if len(sub):
                fmax = sub.fraction.max()
                ra = sub[(sub.model == model) & (sub.method == "recal_affine") & (sub.fraction == fmax)]
                ft = sub[(sub.model == model) & (sub.method == "finetune") & (sub.fraction == fmax)]
                recal_plateau = float(ra["mae_mean"].iloc[0]) if len(ra) else np.nan
                oracle = float(ft["mae_mean"].iloc[0]) if len(ft) else np.nan
        c1, c2 = vals.get("zero_shot", np.nan), vals.get("ocv_mean", np.nan)
        pct = ((c1 - c2) / (c1 - oracle)) if (np.isfinite(oracle) and (c1 - oracle) != 0) else np.nan
        gc_rows.append({"direction": dn, "model": model, **vals,
                        "recal_affine_plateau": round(recal_plateau, 5) if np.isfinite(recal_plateau) else np.nan,
                        "finetune_oracle": round(oracle, 5) if np.isfinite(oracle) else np.nan,
                        "pct_gap_closed": round(pct, 4) if np.isfinite(pct) else np.nan})
    pd.DataFrame(gc_rows).to_csv(diag / "gapclose.csv", index=False)
    print(f"wrote diagnostics to {diag} ({len(summ)} summary rows)")


# ---- C1 == S1 verification (linear, full resolution) ----------------------
def verify_c1(cfg):
    soc = read_table(Path(cfg["processed_dir"]) / "soc_timeseries.parquet")
    meta = P.profile_meta(soc)
    s1 = pd.read_csv(cfg["stage1_summary"])
    print("\nC1 == S1 verification (linear, full windows, seed 0):")
    ok = True
    for d in cfg["directions"]:
        base = prepare_base(d, cfg, soc, meta)
        m = build_model("linear").fit(base["Xtr"], base["ytr"], base["Xva"], base["yva"],
                                      run_cfg_for(cfg, "linear", 0, smoke=False))
        mae = soc_report(base["yte"], m.predict(base["Xte"]))["mae"]
        ref = s1[(s1.experiment == d["name"]) & (s1.model == "linear")]["mae_mean"]
        ref = float(ref.iloc[0]) if len(ref) else float("nan")
        match = np.isfinite(ref) and abs(mae - ref) < 1e-3
        ok &= match
        print(f"  {d['name']}: C1 linear MAE {mae:.5f} vs S1 {ref:.5f} -> {'MATCH' if match else 'MISMATCH'}")
    print("C1==S1:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/soc_stage3.yaml")
    ap.add_argument("--out-dir", default="",
                    help="Override cfg['out_dir'] (the full-sweep root). Empty = use the "
                         "config's value. Smoke still appends '_smoke' to whatever this resolves to.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--directions", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-c1", action="store_true")
    ap.add_argument("--diagnostics-only", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    if args.verify_c1:
        return 0 if verify_c1(cfg) else 1

    out_dir = resolve_out_dir(cfg, args.smoke)
    if not args.diagnostics_only:
        out_dir = run_sweep(cfg, args.smoke,
                            [s for s in args.directions.split(",") if s],
                            [s for s in args.models.split(",") if s],
                            [s for s in args.conditions.split(",") if s], args.force)
    diagnostics(cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
