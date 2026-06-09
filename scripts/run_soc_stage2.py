#!/usr/bin/env python
"""Stage 2 — fine-tuning data-efficiency ladder (target-domain adaptation).

For each cross-dataset direction it measures how target-TEST error falls as the
model is given a growing fraction of labeled TARGET data, comparing:
  * finetune          — continue-train the source net on the fraction (neural only);
  * recal_affine/bias — fit SOC' = a*y_hat + b (or a=1, bias only) on the
                        fraction's zero-shot predictions and apply to TEST.

This decomposes the Stage-1 transfer gap into removable systematic bias
(recalibration) vs genuine adaptation (fine-tuning). Reuses the Stage-1 pipeline
/ models / metrics and resolve_out_dir WITHOUT modifying any of them.

SHARED BUDGET: the TARGET is split once into disjoint TEST + POOL (fixed seed,
constant across all fractions/methods/models). At each fraction f the sampled
f*POOL profiles are the TOTAL target-label budget, shared identically by every
method: recalibration fits on the full sample, fine-tuning carves its
early-stopping val from INSIDE the same sample (fixed-epoch fallback when the
sample is below `finetune.min_profiles`). There is no off-budget external VAL, so
the x-axis fraction is the honest shared budget; total sampled profiles + windows
are reported per fraction. TEST is never used for fitting, recalibration, or
early stopping. The SOURCE scaler is fit on source-train only and kept FIXED
across fractions, so the curve isolates adaptation, not renormalization.

Usage:
  python scripts/run_soc_stage2.py --config configs/soc_stage2.yaml --smoke   # -> runs/soc_stage2_smoke/
  python scripts/run_soc_stage2.py --config configs/soc_stage2.yaml           # full -> runs/soc_stage2/
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

# reuse Stage-1 resolve_out_dir WITHOUT modifying it
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_soc_stage1 import resolve_out_dir  # noqa: E402

from battery_bench import soc_finetune as FT  # noqa: E402
from battery_bench import soc_pipeline as P  # noqa: E402
from battery_bench.metrics.regression import soc_report, r2  # noqa: E402
from battery_bench.models import build_model  # noqa: E402
from battery_bench.models.torch_models import seed_everything  # noqa: E402
from battery_bench.utils import read_table  # noqa: E402

RESULTS_COLUMNS = ["direction", "model", "method", "fraction", "seed",
                   "n_pool_profiles", "n_pool_windows", "n_test_windows",
                   "rmse", "mae", "max_error", "mape_pct", "r2", "signed_bias",
                   "recal_a", "recal_b", "train_time_s", "device", "smoke", "timestamp"]


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def src_cfg(cfg, model_name, seed, smoke):
    run = dict(cfg.get("deep_defaults", {}))
    run.update(cfg.get("model_overrides", {}).get(model_name, {}))
    run["seed"] = int(seed)
    if smoke:
        run["max_epochs"] = int(cfg["smoke"].get("max_epochs", 2))
    return run


# --------------------------------------------------------------------------
def _predict_module(module, device, X, batch=1024):
    module.eval()
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32).to(device)
    with torch.no_grad():
        if len(Xt) == 0:
            return np.empty(0)
        out = torch.cat([module(Xt[s:s + batch]) for s in range(0, len(Xt), batch)])
    return out.detach().cpu().numpy().ravel()


def continue_train(module, device, X, y, Xval, yval, lr, max_epochs, patience, batch, clip, seed):
    """Low-LR continue-training of an already-trained module on target data,
    early-stopping on the fixed target VAL. Modifies `module` in place."""
    seed_everything(seed)
    Xtr = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    ytr = torch.as_tensor(np.asarray(y), dtype=torch.float32)
    has_val = Xval is not None and len(Xval) > 0
    if has_val:
        Xv = torch.as_tensor(np.asarray(Xval), dtype=torch.float32).to(device)
        yv = torch.as_tensor(np.asarray(yval), dtype=torch.float32).to(device)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(int(seed))
    best, best_state, bad = float("inf"), None, 0
    n = len(Xtr)
    for _ in range(max_epochs):
        module.train()
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = loss_fn(module(Xtr[idx].to(device)), ytr[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(module.parameters(), clip)
            opt.step()
        if not has_val:
            continue
        module.eval()
        with torch.no_grad():
            vp = torch.cat([module(Xv[s:s + 1024]) for s in range(0, len(Xv), 1024)])
            vloss = float(loss_fn(vp, yv))
        if vloss < best - 1e-6:
            best, best_state, bad = vloss, copy.deepcopy(module.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        module.load_state_dict(best_state)


# --------------------------------------------------------------------------
def prepare_direction(direction, cfg, soc, meta, smoke):
    feats, target = cfg["features"], cfg["target"]
    length, stride = cfg["window"]["length"], cfg["window"]["stride"]
    manifest = json.loads(Path(direction["manifest"]).read_text())

    # source train/val (val for SOURCE early stopping); ignore manifest test (all target)
    st = cfg["source_train"]
    source_train, source_val, _ = P.split_cross(manifest, meta, st["val_frac"], st["split_seed"])

    # FIXED target partition (TEST + POOL only, constant across the ladder). There
    # is NO external VAL: fine-tuning carves its early-stopping val from inside the
    # sampled fraction so every method shares the same per-fraction label budget.
    ts = cfg["target_split"]
    test_t, pool_t = FT.split_target(meta, manifest["test_profiles"], ts["test_frac"], ts["seed"])

    scaler = P.fit_source_scaler(soc, source_train, feats)          # SOURCE-only, fixed
    Xtr, ytr, _ = P.make_split_windows(soc, source_train, scaler, feats, target, length, stride)
    Xvs, yvs, _ = P.make_split_windows(soc, source_val, scaler, feats, target, length, stride)
    Xte, yte, _ = P.make_split_windows(soc, test_t, scaler, feats, target, length, stride)
    Xpool, ypool, gpool = P.make_split_windows(soc, pool_t, scaler, feats, target, length, stride)

    if smoke:
        Xtr, ytr = P.subsample_windows(Xtr, ytr, cfg["smoke"]["max_windows_train"], seed=0)
        Xvs, yvs = P.subsample_windows(Xvs, yvs, cfg["smoke"]["max_windows_eval"], seed=1)
        Xte, yte = P.subsample_windows(Xte, yte, cfg["smoke"]["max_windows_eval"], seed=2)
        # subsample pool windows together with their group labels
        if len(Xpool) > cfg["smoke"]["max_windows_train"]:
            rng = np.random.default_rng(4)
            keep = np.sort(rng.choice(len(Xpool), cfg["smoke"]["max_windows_train"], replace=False))
            Xpool, ypool, gpool = Xpool[keep], ypool[keep], gpool[keep]

    return {
        "scaler": scaler, "splits": {"test": test_t, "pool": pool_t,
                                     "source_train": source_train, "source_val": source_val},
        "Xtr": Xtr, "ytr": ytr, "Xvs": Xvs, "yvs": yvs,
        "Xte": Xte, "yte": yte,
        "Xpool": Xpool, "ypool": ypool, "gpool": gpool,
    }


def methods_for(model_name, cfg):
    out = []
    if model_name in cfg["recalibrate_models"]:
        out += [f"recal_{m}" for m in cfg["recalibrate_modes"]]
    if model_name in cfg["finetune"]["models"]:
        out += ["finetune"]
    return out


def metrics_path(out_dir, direction, model, method, seed, frac):
    return out_dir / direction / model / method / f"seed_{seed}" / f"frac_{frac:.2f}.json"


def append_result(results_csv, row):
    new = not results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------
def run_sweep(cfg, smoke, dir_filter, model_filter, force):
    out_dir = resolve_out_dir(cfg, smoke)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"

    soc = read_table(Path(cfg["processed_dir"]) / "soc_timeseries.parquet")
    meta = P.profile_meta(soc)

    directions = [d for d in cfg["directions"] if not dir_filter or d["name"] in dir_filter]
    all_models = cfg["smoke"]["models"] if smoke else \
        sorted(set(cfg["recalibrate_models"]) | set(cfg["finetune"]["models"]))
    if model_filter:
        all_models = [m for m in all_models if m in model_filter]
    fractions = cfg["smoke"]["fractions"] if smoke else cfg["fractions"]
    seeds = cfg["smoke"]["seeds"] if smoke else cfg["seeds"]
    ft_cfg = cfg["finetune"]

    print(f"{'SMOKE' if smoke else 'FULL'} stage-2 | directions={[d['name'] for d in directions]} "
          f"| models={all_models} | fractions={fractions} | seeds={seeds}")

    for direction in directions:
        dname = direction["name"]
        t0 = time.time()
        D = prepare_direction(direction, cfg, soc, meta, smoke)
        print(f"\n=== {dname} | source_tr/val={len(D['Xtr'])}/{len(D['Xvs'])} "
              f"test/pool windows={len(D['Xte'])}/{len(D['Xpool'])} "
              f"| profiles test/pool={len(D['splits']['test'])}/{len(D['splits']['pool'])} "
              f"| prep {time.time()-t0:.1f}s ===")

        for model_name in all_models:
            methods = methods_for(model_name, cfg)
            if not methods:
                continue
            for seed in seeds:
                expected = [metrics_path(out_dir, dname, model_name, mth, seed, f)
                            for mth in methods for f in fractions]
                if all(p.exists() for p in expected) and not force:
                    print(f"  skip (done) {dname}/{model_name} seed {seed}")
                    continue

                # ---- train SOURCE model once ----
                run_cfg = src_cfg(cfg, model_name, seed, smoke)
                t1 = time.time()
                src = build_model(model_name).fit(D["Xtr"], D["ytr"], D["Xvs"], D["yvs"], run_cfg)
                src_time = time.time() - t1
                device = getattr(getattr(src, "device", None), "type", "cpu")
                yzs_test = src.predict(D["Xte"])            # zero-shot TEST predictions
                is_neural = hasattr(src, "module") and src.module is not None

                for frac in fractions:
                    # The sampled fraction is the TOTAL target-label budget at f,
                    # shared identically by every method. recal fits on all of it;
                    # finetune reserves an internal val carved from the same sample.
                    sampled = FT.sample_pool(D["splits"]["pool"], frac, seed)
                    mask = np.isin(D["gpool"], sampled) if sampled else np.zeros(len(D["gpool"]), bool)
                    Xp, yp = D["Xpool"][mask], D["ypool"][mask]

                    for method in methods:
                        mp = metrics_path(out_dir, dname, model_name, method, seed, frac)
                        if mp.exists() and not force:
                            continue
                        mp.parent.mkdir(parents=True, exist_ok=True)
                        a, b, tt = 1.0, 0.0, 0.0
                        extra = {}

                        if method.startswith("recal_"):
                            mode = method.split("_", 1)[1]
                            pool_pred = src.predict(Xp) if len(Xp) else np.empty(0)
                            y_hat, (a, b) = FT.recalibrate_test(pool_pred, yp, yzs_test, mode)
                        elif method == "finetune":
                            if frac == 0 or len(sampled) == 0 or not is_neural:
                                y_hat = yzs_test                      # anchor = zero-shot
                            else:
                                # carve early-stop val from INSIDE the sampled budget
                                ft_tr, ft_val, do_es = FT.split_finetune_train_val(
                                    sampled, int(seed), float(frac),
                                    val_frac=ft_cfg["internal_val_frac"],
                                    min_profiles=ft_cfg["min_profiles"])
                                Xtr_ft = D["Xpool"][np.isin(D["gpool"], ft_tr)]
                                ytr_ft = D["ypool"][np.isin(D["gpool"], ft_tr)]
                                if do_es and ft_val:
                                    vm = np.isin(D["gpool"], ft_val)
                                    Xv_ft, yv_ft = D["Xpool"][vm], D["ypool"][vm]
                                    ep = cfg["smoke"]["max_epochs"] if smoke else ft_cfg["max_epochs"]
                                else:
                                    Xv_ft, yv_ft = None, None        # fixed-epoch path
                                    ep = cfg["smoke"]["max_epochs"] if smoke else ft_cfg["fixed_epochs"]
                                extra = {"ft_train_profiles": len(ft_tr),
                                         "ft_val_profiles": len(ft_val), "ft_early_stop": bool(do_es)}
                                if len(Xtr_ft) == 0:                  # no windows (smoke subsample) -> anchor
                                    y_hat = yzs_test
                                else:
                                    ft_mod = copy.deepcopy(src.module)
                                    t2 = time.time()
                                    continue_train(ft_mod, src.device, Xtr_ft, ytr_ft, Xv_ft, yv_ft,
                                                   lr=run_cfg["lr"] * ft_cfg["lr_factor"], max_epochs=ep,
                                                   patience=ft_cfg["patience"],
                                                   batch=int(run_cfg.get("batch_size", 256)),
                                                   clip=float(run_cfg.get("grad_clip", 1.0)),
                                                   seed=int(seed))
                                    tt = time.time() - t2
                                    y_hat = _predict_module(ft_mod, src.device, D["Xte"])
                        else:
                            continue

                        rep = soc_report(D["yte"], y_hat)
                        rep["r2"] = r2(D["yte"], y_hat)
                        signed_bias = float(np.mean(np.asarray(y_hat) - np.asarray(D["yte"])))

                        rec = {"direction": dname, "model": model_name, "method": method,
                               "fraction": float(frac), "seed": int(seed),
                               "n_pool_profiles": len(sampled), "n_pool_windows": int(len(Xp)),
                               "n_test_windows": int(len(D["Xte"])), "signed_bias": signed_bias,
                               "recal_a": a, "recal_b": b, **extra,
                               "train_time_s": round(src_time if frac == 0 else tt, 3),
                               "device": device, "smoke": bool(smoke), **rep}
                        mp.write_text(json.dumps(rec, indent=2))
                        append_result(results_csv, {
                            **{k: rec.get(k) for k in RESULTS_COLUMNS if k != "timestamp"},
                            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")})

                print(f"  {model_name:14} seed {seed} | src {src_time:.1f}s | "
                      f"methods={methods} | zero-shot MAE {soc_report(D['yte'], yzs_test)['mae']:.4f}")
    return out_dir


# --------------------------------------------------------------------------
def aggregate_summary(out_dir):
    rows = [json.loads(p.read_text()) for p in out_dir.glob("*/*/*/seed_*/frac_*.json")]
    if not rows:
        print("no Stage-2 runs to summarize")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["direction", "model", "method", "fraction"], as_index=False)
    summary = g.agg(n_seeds=("seed", "nunique"),
                    mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                    rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                    bias_mean=("signed_bias", "mean"),
                    n_pool_windows=("n_pool_windows", "mean")).round(5)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nwrote {out_dir/'summary.csv'} ({len(summary)} rows)")
    return summary


def headline_and_curves(cfg, out_dir):
    spath = out_dir / "summary.csv"
    if not spath.exists():
        return
    summ = pd.read_csv(spath)
    diag = out_dir / "diagnostics"
    diag.mkdir(exist_ok=True)

    # within-target oracle MAE per model (from Stage 1)
    oracle = {}
    s1p = Path(cfg["stage1_summary"])
    if s1p.exists():
        s1 = pd.read_csv(s1p)
        for d in cfg["directions"]:
            sub = s1[s1["experiment"] == d["oracle"]].set_index("model")["mae_mean"]
            oracle[d["name"]] = sub.to_dict()

    tgt = cfg.get("gap_closure_target", 0.8)
    head_rows, share_rows = [], []
    for (dname, model, method), grp in summ.groupby(["direction", "model", "method"]):
        grp = grp.sort_values("fraction")
        zs = float(grp[grp.fraction == 0]["mae_mean"].iloc[0]) if (grp.fraction == 0).any() else np.nan
        orc = oracle.get(dname, {}).get(model, np.nan)
        gap = zs - orc if np.isfinite(orc) else np.nan
        frac80 = np.nan
        if np.isfinite(gap) and gap > 0:
            thresh = zs - tgt * gap
            below = grp[(grp.fraction > 0) & (grp.mae_mean <= thresh)]
            frac80 = float(below["fraction"].min()) if len(below) else np.nan
        head_rows.append({"direction": dname, "model": model, "method": method,
                          "zeroshot_mae": round(zs, 5), "oracle_mae": round(orc, 5) if np.isfinite(orc) else np.nan,
                          "gap": round(gap, 5) if np.isfinite(gap) else np.nan,
                          f"frac_to_close_{int(tgt*100)}pct": frac80})
        for _, r in grp.iterrows():
            share = (zs - r["mae_mean"]) / gap if (np.isfinite(gap) and gap > 0) else np.nan
            share_rows.append({"direction": dname, "model": model, "method": method,
                               "fraction": r["fraction"], "mae_mean": r["mae_mean"],
                               "gap_share_closed": round(share, 4) if np.isfinite(share) else np.nan})

    pd.DataFrame(head_rows).to_csv(diag / "headline_gap_closure.csv", index=False)
    pd.DataFrame(share_rows).to_csv(diag / "gap_share_by_fraction.csv", index=False)

    # data-efficiency curves: one panel per direction
    cmap = plt.get_cmap("tab10")
    for d in cfg["directions"]:
        dname = d["name"]
        sub = summ[summ.direction == dname]
        if sub.empty:
            continue
        plot_models = list(dict.fromkeys(cfg["finetune"]["models"] + ["linear"]))
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, model in enumerate(plot_models):
            color = cmap(i % 10)
            for method, ls, mk in [("finetune", "-", "o"), ("recal_affine", "--", "x")]:
                c = sub[(sub.model == model) & (sub.method == method)].sort_values("fraction")
                if c.empty:
                    continue
                ax.plot(c.fraction, c.mae_mean, ls=ls, marker=mk, color=color,
                        label=f"{model}:{method.replace('recal_','recal-')}")
            orc = oracle.get(dname, {}).get(model)
            if orc is not None and np.isfinite(orc):
                ax.axhline(orc, color=color, ls=":", lw=0.8, alpha=0.7)
        ax.set_xlabel("fraction of target POOL (labeled)")
        ax.set_ylabel("target-TEST MAE (mean over seeds)")
        ax.set_title(f"{dname}\ndata-efficiency: fine-tune vs recalibration "
                     f"(dotted = within-target oracle)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(diag / f"data_efficiency_{dname}.png", dpi=120)
        plt.close(fig)
    print(f"wrote diagnostics to {diag}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/soc_stage2.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--directions", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--diagnostics-only", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dir_filter = [s for s in args.directions.split(",") if s]
    model_filter = [s for s in args.models.split(",") if s]
    out_dir = resolve_out_dir(cfg, args.smoke)

    if not args.diagnostics_only:
        out_dir = run_sweep(cfg, args.smoke, dir_filter, model_filter, args.force)
    aggregate_summary(out_dir)
    headline_and_curves(cfg, out_dir)

    spath = out_dir / "summary.csv"
    if spath.exists():
        print("\n================ STAGE-2 SUMMARY ================")
        print(pd.read_csv(spath).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
