#!/usr/bin/env python
"""Stage 3 WIDE combiner — unify the full 9-model roster for C2 + C4.

Reads, strictly READ-ONLY:
  * results/stage3/results.csv      — the 5 models Stage 3 already ran (C2/C4 kept)
  * runs/soc_stage3_wide/results.csv — the 4 models added by the wide sweep
  * results/stage1/summary.csv       — C1 (cross zero-shot) + oracle (within-dataset)

Writes results/stage3_wide/{summary.csv, gapclose.csv, bias_by_condition.csv}.
NEVER writes under results/stage3/ (asserted).

gapclose = (C1 - cond) / (C1 - oracle), per direction/model/condition, where
  C1     = Stage-1 cross-dataset zero-shot MAE for that direction+model
  oracle = Stage-1 within-dataset MAE for the target chemistry (direction.oracle_exp)
  cond   = mean MAE over seeds for the label-free condition (C2_ocv_mean / C4_coral)

Usage:
  python scripts/combine_stage3_wide.py
  python scripts/combine_stage3_wide.py --wide-results runs/soc_stage3_wide_smoke/results.csv \
         --out-dir results/stage3_wide_smoke --allow-smoke      # wiring check only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

CONDITIONS = ["C2_ocv_mean", "C4_coral"]
FULL_ROSTER = ["linear", "mlp", "lstm", "gru", "patchtst",            # already in Stage 3
               "bilstm", "cnn_bilstm_attn", "tcn", "transformer"]     # added by the wide sweep


def _read_seed_rows(path, allow_smoke):
    """Per-seed metric rows for the two wide conditions from a results.csv."""
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["condition"].isin(CONDITIONS)].copy()
    if not allow_smoke and "smoke" in df.columns:
        df = df[~df["smoke"].astype(str).str.lower().isin(["true", "1"])]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/soc_stage3_wide.yaml")
    ap.add_argument("--stage3-results", default="results/stage3/results.csv")
    ap.add_argument("--wide-results", default="runs/soc_stage3_wide/results.csv")
    ap.add_argument("--stage1-summary", default="results/stage1/summary.csv")
    ap.add_argument("--out-dir", default="results/stage3_wide")
    ap.add_argument("--allow-smoke", action="store_true",
                    help="Keep smoke rows (wiring check only; default drops them).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    assert out_dir.resolve() != Path("results/stage3").resolve(), \
        "refusing to write into results/stage3 (the protected Stage-3 artifact)"

    cfg = yaml.safe_load(Path(args.config).read_text())
    # direction name -> within-dataset oracle experiment name (target chemistry)
    oracle_exp = {d["name"]: d["oracle_exp"] for d in cfg["directions"]}

    # --- per-seed rows: 5 models from Stage 3 (read-only) + 4 from the wide sweep ---
    s3 = _read_seed_rows(args.stage3_results, args.allow_smoke)
    wide = _read_seed_rows(args.wide_results, args.allow_smoke)
    rows = pd.concat([s3, wide], ignore_index=True)
    if rows.empty:
        raise SystemExit("no C2/C4 rows found in either results.csv — nothing to combine")
    rows = rows.drop_duplicates(["direction", "condition", "model", "seed"], keep="last")

    # --- summary.csv (mean +/- std over seeds, per direction/model/condition) ---
    summ = (rows.groupby(["direction", "model", "condition"], as_index=False)
                .agg(n_seeds=("seed", "nunique"),
                     mae_mean=("mae", "mean"), mae_std=("mae", "std"),
                     rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                     bias_mean=("signed_bias", "mean"), bias_std=("signed_bias", "std"))
                .round(5)
                .sort_values(["direction", "condition", "model"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    summ.to_csv(out_dir / "summary.csv", index=False)

    # --- bias_by_condition.csv (per direction/condition) ---
    bias = (rows.groupby(["direction", "condition"], as_index=False)
                .agg(bias_mean=("signed_bias", "mean"), bias_std=("signed_bias", "std"),
                     mae_mean=("mae", "mean"))
                .round(5)
                .sort_values(["direction", "condition"]))
    bias.to_csv(out_dir / "bias_by_condition.csv", index=False)

    # --- gapclose.csv: (C1 - cond)/(C1 - oracle), C1 + oracle from Stage 1 ---
    s1 = pd.read_csv(args.stage1_summary).set_index(["experiment", "model"])["mae_mean"]

    def s1_mae(exp, model):
        return float(s1[(exp, model)]) if (exp, model) in s1.index else float("nan")

    gc_rows = []
    for (direction, model, cond), g in summ.groupby(["direction", "model", "condition"]):
        c1 = s1_mae(direction, model)                       # cross-dataset zero-shot
        oracle = s1_mae(oracle_exp[direction], model)       # within-dataset target
        cond_mae = float(g["mae_mean"].iloc[0])
        denom = c1 - oracle
        gapclose = (c1 - cond_mae) / denom if (pd.notna(denom) and denom != 0) else float("nan")
        gc_rows.append({"direction": direction, "model": model, "condition": cond,
                        "c1_mae": round(c1, 5), "cond_mae": round(cond_mae, 5),
                        "oracle_mae": round(oracle, 5),
                        "gapclose": round(gapclose, 4) if pd.notna(gapclose) else float("nan")})
    gc = (pd.DataFrame(gc_rows)
            .sort_values(["direction", "condition", "model"]))
    gc.to_csv(out_dir / "gapclose.csv", index=False)

    found = sorted(rows["model"].unique())
    missing = [m for m in FULL_ROSTER if m not in found]
    print(f"wrote {out_dir}/ : summary({len(summ)}) gapclose({len(gc)}) "
          f"bias_by_condition({len(bias)}) | models={found}")
    if missing:
        print(f"  WARNING: roster incomplete, missing {missing} "
              f"(expected once the full wide sweep + Stage 3 are both present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
