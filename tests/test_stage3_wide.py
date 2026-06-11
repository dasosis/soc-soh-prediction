"""Stage 3 WIDE — guards for the separate 9-model C2/C4 artifact.

Covers:
  * the optional --out-dir override does NOT change the default config's location
    (configs/soc_stage3.yaml must still resolve to runs/soc_stage3);
  * the 4 newly-added models accept a 4-channel (N, 100, 4) OCV input — in
    particular the conv-based cnn_bilstm_attn and tcn read the channel count from
    the data, not a hardcoded 3;
  * the combiner unifies the roster and never writes under results/stage3.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_soc_stage1 import resolve_out_dir  # noqa: E402
from battery_bench.models import build_model  # noqa: E402

WIDE_MODELS = ["bilstm", "cnn_bilstm_attn", "tcn", "transformer"]


def _cfg(name):
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_default_stage3_config_unchanged():
    # The headline invariant of STEP 1: defaults are untouched.
    assert resolve_out_dir(_cfg("soc_stage3.yaml"), smoke=False) == Path("runs/soc_stage3")
    assert resolve_out_dir(_cfg("soc_stage3.yaml"), smoke=True) == Path("runs/soc_stage3_smoke")


def test_wide_config_resolves_to_wide():
    cfg = _cfg("soc_stage3_wide.yaml")
    assert resolve_out_dir(cfg, smoke=False) == Path("runs/soc_stage3_wide")
    assert resolve_out_dir(cfg, smoke=True) == Path("runs/soc_stage3_wide_smoke")
    assert cfg["models"] == WIDE_MODELS
    assert cfg["conditions"] == ["C2_ocv_mean", "C4_coral"]


def test_out_dir_override_in_runner():
    # Simulating `--out-dir runs/soc_stage3_wide`: only the override changes location.
    cfg = _cfg("soc_stage3.yaml")
    cfg["out_dir"] = "runs/soc_stage3_wide"
    assert resolve_out_dir(cfg, smoke=True) == Path("runs/soc_stage3_wide_smoke")


def test_wide_models_accept_4_channel_ocv_input():
    rng = np.random.default_rng(0)
    n, L, C = 24, 100, 4                      # 4 = V/I/T + appended OCV-SOC channel
    X = rng.standard_normal((n, L, C)).astype(np.float32)
    y = rng.random(n).astype(np.float32)
    cfg = dict(_cfg("soc_stage3_wide.yaml")["deep_defaults"], max_epochs=1, device="cpu", seed=0)
    overrides = _cfg("soc_stage3_wide.yaml")["model_overrides"]
    for name in WIDE_MODELS:
        rc = dict(cfg, **overrides.get(name, {}))
        m = build_model(name).fit(X, y, X[:8], y[:8], rc)
        # build_module saw n_features straight from X.shape[-1]; predict must run on 4ch.
        assert m.module is not None
        yhat = m.predict(X)
        assert yhat.shape == (n,)
        assert np.isfinite(yhat).all()


def test_combine_unifies_and_never_writes_stage3(tmp_path, monkeypatch):
    import importlib
    combine = importlib.import_module("combine_stage3_wide")

    direction = "cross_A_train_PAN_test_LG"
    oracle = "within_LG"

    def seed_rows(models, conditions):
        rows = []
        for mdl in models:
            for cond in conditions:
                for sd in range(2):
                    rows.append(dict(direction=direction, condition=cond, model=mdl,
                                     seed=sd, mae=0.08, rmse=0.10, signed_bias=0.01,
                                     smoke=False))
        return pd.DataFrame(rows)

    s3 = tmp_path / "stage3_results.csv"
    # Stage 3 file also carries C1/C3 rows that must be ignored by the combiner.
    pd.concat([seed_rows(["linear", "lstm"], ["C1_baseline", "C2_ocv_mean", "C4_coral"]),
               seed_rows(["lstm"], ["C3_ocv_dischg"])]).to_csv(s3, index=False)
    wide = tmp_path / "wide_results.csv"
    seed_rows(WIDE_MODELS, ["C2_ocv_mean", "C4_coral"]).to_csv(wide, index=False)

    s1 = tmp_path / "stage1_summary.csv"
    pd.DataFrame([{"experiment": direction, "model": m, "mae_mean": 0.10} for m in
                  ["linear", "lstm", *WIDE_MODELS]]
                 + [{"experiment": oracle, "model": m, "mae_mean": 0.02} for m in
                    ["linear", "lstm", *WIDE_MODELS]]).to_csv(s1, index=False)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"directions": [
        {"name": direction, "oracle_exp": oracle}]}))

    out = tmp_path / "stage3_wide"
    monkeypatch.setattr(sys, "argv", [
        "combine_stage3_wide.py", "--config", str(cfg), "--stage3-results", str(s3),
        "--wide-results", str(wide), "--stage1-summary", str(s1), "--out-dir", str(out)])
    assert combine.main() == 0

    summ = pd.read_csv(out / "summary.csv")
    # all 6 models (2 from stage3 + 4 wide), only C2/C4, no C1/C3 leakage
    assert set(summ["model"]) == {"linear", "lstm", *WIDE_MODELS}
    assert set(summ["condition"]) == {"C2_ocv_mean", "C4_coral"}

    gc = pd.read_csv(out / "gapclose.csv")
    # gapclose = (C1 - cond)/(C1 - oracle) = (0.10 - 0.08)/(0.10 - 0.02) = 0.25
    assert np.allclose(gc["gapclose"], 0.25)
    assert {"c1_mae", "cond_mae", "oracle_mae"}.issubset(gc.columns)

    assert (out / "bias_by_condition.csv").exists()
    # never wrote the protected artifact
    assert not (tmp_path / "stage3").exists()
