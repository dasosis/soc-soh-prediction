"""Stage 1 model-layer tests: every registered model fits+predicts the right
shape on synthetic windows, and the cross-dataset scaler is fit on source only.
"""
import numpy as np
import pandas as pd
import pytest

from battery_bench import soc_pipeline as P
from battery_bench.models import build_model, list_models

FEATS = ["voltage_V", "current_A", "temperature_C"]


@pytest.fixture
def windows():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 24, 3)).astype("float32")
    y = rng.random(50).astype("float32")
    return X, y


@pytest.mark.parametrize("name", list_models())
def test_model_fit_predict_shape(name, windows):
    X, y = windows
    cfg = dict(seed=0, max_epochs=2, batch_size=16, device="cpu", patience=2)
    model = build_model(name).fit(X, y, X[:10], y[:10], cfg)
    pred = model.predict(X[:12])
    assert pred.shape == (12,)
    assert np.isfinite(pred).all()
    assert model.n_params > 0
    assert model.name == name


def _toy_soc():
    """Two datasets with deliberately different feature scales: source 'PAN' ~+10,
    target 'LG' ~-10, so a source-only scaler is distinguishable from pooled."""
    rows = []
    for ds, prof, base in [("PAN", "PAN_p1", 10.0), ("PAN", "PAN_p2", 10.0), ("LG", "LG_p1", -10.0)]:
        for t in range(120):
            rows.append(dict(dataset=ds, cell_id=ds, profile_id=prof,
                             nominal_temperature_C=25.0, drive_cycle="US06",
                             timestep=t, time_s=float(t),
                             voltage_V=base + 0.001 * t, current_A=base,
                             temperature_C=base, soc=1.0 - 0.005 * t))
    return pd.DataFrame(rows)


def test_cross_scaler_fit_on_source_only():
    soc = _toy_soc()
    meta = P.profile_meta(soc)
    manifest = {"train_profiles": ["PAN_p1", "PAN_p2"], "test_profiles": ["LG_p1"]}
    train, val, test = P.split_cross(manifest, meta, val_frac=0.2, seed=0)

    # no target profile may leak into train or val; test is the target
    assert set(train) | set(val) <= {"PAN_p1", "PAN_p2"}
    assert test == ["LG_p1"]

    scaler = P.fit_source_scaler(soc, train, FEATS)
    # source current mean is ~+10; pooled would be ~+3.3; target ~-10.
    assert scaler.mean_["current_A"] > 5.0
    src_mean = soc[soc["profile_id"].isin(train)]["current_A"].mean()
    assert abs(scaler.mean_["current_A"] - src_mean) < 1e-9


def test_within_split_no_overlap_all_temps_in_train():
    rng_rows = []
    for ds in ["LG_HG2"]:
        for temp in [0.0, 25.0, 40.0]:
            for p in range(4):
                pid = f"{ds}__{int(temp)}__{p}"
                for t in range(50):
                    rng_rows.append(dict(dataset=ds, cell_id=ds, profile_id=pid,
                                         nominal_temperature_C=temp, drive_cycle="US06",
                                         timestep=t, time_s=float(t), voltage_V=3.7,
                                         current_A=-1.0, temperature_C=temp, soc=1 - 0.01 * t))
    soc = pd.DataFrame(rng_rows)
    meta = P.profile_meta(soc)
    train, val, test = P.split_within(meta, "LG_HG2", test_frac=0.25, val_frac=0.25, seed=0)
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    train_temps = set(meta[meta.profile_id.isin(train)]["nominal_temperature_C"])
    assert train_temps == {0.0, 25.0, 40.0}  # every temperature represented in train
