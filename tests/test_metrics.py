import numpy as np
from battery_bench.metrics import rmse, mae, r2, soc_report, soh_report


def test_perfect_prediction():
    y = np.array([0.1, 0.5, 0.9])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_reports_have_expected_keys():
    y = np.array([0.8, 0.85, 0.9])
    p = np.array([0.79, 0.86, 0.88])
    assert set(soc_report(y, p)) == {"rmse", "mae", "max_error", "mape_pct"}
    assert "r2" in soh_report(y, p)
