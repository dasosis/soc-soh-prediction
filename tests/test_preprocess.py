import numpy as np
from battery_bench.preprocess import Standardizer, make_windows
from battery_bench.schema import RAW_SOC_FEATURES


def test_standardizer_fits_on_train_only(soc_df):
    train = soc_df[soc_df["profile_id"] == "P1"]
    test = soc_df[soc_df["profile_id"] == "P2"]
    sc = Standardizer(RAW_SOC_FEATURES).fit(train)
    tr_t = sc.transform(train)
    # train mean ~0 after standardization; test uses train stats (mean != 0 ok)
    assert abs(tr_t["voltage_V"].mean()) < 1e-6
    _ = sc.transform(test)  # must not raise


def test_windows_never_cross_profile(soc_df):
    X, y, g = make_windows(soc_df, RAW_SOC_FEATURES, "soc", length=10, stride=5)
    assert X.shape[1] == 10 and X.shape[2] == len(RAW_SOC_FEATURES)
    # each window tagged with exactly one profile; counts match per-profile math
    # P1 and P2 each have 50 steps -> floor((50-10)/5)+1 = 9 windows each = 18
    assert len(y) == 18
    assert set(np.unique(g)) == {"P1", "P2"}
