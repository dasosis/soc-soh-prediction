import pytest
from battery_bench.splits import group_holdout, leave_one_group_out, forward_cycle_split


def test_group_holdout_disjoint(cells_df):
    sp = group_holdout(cells_df, "dataset", ["CALCE"])
    assert set(sp["train_cells"]).isdisjoint(sp["test_cells"])
    assert sp["test_cells"] == ["CALCE_CS2_35"]


def test_leave_one_dataset_out_covers_all(cells_df):
    seen = {v for v, _ in leave_one_group_out(cells_df, "dataset")}
    assert seen == set(cells_df["dataset"].unique())


def test_group_holdout_rejects_absent_value(cells_df):
    with pytest.raises(ValueError):
        group_holdout(cells_df, "dataset", ["DOES_NOT_EXIST"])


def test_forward_cycle_no_leakage(soh_df):
    sp = forward_cycle_split(soh_df, train_frac=0.6)
    tr = soh_df.loc[sp["train_idx"]]
    te = soh_df.loc[sp["test_idx"]]
    for cell in soh_df["cell_id"].unique():
        a = tr[tr["cell_id"] == cell]["cycle_index"].max()
        b = te[te["cell_id"] == cell]["cycle_index"].min()
        assert a < b  # past strictly precedes future
