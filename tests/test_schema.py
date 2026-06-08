import pytest
from battery_bench import schema


def test_cells_ok(cells_df):
    schema.validate_cells(cells_df)


def test_soc_ok(soc_df):
    schema.validate_soc_timeseries(soc_df)


def test_soh_ok(soh_df):
    schema.validate_soh_cycles(soh_df)


def test_soc_rejects_percent_units(soc_df):
    bad = soc_df.copy()
    bad["soc"] = bad["soc"] * 100.0  # someone used % instead of fraction
    with pytest.raises(ValueError):
        schema.validate_soc_timeseries(bad)


def test_soh_rejects_bad_reference(soh_df):
    bad = soh_df.copy()
    bad.loc[0, "soh"] = 5.0
    with pytest.raises(ValueError):
        schema.validate_soh_cycles(bad)
