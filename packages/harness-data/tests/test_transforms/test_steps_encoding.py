"""Tests for encoding transform steps: encode, bin, datetime."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from harness.data.transforms.steps import encode, bin as bin_step, datetime as datetime_step


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------

def test_encode_frequency(sample_df):
    result = encode.step(sample_df, {"column": "grade"})
    assert "grade_encoded" in result.columns
    # Each grade appears: A=2, B=2, C=1 out of 5 rows
    grade_a_freq = result.loc[result["grade"] == "A", "grade_encoded"].iloc[0]
    assert abs(grade_a_freq - 2 / 5) < 1e-6
    grade_c_freq = result.loc[result["grade"] == "C", "grade_encoded"].iloc[0]
    assert abs(grade_c_freq - 1 / 5) < 1e-6


def test_encode_frequency_custom_output(sample_df):
    result = encode.step(sample_df, {"column": "grade", "output": "grade_freq"})
    assert "grade_freq" in result.columns
    assert "grade_encoded" not in result.columns


def test_encode_ordinal(sample_df):
    result = encode.step(sample_df, {"column": "grade", "method": "ordinal"})
    assert "grade_encoded" in result.columns
    # sorted unique grades: A=0, B=1, C=2
    assert result.loc[result["grade"] == "A", "grade_encoded"].iloc[0] == 0
    assert result.loc[result["grade"] == "B", "grade_encoded"].iloc[0] == 1
    assert result.loc[result["grade"] == "C", "grade_encoded"].iloc[0] == 2


def test_encode_missing_column(sample_df):
    with pytest.raises(ValueError, match="encode step requires"):
        encode.step(sample_df, {})


def test_encode_unknown_method(sample_df):
    with pytest.raises(ValueError, match="Unknown encode method"):
        encode.step(sample_df, {"column": "grade", "method": "bogus"})


# ---------------------------------------------------------------------------
# bin
# ---------------------------------------------------------------------------

def test_bin_quantile(sample_df):
    result = bin_step.step(sample_df, {"column": "score", "method": "quantile", "n_bins": 3})
    assert "score_binned" in result.columns
    assert result["score_binned"].notna().all()
    # bins should be 0-indexed integers
    assert result["score_binned"].min() >= 0


def test_bin_uniform(sample_df):
    result = bin_step.step(sample_df, {"column": "score", "method": "uniform", "n_bins": 4})
    assert "score_binned" in result.columns
    # uniform cut into 4 bins — values should all be 0-3
    vals = result["score_binned"].dropna().unique()
    assert all(v in [0.0, 1.0, 2.0, 3.0] for v in vals)


def test_bin_custom_output(sample_df):
    result = bin_step.step(
        sample_df,
        {"column": "score", "method": "quantile", "n_bins": 2, "output": "score_bucket"},
    )
    assert "score_bucket" in result.columns
    assert "score_binned" not in result.columns


def test_bin_missing_column(sample_df):
    with pytest.raises(ValueError, match="bin step requires"):
        bin_step.step(sample_df, {})


def test_bin_unknown_method(sample_df):
    with pytest.raises(ValueError, match="Unknown bin method"):
        bin_step.step(sample_df, {"column": "score", "method": "bogus"})


# ---------------------------------------------------------------------------
# datetime
# ---------------------------------------------------------------------------

@pytest.fixture
def date_df():
    return pd.DataFrame({
        "ts": pd.to_datetime([
            "2023-03-15 08:30:00",
            "2023-06-21 14:00:00",
            "2024-01-01 00:00:00",
        ]),
        "value": [1.0, 2.0, 3.0],
    })


def test_datetime_extract_year(date_df):
    result = datetime_step.step(date_df, {"column": "ts", "extract": ["year"]})
    assert "ts_year" in result.columns
    assert list(result["ts_year"]) == [2023, 2023, 2024]


def test_datetime_extract_month(date_df):
    result = datetime_step.step(date_df, {"column": "ts", "extract": ["month"]})
    assert "ts_month" in result.columns
    assert list(result["ts_month"]) == [3, 6, 1]


def test_datetime_extract_dayofweek(date_df):
    result = datetime_step.step(date_df, {"column": "ts", "extract": ["dayofweek"]})
    assert "ts_dayofweek" in result.columns
    # 2023-03-15 is Wednesday = 2
    assert result["ts_dayofweek"].iloc[0] == 2


def test_datetime_extract_multiple(date_df):
    result = datetime_step.step(date_df, {"column": "ts", "extract": ["year", "month", "day"]})
    assert "ts_year" in result.columns
    assert "ts_month" in result.columns
    assert "ts_day" in result.columns


def test_datetime_cyclical_month(date_df):
    result = datetime_step.step(date_df, {"column": "ts", "cyclical": ["month"]})
    assert "ts_month_sin" in result.columns
    assert "ts_month_cos" in result.columns
    # sin and cos should be in [-1, 1]
    assert result["ts_month_sin"].between(-1, 1).all()
    assert result["ts_month_cos"].between(-1, 1).all()


def test_datetime_missing_column(date_df):
    with pytest.raises(ValueError, match="datetime step requires"):
        datetime_step.step(date_df, {})


def test_datetime_unknown_component(date_df):
    with pytest.raises(ValueError, match="Unknown datetime component"):
        datetime_step.step(date_df, {"column": "ts", "extract": ["bogus"]})
