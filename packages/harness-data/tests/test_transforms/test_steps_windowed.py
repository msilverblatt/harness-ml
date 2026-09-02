"""Tests for windowed operation steps: rolling, lag, ewm, diff, trend."""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from harness.data.transforms.steps import rolling, lag, ewm, diff, trend


# ---------------------------------------------------------------------------
# rolling
# ---------------------------------------------------------------------------

def test_rolling_mean(numeric_df):
    result = rolling.step(numeric_df, {
        "window": 2,
        "aggs": {"points_ma": "points:mean"},
        "order_by": "period",
        "keys": ["entity_id"],
    })
    assert "points_ma" in result.columns
    assert result["points_ma"].notna().sum() > 0


def test_rolling_sum(numeric_df):
    result = rolling.step(numeric_df, {
        "window": 2,
        "aggs": {"points_sum": "points:sum"},
        "order_by": "period",
        "keys": ["entity_id"],
    })
    assert "points_sum" in result.columns
    # For entity_id=1, period=2: sum of rows 1 and 2 = 10+15 = 25
    row = result[(result["entity_id"] == 1) & (result["period"] == 2)]
    assert row["points_sum"].iloc[0] == pytest.approx(25.0)


def test_rolling_no_keys(numeric_df):
    result = rolling.step(numeric_df, {
        "window": 3,
        "aggs": {"points_mean3": "points:mean"},
    })
    assert "points_mean3" in result.columns


def test_rolling_missing_params(numeric_df):
    with pytest.raises(ValueError, match="rolling step requires"):
        rolling.step(numeric_df, {"window": 3})

    with pytest.raises(ValueError, match="rolling step requires"):
        rolling.step(numeric_df, {"aggs": {"x": "points:mean"}})


def test_rolling_bad_spec(numeric_df):
    with pytest.raises(ValueError, match="Rolling agg spec must be"):
        rolling.step(numeric_df, {"window": 2, "aggs": {"x": "points"}})


# ---------------------------------------------------------------------------
# lag
# ---------------------------------------------------------------------------

def test_lag_first_row_is_nan(numeric_df):
    result = lag.step(numeric_df, {
        "columns": {"points_lag1": "points:1"},
        "order_by": "period",
        "keys": ["entity_id"],
    })
    assert "points_lag1" in result.columns
    # First period per entity should be NaN
    first_rows = result[result["period"] == 1]
    assert first_rows["points_lag1"].isna().all()


def test_lag_correct_values(numeric_df):
    result = lag.step(numeric_df, {
        "columns": {"points_lag1": "points:1"},
        "order_by": "period",
        "keys": ["entity_id"],
    })
    # For entity_id=1 period=2, lag should equal period=1 points (10.0)
    row_period2 = result[(result["entity_id"] == 1) & (result["period"] == 2)]
    assert row_period2["points_lag1"].iloc[0] == pytest.approx(10.0)


def test_lag_no_keys(numeric_df):
    result = lag.step(numeric_df, {"columns": {"points_lag1": "points:1"}})
    assert "points_lag1" in result.columns
    assert pd.isna(result["points_lag1"].iloc[0])


def test_lag_missing_columns(numeric_df):
    with pytest.raises(ValueError, match="lag step requires"):
        lag.step(numeric_df, {})


def test_lag_bad_spec(numeric_df):
    with pytest.raises(ValueError, match="Lag spec must be"):
        lag.step(numeric_df, {"columns": {"x": "points"}})


# ---------------------------------------------------------------------------
# ewm
# ---------------------------------------------------------------------------

def test_ewm_mean(numeric_df):
    result = ewm.step(numeric_df, {
        "span": 3,
        "aggs": {"points_ewm": "points:mean"},
        "order_by": "period",
        "keys": ["entity_id"],
    })
    assert "points_ewm" in result.columns
    assert result["points_ewm"].notna().all()


def test_ewm_no_keys(numeric_df):
    result = ewm.step(numeric_df, {
        "span": 3,
        "aggs": {"points_ewm": "points:mean"},
    })
    assert "points_ewm" in result.columns


def test_ewm_missing_params(numeric_df):
    with pytest.raises(ValueError, match="ewm step requires"):
        ewm.step(numeric_df, {"span": 3})

    with pytest.raises(ValueError, match="ewm step requires"):
        ewm.step(numeric_df, {"aggs": {"x": "points:mean"}})


def test_ewm_bad_spec(numeric_df):
    with pytest.raises(ValueError, match="EWM spec must be"):
        ewm.step(numeric_df, {"span": 3, "aggs": {"x": "points"}})


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

def test_diff_first_difference(numeric_df):
    result = diff.step(numeric_df, {
        "columns": {"points_diff": "points:1"},
        "order_by": "period",
        "keys": ["entity_id"],
    })
    assert "points_diff" in result.columns
    # First period per entity should be NaN
    first_rows = result[result["period"] == 1]
    assert first_rows["points_diff"].isna().all()
    # entity_id=1: period2 - period1 = 15 - 10 = 5
    row = result[(result["entity_id"] == 1) & (result["period"] == 2)]
    assert row["points_diff"].iloc[0] == pytest.approx(5.0)


def test_diff_pct_change(numeric_df):
    result = diff.step(numeric_df, {
        "columns": {"points_pct": "points:1"},
        "order_by": "period",
        "keys": ["entity_id"],
        "pct": True,
    })
    assert "points_pct" in result.columns
    # entity_id=1: (15-10)/10 = 0.5
    row = result[(result["entity_id"] == 1) & (result["period"] == 2)]
    assert row["points_pct"].iloc[0] == pytest.approx(0.5)


def test_diff_no_keys(numeric_df):
    result = diff.step(numeric_df, {"columns": {"points_diff": "points:1"}})
    assert "points_diff" in result.columns
    assert pd.isna(result["points_diff"].iloc[0])


def test_diff_missing_columns(numeric_df):
    with pytest.raises(ValueError, match="diff step requires"):
        diff.step(numeric_df, {})


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------

def test_trend_slope(numeric_df):
    result = trend.step(numeric_df, {
        "window": 3,
        "columns": {"points_trend": "points"},
        "order_by": "period",
        "keys": ["entity_id"],
    })
    assert "points_trend" in result.columns
    # With window=3 and 3 periods per entity, all should be computed
    non_nan = result["points_trend"].dropna()
    assert len(non_nan) > 0


def test_trend_no_keys(numeric_df):
    result = trend.step(numeric_df, {
        "window": 3,
        "columns": {"points_trend": "points"},
    })
    assert "points_trend" in result.columns


def test_trend_missing_params(numeric_df):
    with pytest.raises(ValueError, match="trend step requires"):
        trend.step(numeric_df, {"window": 3})

    with pytest.raises(ValueError, match="trend step requires"):
        trend.step(numeric_df, {"columns": {"x": "points"}})


def test_trend_positive_slope():
    # Monotonically increasing series -> slope > 0
    df = pd.DataFrame({
        "entity_id": [1, 1, 1, 1, 1],
        "period": [1, 2, 3, 4, 5],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    result = trend.step(df, {
        "window": 3,
        "columns": {"value_trend": "value"},
        "order_by": "period",
        "keys": ["entity_id"],
    })
    # Last row slope should be 1.0 (perfect linear increase)
    last_row = result[result["period"] == 5]
    assert last_row["value_trend"].iloc[0] == pytest.approx(1.0)
