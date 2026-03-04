"""Tests for the 10 new rolling aggregation functions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.core.runner.schema import RollingStep
from easyml.core.runner.view_executor import execute_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(vals: list[float], n_groups: int = 1) -> pd.DataFrame:
    """Build a simple DataFrame with one group and sequential ordering."""
    n = len(vals)
    return pd.DataFrame({
        "group": ["A"] * n,
        "order": list(range(1, n + 1)),
        "val": vals,
    })


def _roll(df: pd.DataFrame, agg_name: str, window: int = 3,
          min_periods: int | None = None) -> pd.DataFrame:
    """Execute a rolling step for val:{agg_name} and return the result."""
    step = RollingStep(
        keys=["group"],
        order_by="order",
        window=window,
        aggs={f"val_{agg_name}": f"val:{agg_name}"},
        min_periods=min_periods,
    )
    return execute_step(df, step)


# ---------------------------------------------------------------------------
# median
# ---------------------------------------------------------------------------


class TestRollingMedian:
    def test_basic(self):
        df = _make_df([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _roll(df, "median")
        col = result["val_median"]
        # Window=3, min_periods=3: first 2 NaN
        assert pd.isna(col.iloc[0])
        assert pd.isna(col.iloc[1])
        assert col.iloc[2] == pytest.approx(20.0)  # median(10,20,30)
        assert col.iloc[3] == pytest.approx(30.0)  # median(20,30,40)
        assert col.iloc[4] == pytest.approx(40.0)  # median(30,40,50)

    def test_even_window(self):
        df = _make_df([1.0, 3.0, 2.0, 4.0])
        result = _roll(df, "median", window=2)
        col = result["val_median"]
        assert pd.isna(col.iloc[0])
        assert col.iloc[1] == pytest.approx(2.0)   # median(1,3)
        assert col.iloc[2] == pytest.approx(2.5)   # median(3,2)
        assert col.iloc[3] == pytest.approx(3.0)   # median(2,4)


# ---------------------------------------------------------------------------
# skew
# ---------------------------------------------------------------------------


class TestRollingSkew:
    def test_symmetric_data(self):
        """Symmetric window should have skew near 0."""
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _roll(df, "skew", min_periods=1)
        # Window [1,2,3] is symmetric -> skew=0
        assert result["val_skew"].iloc[2] == pytest.approx(0.0, abs=1e-6)

    def test_skewed_data(self):
        df = _make_df([1.0, 1.0, 1.0, 100.0])
        result = _roll(df, "skew", window=4, min_periods=3)
        # The window [1,1,1,100] should have positive skew
        assert result["val_skew"].iloc[3] > 0


# ---------------------------------------------------------------------------
# kurt
# ---------------------------------------------------------------------------


class TestRollingKurt:
    def test_returns_numeric(self):
        df = _make_df([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = _roll(df, "kurt", window=5, min_periods=4)
        # Just verify it produces numeric values without error
        assert "val_kurt" in result.columns
        non_null = result["val_kurt"].dropna()
        assert len(non_null) > 0


# ---------------------------------------------------------------------------
# slope
# ---------------------------------------------------------------------------


class TestRollingSlope:
    def test_linear_trend(self):
        """Perfectly linear data should have slope = step size."""
        df = _make_df([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _roll(df, "slope", min_periods=1)
        # Window [10,20,30]: x=[0,1,2], y=[10,20,30] -> slope=10
        assert result["val_slope"].iloc[2] == pytest.approx(10.0)
        assert result["val_slope"].iloc[4] == pytest.approx(10.0)

    def test_flat_data(self):
        df = _make_df([5.0, 5.0, 5.0, 5.0])
        result = _roll(df, "slope", min_periods=1)
        # Flat data -> slope=0
        assert result["val_slope"].iloc[2] == pytest.approx(0.0)
        assert result["val_slope"].iloc[3] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ema
# ---------------------------------------------------------------------------


class TestRollingEma:
    def test_basic(self):
        df = _make_df([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _roll(df, "ema", min_periods=1)
        assert "val_ema" in result.columns
        # EMA should be between min and max of the window
        col = result["val_ema"].dropna()
        assert len(col) > 0
        # EMA of [10,20,30] with alpha=0.3 should be weighted toward recent
        # The ema of the last window element should be > simple average
        # since alpha=0.3 weights recent values more

    def test_single_value(self):
        df = _make_df([42.0])
        result = _roll(df, "ema", window=1, min_periods=1)
        assert result["val_ema"].iloc[0] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# range
# ---------------------------------------------------------------------------


class TestRollingRange:
    def test_basic(self):
        df = _make_df([10.0, 50.0, 30.0, 20.0, 40.0])
        result = _roll(df, "range", min_periods=1)
        col = result["val_range"]
        # Window [10,50,30]: range = 50-10 = 40
        assert col.iloc[2] == pytest.approx(40.0)
        # Window [50,30,20]: range = 50-20 = 30
        assert col.iloc[3] == pytest.approx(30.0)
        # Window [30,20,40]: range = 40-20 = 20
        assert col.iloc[4] == pytest.approx(20.0)

    def test_constant(self):
        df = _make_df([7.0, 7.0, 7.0])
        result = _roll(df, "range", min_periods=1)
        assert result["val_range"].iloc[2] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cv (coefficient of variation)
# ---------------------------------------------------------------------------


class TestRollingCv:
    def test_basic(self):
        df = _make_df([10.0, 20.0, 30.0])
        result = _roll(df, "cv", min_periods=1)
        col = result["val_cv"]
        # Window [10,20,30]: std=10, mean=20 -> cv=0.5
        assert col.iloc[2] == pytest.approx(0.5, abs=0.01)

    def test_near_zero_mean(self):
        """CV with near-zero mean should not blow up thanks to epsilon."""
        df = _make_df([0.001, -0.001, 0.001])
        result = _roll(df, "cv", min_periods=1)
        # Should not raise or be inf
        assert np.isfinite(result["val_cv"].iloc[2])


# ---------------------------------------------------------------------------
# pct_change
# ---------------------------------------------------------------------------


class TestRollingPctChange:
    def test_basic(self):
        df = _make_df([100.0, 110.0, 120.0, 150.0])
        result = _roll(df, "pct_change", min_periods=1)
        col = result["val_pct_change"]
        # Window [100,110,120]: (120-100)/|100| = 0.2
        assert col.iloc[2] == pytest.approx(0.2)
        # Window [110,120,150]: (150-110)/|110| = 40/110 ~ 0.3636
        assert col.iloc[3] == pytest.approx(40.0 / 110.0, abs=0.001)

    def test_zero_start(self):
        """Starting from zero should not blow up thanks to epsilon."""
        df = _make_df([0.0, 10.0, 20.0])
        result = _roll(df, "pct_change", min_periods=1)
        # (20-0)/(|0|+1e-8) is very large but finite
        assert np.isfinite(result["val_pct_change"].iloc[2])


# ---------------------------------------------------------------------------
# first
# ---------------------------------------------------------------------------


class TestRollingFirst:
    def test_basic(self):
        df = _make_df([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _roll(df, "first", min_periods=1)
        col = result["val_first"]
        # Window [10]: first=10
        assert col.iloc[0] == pytest.approx(10.0)
        # Window [10,20]: first=10
        assert col.iloc[1] == pytest.approx(10.0)
        # Window [10,20,30]: first=10
        assert col.iloc[2] == pytest.approx(10.0)
        # Window [20,30,40]: first=20
        assert col.iloc[3] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# last
# ---------------------------------------------------------------------------


class TestRollingLast:
    def test_basic(self):
        df = _make_df([10.0, 20.0, 30.0, 40.0, 50.0])
        result = _roll(df, "last", min_periods=1)
        col = result["val_last"]
        # last should always be the most recent value in the window
        assert col.iloc[0] == pytest.approx(10.0)
        assert col.iloc[1] == pytest.approx(20.0)
        assert col.iloc[2] == pytest.approx(30.0)
        assert col.iloc[3] == pytest.approx(40.0)
        assert col.iloc[4] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Multi-group test
# ---------------------------------------------------------------------------


class TestRollingMultiGroup:
    def test_groups_are_independent(self):
        """Rolling aggs should not bleed across groups."""
        df = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B"],
            "order": [1, 2, 3, 1, 2, 3],
            "val": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
        })
        result = _roll(df, "median", min_periods=1)
        a = result[result["group"] == "A"].sort_values("order")
        b = result[result["group"] == "B"].sort_values("order")
        # Group A window [10,20,30] median=20
        assert a["val_median"].iloc[2] == pytest.approx(20.0)
        # Group B window [100,200,300] median=200
        assert b["val_median"].iloc[2] == pytest.approx(200.0)
