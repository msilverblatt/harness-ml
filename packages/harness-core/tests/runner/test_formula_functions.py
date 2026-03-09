"""Tests for the 15 new formula functions added to _SAFE_FUNCTIONS."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.features.engine import _resolve_formula

# ---------------------------------------------------------------------------
# Power / distribution functions
# ---------------------------------------------------------------------------


class TestReciprocal:
    def test_basic(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 4.0, 0.0]})
        result = _resolve_formula("reciprocal(x)", df)
        expected = pd.Series([1.0, 0.5, 0.25, 0.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_negative(self):
        df = pd.DataFrame({"x": [-2.0, -0.5]})
        result = _resolve_formula("reciprocal(x)", df)
        expected = pd.Series([-0.5, -2.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestExp:
    def test_basic(self):
        df = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
        result = _resolve_formula("exp(x)", df)
        expected = pd.Series(np.exp([0.0, 1.0, 2.0]))
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestExpm1:
    def test_basic(self):
        df = pd.DataFrame({"x": [0.0, 1.0, 0.001]})
        result = _resolve_formula("expm1(x)", df)
        expected = pd.Series(np.expm1([0.0, 1.0, 0.001]))
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_zero(self):
        df = pd.DataFrame({"x": [0.0]})
        result = _resolve_formula("expm1(x)", df)
        assert result.iloc[0] == pytest.approx(0.0)


class TestPower:
    def test_basic(self):
        df = pd.DataFrame({"x": [2.0, 3.0, 4.0], "y": [3.0, 2.0, 0.5]})
        result = _resolve_formula("power(x, y)", df)
        expected = pd.Series(np.power([2.0, 3.0, 4.0], [3.0, 2.0, 0.5]))
        pd.testing.assert_series_equal(result, expected, check_names=False)


# ---------------------------------------------------------------------------
# Cyclical functions
# ---------------------------------------------------------------------------


class TestSinCycle:
    def test_basic(self):
        df = pd.DataFrame({"x": [0.0, 3.0, 6.0, 12.0]})
        result = _resolve_formula("sin_cycle(x, 12)", df)
        expected = pd.Series(np.sin(2 * np.pi * np.array([0.0, 3.0, 6.0, 12.0]) / 12))
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_full_period_returns_zero(self):
        df = pd.DataFrame({"x": [0.0, 10.0]})
        result = _resolve_formula("sin_cycle(x, 10)", df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-10)
        assert result.iloc[1] == pytest.approx(0.0, abs=1e-10)


class TestCosCycle:
    def test_basic(self):
        df = pd.DataFrame({"x": [0.0, 3.0, 6.0, 12.0]})
        result = _resolve_formula("cos_cycle(x, 12)", df)
        expected = pd.Series(np.cos(2 * np.pi * np.array([0.0, 3.0, 6.0, 12.0]) / 12))
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_start_and_end_are_one(self):
        df = pd.DataFrame({"x": [0.0, 10.0]})
        result = _resolve_formula("cos_cycle(x, 10)", df)
        assert result.iloc[0] == pytest.approx(1.0, abs=1e-10)
        assert result.iloc[1] == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------


class TestZscore:
    def test_basic(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
        result = _resolve_formula("zscore(x)", df)
        # mean=20, std=10
        assert result.iloc[0] == pytest.approx(-1.0, abs=0.01)
        assert result.iloc[1] == pytest.approx(0.0, abs=0.01)
        assert result.iloc[2] == pytest.approx(1.0, abs=0.01)

    def test_constant_series(self):
        """All identical values: std ~0, zscore should be ~0 thanks to epsilon."""
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        result = _resolve_formula("zscore(x)", df)
        assert all(abs(v) < 1e-4 for v in result)


class TestMinmax:
    def test_basic(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0]})
        result = _resolve_formula("minmax(x)", df)
        assert result.iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert result.iloc[2] == pytest.approx(1.0, abs=1e-6)
        assert result.iloc[1] == pytest.approx(0.5, abs=1e-6)

    def test_constant_series(self):
        """All identical values: range ~0, should be ~0 thanks to epsilon."""
        df = pd.DataFrame({"x": [7.0, 7.0, 7.0]})
        result = _resolve_formula("minmax(x)", df)
        assert all(abs(v) < 1e-4 for v in result)


class TestRankPct:
    def test_basic(self):
        df = pd.DataFrame({"x": [30.0, 10.0, 20.0]})
        result = _resolve_formula("rank_pct(x)", df)
        # rank_pct: 10->1/3, 20->2/3, 30->3/3
        assert result.iloc[0] == pytest.approx(1.0)       # 30 is rank 3/3
        assert result.iloc[1] == pytest.approx(1 / 3)     # 10 is rank 1/3
        assert result.iloc[2] == pytest.approx(2 / 3)     # 20 is rank 2/3


class TestWinsorize:
    def test_basic(self):
        df = pd.DataFrame({"x": list(range(100))})
        df["x"] = df["x"].astype(float)
        result = _resolve_formula("winsorize(x)", df)
        # Default 5%/95%: values below 5th pctile clipped up, above 95th clipped down
        lower = df["x"].quantile(0.05)
        upper = df["x"].quantile(0.95)
        assert result.min() >= lower - 1e-8
        assert result.max() <= upper + 1e-8

    def test_custom_bounds(self):
        df = pd.DataFrame({"x": list(range(100))})
        df["x"] = df["x"].astype(float)
        result = _resolve_formula("winsorize(x, 0.1, 0.9)", df)
        lower = df["x"].quantile(0.1)
        upper = df["x"].quantile(0.9)
        assert result.min() >= lower - 1e-8
        assert result.max() <= upper + 1e-8


# ---------------------------------------------------------------------------
# Comparison functions
# ---------------------------------------------------------------------------


class TestMaximum:
    def test_basic(self):
        df = pd.DataFrame({"x": [1.0, 5.0, 3.0], "y": [4.0, 2.0, 3.0]})
        result = _resolve_formula("maximum(x, y)", df)
        expected = pd.Series([4.0, 5.0, 3.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestMinimum:
    def test_basic(self):
        df = pd.DataFrame({"x": [1.0, 5.0, 3.0], "y": [4.0, 2.0, 3.0]})
        result = _resolve_formula("minimum(x, y)", df)
        expected = pd.Series([1.0, 2.0, 3.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestWhere:
    def test_basic(self):
        df = pd.DataFrame({"x": [1.0, -2.0, 3.0, -4.0]})
        result = _resolve_formula("where(x > 0, x, 0)", df)
        expected = pd.Series([1.0, 0.0, 3.0, 0.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestIsnull:
    def test_basic(self):
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan]})
        result = _resolve_formula("isnull(x)", df)
        expected = pd.Series([0.0, 1.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_no_nulls(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = _resolve_formula("isnull(x)", df)
        assert all(v == 0.0 for v in result)


# ---------------------------------------------------------------------------
# Composition functions
# ---------------------------------------------------------------------------


class TestSafeDiv:
    def test_basic(self):
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0], "y": [2.0, 0.0, 5.0]})
        result = _resolve_formula("safe_div(x, y)", df)
        expected = pd.Series([5.0, 0.0, 6.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_all_zeros_denom(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [0.0, 0.0]})
        result = _resolve_formula("safe_div(x, y)", df)
        expected = pd.Series([0.0, 0.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestPctOfTotal:
    def test_basic(self):
        df = pd.DataFrame({"x": [25.0, 50.0, 75.0], "y": [100.0, 200.0, 0.0]})
        result = _resolve_formula("pct_of_total(x, y)", df)
        expected = pd.Series([25.0, 25.0, 0.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_zero_denom(self):
        df = pd.DataFrame({"x": [10.0], "y": [0.0]})
        result = _resolve_formula("pct_of_total(x, y)", df)
        assert result.iloc[0] == 0.0
