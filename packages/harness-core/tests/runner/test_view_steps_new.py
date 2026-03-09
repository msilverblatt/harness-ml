"""Tests for the 8 new view steps: lag, ewm, diff, trend, encode, bin, datetime, null_indicator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.schema import (
    BinStep,
    DatetimeStep,
    DiffStep,
    EncodeStep,
    EwmStep,
    LagStep,
    NullIndicatorStep,
    TrendStep,
)
from harnessml.core.runner.views.executor import execute_step

# ---------------------------------------------------------------------------
# Lag step
# ---------------------------------------------------------------------------


class TestLag:
    def test_basic_lag(self):
        df = pd.DataFrame({
            "team": ["A", "A", "A", "B", "B", "B"],
            "day": [1, 2, 3, 1, 2, 3],
            "pts": [10.0, 20.0, 30.0, 5.0, 15.0, 25.0],
        })
        step = LagStep(keys=["team"], order_by="day", columns={"pts_lag1": "pts:1"})
        result = execute_step(df, step)
        assert "pts_lag1" in result.columns
        team_a = result[result["team"] == "A"].sort_values("day")
        assert pd.isna(team_a["pts_lag1"].iloc[0])
        assert team_a["pts_lag1"].iloc[1] == pytest.approx(10.0)
        assert team_a["pts_lag1"].iloc[2] == pytest.approx(20.0)

    def test_lag_two_periods(self):
        df = pd.DataFrame({
            "team": ["A", "A", "A", "A"],
            "day": [1, 2, 3, 4],
            "pts": [10.0, 20.0, 30.0, 40.0],
        })
        step = LagStep(keys=["team"], order_by="day", columns={"pts_lag2": "pts:2"})
        result = execute_step(df, step)
        team_a = result.sort_values("day")
        assert pd.isna(team_a["pts_lag2"].iloc[0])
        assert pd.isna(team_a["pts_lag2"].iloc[1])
        assert team_a["pts_lag2"].iloc[2] == pytest.approx(10.0)
        assert team_a["pts_lag2"].iloc[3] == pytest.approx(20.0)

    def test_lag_negative_is_lead(self):
        df = pd.DataFrame({
            "team": ["A", "A", "A"],
            "day": [1, 2, 3],
            "pts": [10.0, 20.0, 30.0],
        })
        step = LagStep(keys=["team"], order_by="day", columns={"pts_lead1": "pts:-1"})
        result = execute_step(df, step)
        vals = result.sort_values("day")
        assert vals["pts_lead1"].iloc[0] == pytest.approx(20.0)
        assert vals["pts_lead1"].iloc[1] == pytest.approx(30.0)
        assert pd.isna(vals["pts_lead1"].iloc[2])

    def test_lag_groups_independent(self):
        df = pd.DataFrame({
            "team": ["A", "A", "B", "B"],
            "day": [1, 2, 1, 2],
            "pts": [100.0, 200.0, 5.0, 15.0],
        })
        step = LagStep(keys=["team"], order_by="day", columns={"prev_pts": "pts:1"})
        result = execute_step(df, step)
        team_b = result[result["team"] == "B"].sort_values("day")
        # First row of B should be NaN (not 200 from A)
        assert pd.isna(team_b["prev_pts"].iloc[0])
        assert team_b["prev_pts"].iloc[1] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# EWM step
# ---------------------------------------------------------------------------


class TestEwm:
    def test_ewm_mean(self):
        df = pd.DataFrame({
            "team": ["A", "A", "A", "A"],
            "day": [1, 2, 3, 4],
            "pts": [10.0, 20.0, 30.0, 40.0],
        })
        step = EwmStep(
            keys=["team"], order_by="day", span=2.0,
            aggs={"ewm_pts": "pts:mean"},
        )
        result = execute_step(df, step)
        assert "ewm_pts" in result.columns
        vals = result.sort_values("day")
        # First value should equal the first raw value (adjust=False)
        assert vals["ewm_pts"].iloc[0] == pytest.approx(10.0)
        # EWM with span=2 means alpha = 2/(2+1) = 2/3
        # Second: 2/3 * 20 + 1/3 * 10 = 16.667
        assert vals["ewm_pts"].iloc[1] == pytest.approx(2 / 3 * 20 + 1 / 3 * 10, rel=1e-4)

    def test_ewm_groups_independent(self):
        df = pd.DataFrame({
            "team": ["A", "A", "B", "B"],
            "day": [1, 2, 1, 2],
            "pts": [100.0, 200.0, 1.0, 2.0],
        })
        step = EwmStep(
            keys=["team"], order_by="day", span=2.0,
            aggs={"ewm_pts": "pts:mean"},
        )
        result = execute_step(df, step)
        team_b = result[result["team"] == "B"].sort_values("day")
        assert team_b["ewm_pts"].iloc[0] == pytest.approx(1.0)

    def test_ewm_std(self):
        df = pd.DataFrame({
            "team": ["A", "A", "A", "A", "A"],
            "day": [1, 2, 3, 4, 5],
            "pts": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        step = EwmStep(
            keys=["team"], order_by="day", span=3.0,
            aggs={"ewm_std": "pts:std"},
        )
        result = execute_step(df, step)
        assert "ewm_std" in result.columns
        # First value std should be NaN (need at least 2 points)
        vals = result.sort_values("day")
        assert pd.isna(vals["ewm_std"].iloc[0])
        # Later values should be positive
        assert vals["ewm_std"].iloc[-1] > 0


# ---------------------------------------------------------------------------
# Diff step
# ---------------------------------------------------------------------------


class TestDiff:
    def test_basic_diff(self):
        df = pd.DataFrame({
            "team": ["A", "A", "A", "B", "B", "B"],
            "day": [1, 2, 3, 1, 2, 3],
            "pts": [10.0, 30.0, 60.0, 5.0, 10.0, 20.0],
        })
        step = DiffStep(
            keys=["team"], order_by="day",
            columns={"pts_diff": "pts:1"},
        )
        result = execute_step(df, step)
        assert "pts_diff" in result.columns
        team_a = result[result["team"] == "A"].sort_values("day")
        assert pd.isna(team_a["pts_diff"].iloc[0])
        assert team_a["pts_diff"].iloc[1] == pytest.approx(20.0)
        assert team_a["pts_diff"].iloc[2] == pytest.approx(30.0)

    def test_diff_second_order(self):
        df = pd.DataFrame({
            "team": ["A", "A", "A", "A"],
            "day": [1, 2, 3, 4],
            "pts": [10.0, 20.0, 40.0, 80.0],
        })
        step = DiffStep(
            keys=["team"], order_by="day",
            columns={"pts_diff2": "pts:2"},
        )
        result = execute_step(df, step)
        vals = result.sort_values("day")
        assert pd.isna(vals["pts_diff2"].iloc[0])
        assert pd.isna(vals["pts_diff2"].iloc[1])
        assert vals["pts_diff2"].iloc[2] == pytest.approx(30.0)  # 40 - 10
        assert vals["pts_diff2"].iloc[3] == pytest.approx(60.0)  # 80 - 20

    def test_pct_change(self):
        df = pd.DataFrame({
            "team": ["A", "A", "A"],
            "day": [1, 2, 3],
            "pts": [100.0, 200.0, 150.0],
        })
        step = DiffStep(
            keys=["team"], order_by="day",
            columns={"pts_pct": "pts:1"},
            pct=True,
        )
        result = execute_step(df, step)
        vals = result.sort_values("day")
        assert pd.isna(vals["pts_pct"].iloc[0])
        assert vals["pts_pct"].iloc[1] == pytest.approx(1.0)      # (200-100)/100
        assert vals["pts_pct"].iloc[2] == pytest.approx(-0.25)    # (150-200)/200

    def test_diff_groups_independent(self):
        df = pd.DataFrame({
            "team": ["A", "A", "B", "B"],
            "day": [1, 2, 1, 2],
            "pts": [100.0, 200.0, 10.0, 20.0],
        })
        step = DiffStep(
            keys=["team"], order_by="day",
            columns={"pts_diff": "pts:1"},
        )
        result = execute_step(df, step)
        team_b = result[result["team"] == "B"].sort_values("day")
        assert pd.isna(team_b["pts_diff"].iloc[0])
        assert team_b["pts_diff"].iloc[1] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Trend step
# ---------------------------------------------------------------------------


class TestTrend:
    def test_linear_trend(self):
        """Perfect linear trend should give constant slope."""
        df = pd.DataFrame({
            "team": ["A"] * 5,
            "day": [1, 2, 3, 4, 5],
            "pts": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        step = TrendStep(
            keys=["team"], order_by="day", window=3,
            columns={"pts_trend": "pts"},
        )
        result = execute_step(df, step)
        assert "pts_trend" in result.columns
        vals = result.sort_values("day")
        # First two rows: window=3, min_periods=3, so NaN
        assert pd.isna(vals["pts_trend"].iloc[0])
        assert pd.isna(vals["pts_trend"].iloc[1])
        # Row 3 onward: OLS slope over [10,20,30], [20,30,40], [30,40,50] = 10.0
        assert vals["pts_trend"].iloc[2] == pytest.approx(10.0)
        assert vals["pts_trend"].iloc[3] == pytest.approx(10.0)
        assert vals["pts_trend"].iloc[4] == pytest.approx(10.0)

    def test_flat_trend(self):
        """Flat values should give slope = 0."""
        df = pd.DataFrame({
            "team": ["A"] * 4,
            "day": [1, 2, 3, 4],
            "pts": [5.0, 5.0, 5.0, 5.0],
        })
        step = TrendStep(
            keys=["team"], order_by="day", window=3,
            columns={"pts_trend": "pts"},
        )
        result = execute_step(df, step)
        vals = result.sort_values("day")
        assert vals["pts_trend"].iloc[2] == pytest.approx(0.0)
        assert vals["pts_trend"].iloc[3] == pytest.approx(0.0)

    def test_trend_groups_independent(self):
        df = pd.DataFrame({
            "team": ["A", "A", "A", "B", "B", "B"],
            "day": [1, 2, 3, 1, 2, 3],
            "pts": [10.0, 20.0, 30.0, 100.0, 80.0, 60.0],
        })
        step = TrendStep(
            keys=["team"], order_by="day", window=3,
            columns={"pts_trend": "pts"},
        )
        result = execute_step(df, step)
        team_a = result[result["team"] == "A"].sort_values("day")
        team_b = result[result["team"] == "B"].sort_values("day")
        assert team_a["pts_trend"].iloc[2] == pytest.approx(10.0)
        assert team_b["pts_trend"].iloc[2] == pytest.approx(-20.0)


# ---------------------------------------------------------------------------
# Encode step
# ---------------------------------------------------------------------------


class TestEncode:
    def test_frequency_encoding(self):
        df = pd.DataFrame({
            "color": ["red", "red", "red", "blue", "blue", "green"],
        })
        step = EncodeStep(column="color", method="frequency")
        result = execute_step(df, step)
        assert "color_encoded" in result.columns
        # red appears 3/6 = 0.5, blue 2/6 = 0.333, green 1/6 = 0.167
        red_rows = result[result["color"] == "red"]
        assert red_rows["color_encoded"].iloc[0] == pytest.approx(3 / 6)
        green_rows = result[result["color"] == "green"]
        assert green_rows["color_encoded"].iloc[0] == pytest.approx(1 / 6)

    def test_ordinal_encoding(self):
        df = pd.DataFrame({
            "color": ["red", "red", "red", "blue", "blue", "green"],
        })
        step = EncodeStep(column="color", method="ordinal")
        result = execute_step(df, step)
        assert "color_encoded" in result.columns
        # red=most common=1, blue=2, green=3
        red_rows = result[result["color"] == "red"]
        assert red_rows["color_encoded"].iloc[0] == 1
        green_rows = result[result["color"] == "green"]
        assert green_rows["color_encoded"].iloc[0] == 3

    def test_target_loo_encoding(self):
        df = pd.DataFrame({
            "cat": ["A", "A", "A", "B", "B"],
            "target": [1, 0, 1, 0, 0],
        })
        step = EncodeStep(column="cat", method="target_loo")
        result = execute_step(df, step)
        assert "cat_encoded" in result.columns
        # For cat A row 0 (target=1): LOO = (0+1)/(3-1) = 0.5
        cat_a = result[result["cat"] == "A"]
        assert cat_a["cat_encoded"].iloc[0] == pytest.approx(0.5)
        # For cat A row 1 (target=0): LOO = (1+1)/(3-1) = 1.0
        assert cat_a["cat_encoded"].iloc[1] == pytest.approx(1.0)

    def test_target_temporal_encoding(self):
        df = pd.DataFrame({
            "cat": ["A", "A", "A", "A"],
            "target": [1, 0, 1, 0],
        })
        step = EncodeStep(column="cat", method="target_temporal")
        result = execute_step(df, step)
        assert "cat_encoded" in result.columns
        # Row 0: no prior data -> NaN
        assert pd.isna(result["cat_encoded"].iloc[0])
        # Row 1: mean of prior = 1.0
        assert result["cat_encoded"].iloc[1] == pytest.approx(1.0)
        # Row 2: mean of [1, 0] = 0.5
        assert result["cat_encoded"].iloc[2] == pytest.approx(0.5)
        # Row 3: mean of [1, 0, 1] = 0.667
        assert result["cat_encoded"].iloc[3] == pytest.approx(2 / 3)

    def test_custom_output_name(self):
        df = pd.DataFrame({"x": ["a", "b", "a"]})
        step = EncodeStep(column="x", method="frequency", output="x_freq")
        result = execute_step(df, step)
        assert "x_freq" in result.columns
        assert "x_encoded" not in result.columns

    def test_target_loo_requires_target(self):
        df = pd.DataFrame({"cat": ["A", "B"]})
        step = EncodeStep(column="cat", method="target_loo")
        with pytest.raises(ValueError, match="target"):
            execute_step(df, step)


# ---------------------------------------------------------------------------
# Bin step
# ---------------------------------------------------------------------------


class TestBin:
    def test_quantile_bin(self):
        df = pd.DataFrame({"val": list(range(100))})
        step = BinStep(column="val", method="quantile", n_bins=4)
        result = execute_step(df, step)
        assert "val_binned" in result.columns
        # Should have 4 unique bins
        assert result["val_binned"].nunique() == 4

    def test_uniform_bin(self):
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
        step = BinStep(column="val", method="uniform", n_bins=2)
        result = execute_step(df, step)
        assert "val_binned" in result.columns
        # Lower half should be bin 0, upper half bin 1
        assert result["val_binned"].iloc[0] == 0  # val=1.0
        assert result["val_binned"].iloc[-1] == 1  # val=10.0

    def test_custom_bin(self):
        df = pd.DataFrame({"val": [5, 15, 25, 35, 45]})
        step = BinStep(
            column="val", method="custom",
            boundaries=[0, 10, 20, 30, 40, 50],
        )
        result = execute_step(df, step)
        assert "val_binned" in result.columns
        assert list(result["val_binned"]) == [0, 1, 2, 3, 4]

    def test_kmeans_bin(self):
        # Two clearly separated clusters
        df = pd.DataFrame({"val": [1.0, 2.0, 3.0, 100.0, 101.0, 102.0]})
        step = BinStep(column="val", method="kmeans", n_bins=2)
        result = execute_step(df, step)
        assert "val_binned" in result.columns
        # The two clusters should get different labels
        low_labels = set(result["val_binned"].iloc[:3])
        high_labels = set(result["val_binned"].iloc[3:])
        assert len(low_labels) == 1
        assert len(high_labels) == 1
        assert low_labels != high_labels

    def test_custom_output_name(self):
        df = pd.DataFrame({"val": list(range(20))})
        step = BinStep(column="val", method="quantile", n_bins=4, output="val_q")
        result = execute_step(df, step)
        assert "val_q" in result.columns
        assert "val_binned" not in result.columns

    def test_custom_requires_boundaries(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        step = BinStep(column="val", method="custom")
        with pytest.raises(ValueError, match="boundaries"):
            execute_step(df, step)


# ---------------------------------------------------------------------------
# Datetime step
# ---------------------------------------------------------------------------


class TestDatetime:
    def test_extract_parts(self):
        df = pd.DataFrame({
            "ts": ["2024-03-15 14:30:00", "2024-06-20 08:00:00", "2024-12-25 23:59:00"],
        })
        step = DatetimeStep(
            column="ts",
            extract=["year", "month", "day", "dayofweek", "hour", "quarter"],
        )
        result = execute_step(df, step)
        assert result["ts_year"].tolist() == [2024, 2024, 2024]
        assert result["ts_month"].tolist() == [3, 6, 12]
        assert result["ts_day"].tolist() == [15, 20, 25]
        assert result["ts_hour"].tolist() == [14, 8, 23]
        assert result["ts_quarter"].tolist() == [1, 2, 4]

    def test_extract_weekofyear(self):
        df = pd.DataFrame({"ts": ["2024-01-01", "2024-06-15"]})
        step = DatetimeStep(column="ts", extract=["weekofyear"])
        result = execute_step(df, step)
        assert "ts_weekofyear" in result.columns
        assert result["ts_weekofyear"].iloc[0] == 1
        assert result["ts_weekofyear"].iloc[1] == 24

    def test_cyclical_encoding(self):
        df = pd.DataFrame({
            "ts": ["2024-01-15", "2024-07-15"],
        })
        step = DatetimeStep(column="ts", cyclical=["month"])
        result = execute_step(df, step)
        assert "ts_month_sin" in result.columns
        assert "ts_month_cos" in result.columns
        # January: sin(2*pi*1/12) ~ 0.5, cos(2*pi*1/12) ~ 0.866
        assert result["ts_month_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 1 / 12))
        assert result["ts_month_cos"].iloc[0] == pytest.approx(np.cos(2 * np.pi * 1 / 12))
        # July: sin(2*pi*7/12)
        assert result["ts_month_sin"].iloc[1] == pytest.approx(np.sin(2 * np.pi * 7 / 12))

    def test_extract_and_cyclical_together(self):
        df = pd.DataFrame({"ts": ["2024-03-15 10:00:00"]})
        step = DatetimeStep(column="ts", extract=["month", "hour"], cyclical=["hour"])
        result = execute_step(df, step)
        assert "ts_month" in result.columns
        assert "ts_hour" in result.columns
        assert "ts_hour_sin" in result.columns
        assert "ts_hour_cos" in result.columns


# ---------------------------------------------------------------------------
# Null indicator step
# ---------------------------------------------------------------------------


class TestNullIndicator:
    def test_basic_null_indicator(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [np.nan, 2.0, np.nan, 4.0],
        })
        step = NullIndicatorStep(columns=["a", "b"])
        result = execute_step(df, step)
        assert "missing_a" in result.columns
        assert "missing_b" in result.columns
        assert list(result["missing_a"]) == [0, 1, 0, 1]
        assert list(result["missing_b"]) == [1, 0, 1, 0]

    def test_custom_prefix(self):
        df = pd.DataFrame({"x": [1.0, np.nan]})
        step = NullIndicatorStep(columns=["x"], prefix="is_null_")
        result = execute_step(df, step)
        assert "is_null_x" in result.columns
        assert "missing_x" not in result.columns
        assert list(result["is_null_x"]) == [0, 1]

    def test_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        step = NullIndicatorStep(columns=["a", "b"])
        result = execute_step(df, step)
        assert list(result["missing_a"]) == [0, 0, 0]
        assert list(result["missing_b"]) == [0, 0, 0]

    def test_preserves_original_columns(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        step = NullIndicatorStep(columns=["a"])
        result = execute_step(df, step)
        # Original column unchanged
        assert result["a"].iloc[0] == 1.0
        assert pd.isna(result["a"].iloc[1])
        assert result["a"].iloc[2] == 3.0
