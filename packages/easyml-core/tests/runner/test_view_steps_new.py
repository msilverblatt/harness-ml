"""Tests for new view steps: lag, ewm, diff, trend."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.core.runner.schema import (
    DiffStep,
    EwmStep,
    LagStep,
    TrendStep,
)
from easyml.core.runner.view_executor import execute_step


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
