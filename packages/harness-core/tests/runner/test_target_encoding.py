"""Tests for target encoding in Preprocessor."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.training.preprocessing import Preprocessor


class TestTargetEncoding:
    """Target encoding with Bayesian smoothing."""

    def test_basic_target_encoding(self):
        df = pd.DataFrame({"cat": ["a", "a", "b", "b", "b"]})
        y = pd.Series([1, 1, 0, 0, 0])

        pp = Preprocessor(categorical_strategy="target", smoothing=0)
        result = pp.fit_transform(df, y=y)

        # "a" has mean target 1.0, "b" has mean target 0.0
        assert result["cat"].iloc[0] == pytest.approx(1.0)
        assert result["cat"].iloc[2] == pytest.approx(0.0)

    def test_smoothing_pulls_toward_global_mean(self):
        df = pd.DataFrame({"cat": ["a", "b", "b", "b", "b", "b"]})
        y = pd.Series([1, 0, 0, 0, 0, 0])

        pp_no_smooth = Preprocessor(categorical_strategy="target", smoothing=0)
        result_no = pp_no_smooth.fit_transform(df, y=y)

        pp_smooth = Preprocessor(categorical_strategy="target", smoothing=10)
        result_smooth = pp_smooth.fit_transform(df, y=y)

        # With smoothing, "a" (only 1 sample) should be pulled toward global mean
        # more than "b" (5 samples)
        global_mean = y.mean()
        a_no = result_no["cat"].iloc[0]
        a_smooth = result_smooth["cat"].iloc[0]
        assert abs(a_smooth - global_mean) < abs(a_no - global_mean)

    def test_unseen_category_gets_global_mean(self):
        train_df = pd.DataFrame({"cat": ["a", "a", "b", "b"]})
        train_y = pd.Series([1, 1, 0, 0])

        pp = Preprocessor(categorical_strategy="target", smoothing=0)
        pp.fit_transform(train_df, y=train_y)

        test_df = pd.DataFrame({"cat": ["a", "c"]})  # "c" is unseen
        result = pp.transform(test_df)

        global_mean = train_y.mean()
        assert result["cat"].iloc[1] == pytest.approx(global_mean)

    def test_requires_y(self):
        df = pd.DataFrame({"cat": ["a", "b"]})
        pp = Preprocessor(categorical_strategy="target")
        with pytest.raises(ValueError, match="requires y"):
            pp.fit_transform(df)

    def test_no_leakage_from_test(self):
        """Target encoding only uses training data, not test."""
        train_df = pd.DataFrame({"cat": ["a", "a", "b", "b"]})
        train_y = pd.Series([0.0, 0.0, 1.0, 1.0])

        pp = Preprocessor(categorical_strategy="target", smoothing=0)
        pp.fit_transform(train_df, y=train_y)

        # Test data with different target distribution shouldn't affect encoding
        test_df = pd.DataFrame({"cat": ["a", "b"]})
        result = pp.transform(test_df)

        # "a" should still map to 0.0 (from training), "b" to 1.0
        assert result["cat"].iloc[0] == pytest.approx(0.0)
        assert result["cat"].iloc[1] == pytest.approx(1.0)

    def test_works_with_regression_target(self):
        rng = np.random.default_rng(42)
        cats = rng.choice(["low", "mid", "high"], size=300)
        target_map = {"low": 10.0, "mid": 50.0, "high": 90.0}
        y = pd.Series([target_map[c] + rng.normal(0, 1) for c in cats])
        df = pd.DataFrame({"quality": cats})

        pp = Preprocessor(categorical_strategy="target", smoothing=5)
        result = pp.fit_transform(df, y=y)

        # Encoded values should roughly follow low < mid < high
        low_vals = result[df["quality"] == "low"]["quality"].mean()
        high_vals = result[df["quality"] == "high"]["quality"].mean()
        assert low_vals < high_vals
