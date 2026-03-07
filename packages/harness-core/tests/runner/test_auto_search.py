"""Tests for automated feature search."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harnessml.core.runner.auto_search import (
    SearchResult,
    auto_search,
    format_auto_search_report,
    _safe_corr,
    _search_interactions,
    _search_lags,
    _search_rolling,
)


@pytest.fixture()
def sample_df():
    """Create a synthetic DataFrame for testing."""
    rng = np.random.default_rng(42)
    n = 200
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    # target correlated with x
    target = (x > 0).astype(float) + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "feat_a": x,
        "feat_b": y,
        "feat_c": x + y * 0.5,
        "result": target,
    })


class TestSafeCorr:
    """Test the _safe_corr helper."""

    def test_perfect_correlation(self):
        s = pd.Series(np.arange(20, dtype=float))
        t = np.arange(20, dtype=float)
        assert _safe_corr(s, t) == pytest.approx(1.0, abs=0.01)

    def test_no_correlation(self):
        rng = np.random.default_rng(123)
        n = 500
        s = pd.Series(rng.standard_normal(n))
        t = rng.standard_normal(n)
        # Should be close to 0 for uncorrelated data
        assert abs(_safe_corr(s, t)) < 0.15

    def test_with_nans(self):
        s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        t = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        corr = _safe_corr(s, t)
        assert isinstance(corr, float)
        assert corr > 0.9

    def test_too_few_samples(self):
        s = pd.Series([1.0, 2.0])
        t = np.array([1.0, 2.0])
        assert _safe_corr(s, t) == 0.0

    def test_constant_series(self):
        s = pd.Series([5.0] * 20)
        t = np.arange(20, dtype=float)
        # Correlation of a constant with anything should return 0
        assert _safe_corr(s, t) == 0.0


class TestSearchInteractions:
    """Test interaction (pairwise arithmetic) search."""

    def test_produces_results(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_interactions(
            sample_df, target, ["feat_a", "feat_b"]
        )
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_all_ops_tested(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_interactions(
            sample_df, target, ["feat_a", "feat_b"]
        )
        op_names = {r.name.split("_")[0] for r in results}
        assert "add" in op_names
        assert "sub" in op_names
        assert "mul" in op_names
        assert "div" in op_names
        assert "abs" in op_names  # abs_diff starts with abs

    def test_search_type_tagged(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_interactions(
            sample_df, target, ["feat_a", "feat_b"]
        )
        assert all(r.search_type == "interactions" for r in results)

    def test_formula_populated(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_interactions(
            sample_df, target, ["feat_a", "feat_b"]
        )
        for r in results:
            assert r.formula_or_spec
            assert "feat_a" in r.formula_or_spec or "feat_b" in r.formula_or_spec

    def test_pair_count(self, sample_df):
        """With 3 features, should have C(3,2)=3 pairs x 5 ops = 15 results."""
        target = sample_df["result"].values.astype(float)
        results = _search_interactions(
            sample_df, target, ["feat_a", "feat_b", "feat_c"]
        )
        assert len(results) == 15

    def test_skips_missing_columns(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_interactions(
            sample_df, target, ["feat_a", "nonexistent"]
        )
        # No valid pair can be formed with a missing column
        assert len(results) == 0


class TestSearchLags:
    """Test lag-based search."""

    def test_produces_results(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_lags(sample_df, target, ["feat_a"])
        assert len(results) > 0

    def test_lag_windows(self, sample_df):
        """Should test lags 1, 3, 5, 10."""
        target = sample_df["result"].values.astype(float)
        results = _search_lags(sample_df, target, ["feat_a"])
        lag_values = [int(r.name.replace("lag", "").split("_")[0]) for r in results]
        assert set(lag_values) == {1, 3, 5, 10}

    def test_search_type_tagged(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_lags(sample_df, target, ["feat_a"])
        assert all(r.search_type == "lags" for r in results)

    def test_formula_contains_shift(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_lags(sample_df, target, ["feat_a"])
        for r in results:
            assert "shift" in r.formula_or_spec

    def test_multiple_features(self, sample_df):
        """With 2 features and 4 lag windows, expect 8 results."""
        target = sample_df["result"].values.astype(float)
        results = _search_lags(sample_df, target, ["feat_a", "feat_b"])
        assert len(results) == 8

    def test_skips_missing_columns(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_lags(sample_df, target, ["nonexistent"])
        assert len(results) == 0


class TestSearchRolling:
    """Test rolling mean search."""

    def test_produces_results(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_rolling(sample_df, target, ["feat_a"])
        assert len(results) > 0

    def test_rolling_windows(self, sample_df):
        """Should test windows 3, 5, 10, 20."""
        target = sample_df["result"].values.astype(float)
        results = _search_rolling(sample_df, target, ["feat_a"])
        window_values = [
            int(r.name.replace("rolling", "").split("_")[0]) for r in results
        ]
        assert set(window_values) == {3, 5, 10, 20}

    def test_search_type_tagged(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_rolling(sample_df, target, ["feat_a"])
        assert all(r.search_type == "rolling" for r in results)

    def test_formula_contains_rolling_mean(self, sample_df):
        target = sample_df["result"].values.astype(float)
        results = _search_rolling(sample_df, target, ["feat_a"])
        for r in results:
            assert "rolling_mean" in r.formula_or_spec

    def test_multiple_features(self, sample_df):
        """With 2 features and 4 windows, expect 8 results."""
        target = sample_df["result"].values.astype(float)
        results = _search_rolling(sample_df, target, ["feat_a", "feat_b"])
        assert len(results) == 8


class TestAutoSearch:
    """Test the main auto_search entry point."""

    def test_default_runs_all_types(self, sample_df):
        results = auto_search(sample_df, "result", ["feat_a", "feat_b"])
        search_types_found = {r.search_type for r in results}
        assert "interactions" in search_types_found
        assert "lags" in search_types_found
        assert "rolling" in search_types_found

    def test_filter_by_search_types(self, sample_df):
        results = auto_search(
            sample_df, "result", ["feat_a", "feat_b"],
            search_types=["lags"],
        )
        assert all(r.search_type == "lags" for r in results)

    def test_top_n_limits_results(self, sample_df):
        results = auto_search(
            sample_df, "result", ["feat_a", "feat_b", "feat_c"],
            top_n=5,
        )
        assert len(results) <= 5

    def test_sorted_by_abs_score(self, sample_df):
        results = auto_search(
            sample_df, "result", ["feat_a", "feat_b"],
            top_n=50,
        )
        scores = [abs(r.score) for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_search_type_raises(self, sample_df):
        with pytest.raises(ValueError, match="Invalid search types"):
            auto_search(
                sample_df, "result", ["feat_a"],
                search_types=["bogus"],
            )

    def test_missing_target_raises(self, sample_df):
        with pytest.raises(ValueError, match="Target column"):
            auto_search(sample_df, "nonexistent_target", ["feat_a"])

    def test_empty_feature_list(self, sample_df):
        results = auto_search(sample_df, "result", [])
        assert results == []

    def test_interactions_only(self, sample_df):
        results = auto_search(
            sample_df, "result", ["feat_a", "feat_b"],
            search_types=["interactions"],
        )
        assert len(results) > 0
        assert all(r.search_type == "interactions" for r in results)

    def test_rolling_only(self, sample_df):
        results = auto_search(
            sample_df, "result", ["feat_a"],
            search_types=["rolling"],
        )
        assert len(results) > 0
        assert all(r.search_type == "rolling" for r in results)


class TestFormatAutoSearchReport:
    """Test markdown report formatting."""

    def test_empty_results(self):
        md = format_auto_search_report([])
        assert "No auto-search results found" in md

    def test_produces_markdown_table(self, sample_df):
        results = auto_search(sample_df, "result", ["feat_a", "feat_b"], top_n=5)
        md = format_auto_search_report(results)
        assert "## Auto Feature Search Results" in md
        assert "| Rank |" in md
        assert "| 1 |" in md

    def test_contains_formulas(self, sample_df):
        results = auto_search(
            sample_df, "result", ["feat_a", "feat_b"],
            search_types=["interactions"],
            top_n=3,
        )
        md = format_auto_search_report(results)
        # Formulas should appear in backticks
        assert "`" in md

    def test_contains_search_type(self, sample_df):
        results = auto_search(
            sample_df, "result", ["feat_a"],
            search_types=["lags"],
            top_n=3,
        )
        md = format_auto_search_report(results)
        assert "lags" in md
