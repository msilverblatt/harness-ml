"""Tests for data profiling utilities."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.runner.data_profiler import DataProfile, profile_dataset


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def sample_parquet(tmp_path) -> Path:
    """Create a sample matchup features parquet for testing."""
    n = 200
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "Season": np.repeat([2020, 2021, 2022, 2023], 50),
        "TeamAWon": rng.integers(0, 2, size=n).astype(float),
        "TeamAMargin": rng.normal(0, 10, size=n),
        "diff_seed_num": rng.normal(0, 3, size=n),
        "diff_sr_srs": rng.normal(0, 5, size=n),
        "diff_adj_net": rng.normal(0, 8, size=n),
        "diff_with_nulls": np.where(
            rng.random(n) < 0.6, np.nan, rng.normal(0, 1, size=n)
        ),
        "seed_num_a": rng.integers(1, 17, size=n).astype(float),
        "win_pct_a": rng.uniform(0.3, 0.95, size=n),
    })

    path = tmp_path / "matchup_features.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def sample_profile(sample_parquet) -> DataProfile:
    """Profile the sample parquet."""
    return profile_dataset(sample_parquet)


# -----------------------------------------------------------------------
# Tests: basic profiling
# -----------------------------------------------------------------------

class TestProfileBasics:
    """profile_dataset returns correct shape and metadata."""

    def test_row_count(self, sample_profile):
        assert sample_profile.n_rows == 200

    def test_col_count(self, sample_profile):
        assert sample_profile.n_cols == 9

    def test_seasons_detected(self, sample_profile):
        assert sample_profile.seasons == [2020, 2021, 2022, 2023]

    def test_season_counts(self, sample_profile):
        assert sample_profile.season_counts[2020] == 50
        assert sample_profile.season_counts[2023] == 50

    def test_label_detected(self, sample_profile):
        assert sample_profile.label_column == "TeamAWon"

    def test_label_distribution(self, sample_profile):
        dist = sample_profile.label_distribution
        assert "mean" in dist
        assert 0.0 <= dist["mean"] <= 1.0

    def test_margin_detected(self, sample_profile):
        assert sample_profile.margin_column == "TeamAMargin"

    def test_margin_stats(self, sample_profile):
        ms = sample_profile.margin_stats
        assert "mean" in ms
        assert "std" in ms
        assert ms["std"] > 0


# -----------------------------------------------------------------------
# Tests: column classification
# -----------------------------------------------------------------------

class TestColumnClassification:
    """Columns are classified as diff_ vs non-diff."""

    def test_diff_columns_identified(self, sample_profile):
        diff_names = [c.name for c in sample_profile.diff_columns]
        assert "diff_seed_num" in diff_names
        assert "diff_sr_srs" in diff_names
        assert "diff_adj_net" in diff_names

    def test_non_diff_columns_identified(self, sample_profile):
        non_diff_names = [c.name for c in sample_profile.non_diff_columns]
        assert "seed_num_a" in non_diff_names
        assert "win_pct_a" in non_diff_names

    def test_id_columns_excluded(self, sample_profile):
        all_names = (
            [c.name for c in sample_profile.diff_columns]
            + [c.name for c in sample_profile.non_diff_columns]
        )
        assert "Season" not in all_names
        assert "TeamAWon" not in all_names
        assert "TeamAMargin" not in all_names


# -----------------------------------------------------------------------
# Tests: null detection
# -----------------------------------------------------------------------

class TestNullDetection:
    """High-null columns are flagged."""

    def test_high_null_detected(self, sample_profile):
        high_null_names = [c.name for c in sample_profile.high_null_columns]
        assert "diff_with_nulls" in high_null_names

    def test_low_null_not_flagged(self, sample_profile):
        high_null_names = [c.name for c in sample_profile.high_null_columns]
        assert "diff_seed_num" not in high_null_names

    def test_null_pct_correct(self, sample_profile):
        null_col = next(
            c for c in sample_profile.high_null_columns
            if c.name == "diff_with_nulls"
        )
        # We set 60% null in fixture
        assert 50 < null_col.null_pct < 70

    def test_custom_threshold(self, sample_parquet):
        profile = profile_dataset(sample_parquet, high_null_threshold=80.0)
        high_null_names = [c.name for c in profile.high_null_columns]
        assert "diff_with_nulls" not in high_null_names


# -----------------------------------------------------------------------
# Tests: column stats
# -----------------------------------------------------------------------

class TestColumnStats:
    """Numeric columns have mean/std/min/max."""

    def test_numeric_stats_present(self, sample_profile):
        seed_col = next(
            c for c in sample_profile.diff_columns
            if c.name == "diff_seed_num"
        )
        assert seed_col.mean is not None
        assert seed_col.std is not None
        assert seed_col.min is not None
        assert seed_col.max is not None

    def test_std_positive(self, sample_profile):
        seed_col = next(
            c for c in sample_profile.diff_columns
            if c.name == "diff_seed_num"
        )
        assert seed_col.std > 0


# -----------------------------------------------------------------------
# Tests: formatting
# -----------------------------------------------------------------------

class TestFormatting:
    """Format methods produce non-empty strings."""

    def test_format_summary(self, sample_profile):
        text = sample_profile.format_summary()
        assert "200 rows" in text
        assert "9 columns" in text

    def test_format_columns(self, sample_profile):
        text = sample_profile.format_columns(category="diff")
        assert "diff_seed_num" in text

    def test_format_null_tiers(self, sample_profile):
        text = sample_profile.format_null_tiers()
        assert "0% null" in text

    def test_format_summary_with_seasons(self, sample_profile):
        text = sample_profile.format_summary()
        assert "2020" in text


# -----------------------------------------------------------------------
# Tests: edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases: empty data, no season column, etc."""

    def test_no_season_column(self, tmp_path):
        df = pd.DataFrame({
            "feature_a": [1.0, 2.0, 3.0],
            "result": [1.0, 0.0, 1.0],
        })
        path = tmp_path / "matchup_features.parquet"
        df.to_parquet(path)
        profile = profile_dataset(path)
        assert profile.seasons == []

    def test_no_label_column(self, tmp_path):
        df = pd.DataFrame({
            "season": [2020, 2021],
            "diff_x": [1.0, -1.0],
        })
        path = tmp_path / "matchup_features.parquet"
        df.to_parquet(path)
        profile = profile_dataset(path)
        assert profile.label_column is None

    def test_easyml_style_columns(self, tmp_path):
        """Test with lowercase column names (easyml convention)."""
        df = pd.DataFrame({
            "season": [2020, 2021, 2022],
            "result": [1.0, 0.0, 1.0],
            "margin": [5.0, -3.0, 12.0],
            "diff_seed_num": [1.0, -2.0, 3.0],
        })
        path = tmp_path / "matchup_features.parquet"
        df.to_parquet(path)
        profile = profile_dataset(path)
        assert profile.label_column == "result"
        assert profile.margin_column == "margin"
