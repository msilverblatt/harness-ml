"""Tests for declarative data ingestion."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.runner.data_ingest import (
    IngestResult,
    ingest_dataset,
    _detect_join_keys,
    _compute_correlation_preview,
    _compute_null_rates,
)


def _make_base_parquet(path: Path, n: int = 200) -> None:
    """Create a base matchup_features.parquet."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "season": rng.choice([2022, 2023, 2024], size=n),
        "game_id": np.arange(n),
        "result": rng.integers(0, 2, size=n),
        "diff_x": rng.standard_normal(n),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_new_csv(path: Path, n: int = 200) -> None:
    """Create a new CSV dataset that can join on season + game_id."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "season": rng.choice([2022, 2023, 2024], size=n),
        "game_id": np.arange(n),
        "coach_win_pct": rng.uniform(0.3, 0.9, size=n),
        "coach_tenure": rng.integers(1, 20, size=n).astype(float),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class TestAutoDetectJoinKeys:
    """Test join key auto-detection."""

    def test_detects_season_game_id(self):
        new = pd.DataFrame({"season": [1], "game_id": [1], "val": [1.0]})
        existing = pd.DataFrame({"season": [1], "game_id": [1], "diff_x": [1.0]})
        keys = _detect_join_keys(new, existing)
        assert keys == ["season", "game_id"]

    def test_detects_season_only(self):
        new = pd.DataFrame({"season": [1], "val": [1.0]})
        existing = pd.DataFrame({"season": [1], "diff_x": [1.0]})
        keys = _detect_join_keys(new, existing)
        assert keys == ["season"]

    def test_returns_none_when_no_match(self):
        new = pd.DataFrame({"foo": [1], "val": [1.0]})
        existing = pd.DataFrame({"bar": [1], "diff_x": [1.0]})
        keys = _detect_join_keys(new, existing)
        assert keys is None


class TestComputeNullRates:
    """Test null rate computation."""

    def test_computes_rates(self):
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [1.0, 2.0, 3.0, 4.0],
        })
        rates = _compute_null_rates(df, ["a", "b"])
        assert rates["a"] == pytest.approx(0.5)
        assert rates["b"] == pytest.approx(0.0)

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"a": [1.0]})
        rates = _compute_null_rates(df, ["a", "missing"])
        assert "a" in rates
        assert "missing" not in rates


class TestComputeCorrelationPreview:
    """Test correlation preview computation."""

    def test_returns_top_correlations(self):
        rng = np.random.default_rng(42)
        n = 200
        target = rng.integers(0, 2, size=n)
        df = pd.DataFrame({
            "result": target,
            "good_feat": target.astype(float) + rng.normal(0, 0.3, n),
            "noise_feat": rng.standard_normal(n),
        })
        preview = _compute_correlation_preview(df, ["good_feat", "noise_feat"])
        assert len(preview) > 0
        # good_feat should have higher abs correlation
        assert abs(preview[0][1]) > abs(preview[-1][1])

    def test_no_target_returns_empty(self):
        df = pd.DataFrame({"feat": [1.0, 2.0]})
        preview = _compute_correlation_preview(df, ["feat"], target_col="result")
        assert preview == []

    def test_limits_to_top_n(self):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "result": rng.integers(0, 2, size=n),
            **{f"f{i}": rng.standard_normal(n) for i in range(10)},
        })
        preview = _compute_correlation_preview(
            df, [f"f{i}" for i in range(10)], top_n=3,
        )
        assert len(preview) <= 3


class TestIngestDataset:
    """Test ingest_dataset end-to-end."""

    def test_csv_ingest_auto_join(self, tmp_path):
        """CSV ingest with auto-detected join keys."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "matchup_features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
        )

        assert "coach_win_pct" in result.columns_added
        assert "coach_tenure" in result.columns_added
        assert result.rows_matched > 0
        assert result.rows_total == 200
        assert result.name == "new_data"

        # Verify parquet was updated
        updated = pd.read_parquet(parquet_path)
        assert "coach_win_pct" in updated.columns
        assert "coach_tenure" in updated.columns

    def test_parquet_ingest_explicit_join(self, tmp_path):
        """Parquet ingest with explicit join keys."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "matchup_features.parquet"
        _make_base_parquet(parquet_path)

        # Create new parquet with different column names
        rng = np.random.default_rng(42)
        new_df = pd.DataFrame({
            "season": rng.choice([2022, 2023, 2024], size=200),
            "game_id": np.arange(200),
            "power_rating": rng.standard_normal(200),
        })
        new_path = tmp_path / "ratings.parquet"
        new_df.to_parquet(new_path, index=False)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(new_path),
            join_on=["season", "game_id"],
        )

        assert "power_rating" in result.columns_added
        assert result.rows_matched > 0

    def test_prefix_avoids_collisions(self, tmp_path):
        """Prefix is applied to new columns."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "matchup_features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            prefix="coach_",
        )

        assert "coach_coach_win_pct" in result.columns_added
        assert "coach_coach_tenure" in result.columns_added

        updated = pd.read_parquet(parquet_path)
        assert "coach_coach_win_pct" in updated.columns

    def test_correlation_preview_populated(self, tmp_path):
        """Correlation preview is populated with numeric columns."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "matchup_features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
        )

        assert len(result.correlation_preview) > 0
        # Each entry is (column_name, correlation)
        for col, corr in result.correlation_preview:
            assert isinstance(col, str)
            assert isinstance(corr, float)

    def test_null_rates_reported(self, tmp_path):
        """Null rates are computed for new columns."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "matchup_features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
        )

        assert "coach_win_pct" in result.null_rates
        assert "coach_tenure" in result.null_rates

    def test_no_new_columns_returns_warning(self, tmp_path):
        """If all columns already exist, returns warning."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "matchup_features.parquet"
        _make_base_parquet(parquet_path)

        # Create CSV with same columns as existing parquet
        rng = np.random.default_rng(42)
        dup_df = pd.DataFrame({
            "season": rng.choice([2022, 2023, 2024], size=200),
            "game_id": np.arange(200),
            "diff_x": rng.standard_normal(200),
        })
        dup_path = tmp_path / "dup.csv"
        dup_df.to_csv(dup_path, index=False)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(dup_path),
        )

        assert len(result.columns_added) == 0
        assert len(result.warnings) > 0

    def test_missing_file_raises(self, tmp_path):
        """Missing data file raises FileNotFoundError."""
        feat_dir = tmp_path / "data" / "features"
        _make_base_parquet(feat_dir / "matchup_features.parquet")

        with pytest.raises(FileNotFoundError):
            ingest_dataset(
                project_dir=tmp_path,
                data_path=str(tmp_path / "nonexistent.csv"),
            )

    def test_no_join_keys_raises(self, tmp_path):
        """When join keys can't be detected, raises ValueError."""
        feat_dir = tmp_path / "data" / "features"
        _make_base_parquet(feat_dir / "matchup_features.parquet")

        # Create CSV with no matching columns
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "foo_id": np.arange(100),
            "bar_val": rng.standard_normal(100),
        })
        path = tmp_path / "no_keys.csv"
        df.to_csv(path, index=False)

        with pytest.raises(ValueError, match="Could not auto-detect"):
            ingest_dataset(
                project_dir=tmp_path,
                data_path=str(path),
            )

    def test_custom_features_dir(self, tmp_path):
        """Can specify a custom features_dir."""
        custom_dir = tmp_path / "custom_features"
        parquet_path = custom_dir / "matchup_features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            features_dir=str(custom_dir),
        )

        assert len(result.columns_added) > 0
        updated = pd.read_parquet(parquet_path)
        assert "coach_win_pct" in updated.columns


class TestIngestResultFormat:
    """Test IngestResult.format_summary()."""

    def test_format_summary_produces_markdown(self):
        result = IngestResult(
            name="coaching_data",
            columns_added=["diff_coach_win_pct", "diff_coach_tenure"],
            rows_matched=180,
            rows_total=200,
            null_rates={"diff_coach_win_pct": 0.05, "diff_coach_tenure": 0.15},
            correlation_preview=[
                ("diff_coach_win_pct", 0.23),
                ("diff_coach_tenure", 0.18),
            ],
            warnings=["Low match rate: 180/200 (90%) rows matched."],
        )
        md = result.format_summary()
        assert "## Ingested: coaching_data" in md
        assert "180 / 200" in md
        assert "Top Correlations" in md
        assert "+0.2300" in md
        assert "Warnings" in md
        assert "High Null" in md
        assert "diff_coach_tenure" in md

    def test_format_summary_no_warnings(self):
        result = IngestResult(
            name="clean_data",
            columns_added=["feat_a"],
            rows_matched=100,
            rows_total=100,
            null_rates={"feat_a": 0.0},
            correlation_preview=[("feat_a", 0.15)],
            warnings=[],
        )
        md = result.format_summary()
        assert "Warnings" not in md
        assert "High Null" not in md
