"""Tests for declarative data ingestion."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.runner.data_ingest import (
    IngestResult,
    ingest_dataset,
    validate_dataset,
    fill_nulls,
    drop_duplicates,
    rename_columns,
    _detect_join_keys,
    _auto_clean,
    _compute_correlation_preview,
    _compute_null_rates,
)


def _make_base_parquet(path: Path, n: int = 200) -> None:
    """Create a base features.parquet."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "season": rng.choice([2022, 2023, 2024], size=n),
        "game_id": np.arange(n),
        "result": rng.integers(0, 2, size=n),
        "feat_x": rng.standard_normal(n),
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

    def test_detects_overlapping_columns(self):
        new = pd.DataFrame({"season": [1], "game_id": [1], "val": [1.0]})
        existing = pd.DataFrame({"season": [1], "game_id": [1], "feat_x": [1.0]})
        keys = _detect_join_keys(new, existing)
        assert "season" in keys
        assert "game_id" in keys

    def test_uses_configured_key_columns(self):
        new = pd.DataFrame({"customer_id": [1], "date": ["2024-01"], "val": [1.0]})
        existing = pd.DataFrame({"customer_id": [1], "date": ["2024-01"], "feat": [1.0]})
        keys = _detect_join_keys(new, existing, key_columns=["customer_id"])
        assert keys == ["customer_id"]

    def test_configured_keys_not_in_both_falls_back(self):
        new = pd.DataFrame({"season": [1], "val": [1.0]})
        existing = pd.DataFrame({"season": [1], "feat_x": [1.0]})
        keys = _detect_join_keys(new, existing, key_columns=["missing_key"])
        # Falls back to overlapping columns
        assert keys == ["season"]

    def test_returns_none_when_no_match(self):
        new = pd.DataFrame({"foo": [1], "val": [1.0]})
        existing = pd.DataFrame({"bar": [1], "feat_x": [1.0]})
        keys = _detect_join_keys(new, existing)
        assert keys is None

    def test_excludes_target_columns(self):
        new = pd.DataFrame({"id": [1], "result": [1], "val": [1.0]})
        existing = pd.DataFrame({"id": [1], "result": [1], "feat_x": [1.0]})
        keys = _detect_join_keys(new, existing, exclude_cols=["result"])
        assert "result" not in keys
        assert "id" in keys


class TestDetectJoinKeysConfigAware:
    """_detect_join_keys uses exclude_cols as negative filter."""

    def test_excludes_specified_columns(self):
        from easyml.runner.data_ingest import _detect_join_keys

        new_df = pd.DataFrame({"season": [2020], "outcome": [1], "feat": [0.5]})
        existing_df = pd.DataFrame({"season": [2020], "outcome": [0], "diff_x": [1.0]})

        # Without exclude_cols, "outcome" is a join key candidate
        keys = _detect_join_keys(new_df, existing_df)
        assert "outcome" in keys

        # With exclude_cols, "outcome" is filtered out
        keys = _detect_join_keys(new_df, existing_df, exclude_cols=["outcome"])
        assert "outcome" not in keys
        assert "season" in keys

    def test_exclude_cols_case_insensitive(self):
        from easyml.runner.data_ingest import _detect_join_keys

        new_df = pd.DataFrame({"Season": [2020], "Result": [1]})
        existing_df = pd.DataFrame({"Season": [2020], "Result": [0]})

        keys = _detect_join_keys(new_df, existing_df, exclude_cols=["result"])
        assert "Result" not in keys


class TestAutoClean:
    """Test auto-cleaning functionality."""

    def test_coerces_string_numerics(self):
        df = pd.DataFrame({
            "num_as_str": pd.array(["1.5", "2.3", "3.1", "4.0"], dtype="string"),
            "actual_str": pd.array(["a", "b", "c", "d"], dtype="string"),
        })
        cleaned, actions = _auto_clean(df)
        assert pd.api.types.is_numeric_dtype(cleaned["num_as_str"])
        assert any("Coerced" in a and "num_as_str" in a for a in actions)

    def test_fills_numeric_nulls_with_median(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0, 5.0],
        })
        cleaned, actions = _auto_clean(df)
        assert cleaned["x"].isna().sum() == 0
        assert any("median" in a for a in actions)

    def test_fills_categorical_nulls_with_mode(self):
        df = pd.DataFrame({
            "cat": pd.array(["a", "a", "b", pd.NA, "a"], dtype="string"),
        })
        cleaned, actions = _auto_clean(df)
        assert cleaned["cat"].isna().sum() == 0
        assert any("mode" in a for a in actions)

    def test_drops_duplicates(self):
        df = pd.DataFrame({
            "x": [1.0, 1.0, 2.0, 2.0, 3.0],
            "y": [10, 10, 20, 20, 30],
        })
        cleaned, actions = _auto_clean(df)
        assert len(cleaned) == 3
        assert any("duplicate" in a for a in actions)

    def test_no_actions_on_clean_data(self):
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [4.0, 5.0, 6.0],
        })
        cleaned, actions = _auto_clean(df)
        assert len(actions) == 0
        assert len(cleaned) == 3


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
        preview = _compute_correlation_preview(df, ["good_feat", "noise_feat"], target_col="result")
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
            df, [f"f{i}" for i in range(10)], target_col="result", top_n=3,
        )
        assert len(preview) <= 3


class TestIngestDataset:
    """Test ingest_dataset end-to-end."""

    def test_csv_ingest_auto_join(self, tmp_path):
        """CSV ingest with auto-detected join keys."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            auto_clean=False,
        )

        assert "coach_win_pct" in result.columns_added
        assert "coach_tenure" in result.columns_added
        assert result.rows_matched > 0
        assert result.rows_total == 200
        assert result.name == "new_data"
        assert result.is_bootstrap is False

        # Verify parquet was updated
        updated = pd.read_parquet(parquet_path)
        assert "coach_win_pct" in updated.columns
        assert "coach_tenure" in updated.columns

    def test_parquet_ingest_explicit_join(self, tmp_path):
        """Parquet ingest with explicit join keys."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "features.parquet"
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
            auto_clean=False,
        )

        assert "power_rating" in result.columns_added
        assert result.rows_matched > 0

    def test_prefix_avoids_collisions(self, tmp_path):
        """Prefix is applied to new columns."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            prefix="coach_",
            auto_clean=False,
        )

        assert "coach_coach_win_pct" in result.columns_added
        assert "coach_coach_tenure" in result.columns_added

        updated = pd.read_parquet(parquet_path)
        assert "coach_coach_win_pct" in updated.columns

    def test_correlation_preview_populated(self, tmp_path):
        """Correlation preview is populated with numeric columns."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            auto_clean=False,
        )

        assert len(result.correlation_preview) > 0
        for col, corr in result.correlation_preview:
            assert isinstance(col, str)
            assert isinstance(corr, float)

    def test_null_rates_reported(self, tmp_path):
        """Null rates are computed for new columns."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            auto_clean=False,
        )

        assert "coach_win_pct" in result.null_rates
        assert "coach_tenure" in result.null_rates

    def test_no_new_columns_returns_warning(self, tmp_path):
        """If all columns already exist, returns warning."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "features.parquet"
        _make_base_parquet(parquet_path)

        # Create CSV with same columns as existing parquet
        rng = np.random.default_rng(42)
        dup_df = pd.DataFrame({
            "season": rng.choice([2022, 2023, 2024], size=200),
            "game_id": np.arange(200),
            "feat_x": rng.standard_normal(200),
        })
        dup_path = tmp_path / "dup.csv"
        dup_df.to_csv(dup_path, index=False)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(dup_path),
            auto_clean=False,
        )

        assert len(result.columns_added) == 0
        assert len(result.warnings) > 0

    def test_missing_file_raises(self, tmp_path):
        """Missing data file raises FileNotFoundError."""
        feat_dir = tmp_path / "data" / "features"
        _make_base_parquet(feat_dir / "features.parquet")

        with pytest.raises(FileNotFoundError):
            ingest_dataset(
                project_dir=tmp_path,
                data_path=str(tmp_path / "nonexistent.csv"),
            )

    def test_no_join_keys_raises(self, tmp_path):
        """When join keys can't be detected, raises ValueError."""
        feat_dir = tmp_path / "data" / "features"
        _make_base_parquet(feat_dir / "features.parquet")

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
        parquet_path = custom_dir / "features.parquet"
        _make_base_parquet(parquet_path)
        csv_path = tmp_path / "new_data.csv"
        _make_new_csv(csv_path)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            features_dir=str(custom_dir),
            auto_clean=False,
        )

        assert len(result.columns_added) > 0
        updated = pd.read_parquet(parquet_path)
        assert "coach_win_pct" in updated.columns

    def test_bootstrap_first_dataset(self, tmp_path):
        """First dataset creates the features file from scratch."""
        csv_path = tmp_path / "initial_data.csv"
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "id": np.arange(100),
            "result": rng.integers(0, 2, size=100),
            "revenue": rng.uniform(100, 10000, size=100),
            "churn_score": rng.standard_normal(100),
        })
        df.to_csv(csv_path, index=False)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            auto_clean=False,
        )

        assert result.is_bootstrap is True
        assert result.rows_total == 100
        assert result.rows_matched == 100
        assert "revenue" in result.columns_added

        # Verify parquet was created
        parquet_path = tmp_path / "data" / "features" / "features.parquet"
        assert parquet_path.exists()
        saved = pd.read_parquet(parquet_path)
        assert len(saved) == 100
        assert "revenue" in saved.columns

    def test_bootstrap_with_auto_clean(self, tmp_path):
        """Bootstrap with auto-clean applies cleaning before saving."""
        csv_path = tmp_path / "dirty_data.csv"
        df = pd.DataFrame({
            "id": [1, 2, 3, 3, 4],
            "result": [0, 1, 0, 0, 1],
            "score": [1.0, np.nan, 3.0, 3.0, 5.0],
        })
        df.to_csv(csv_path, index=False)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            auto_clean=True,
        )

        assert result.is_bootstrap is True
        assert len(result.cleaning_actions) > 0
        # Should fill nulls and drop duplicates
        assert any("null" in a.lower() or "median" in a.lower() for a in result.cleaning_actions)
        assert any("duplicate" in a for a in result.cleaning_actions)

    def test_auto_clean_on_merge(self, tmp_path):
        """Auto-clean applies to new data before merging."""
        feat_dir = tmp_path / "data" / "features"
        _make_base_parquet(feat_dir / "features.parquet")

        # Create CSV with data that has nulls needing cleaning
        csv_path = tmp_path / "dirty.csv"
        rng = np.random.default_rng(42)
        n = 200
        rating = rng.standard_normal(n)
        rating[0] = np.nan  # introduce a null
        df = pd.DataFrame({
            "season": rng.choice([2022, 2023, 2024], size=n),
            "game_id": np.arange(n),
            "rating": rating,
        })
        df.to_csv(csv_path, index=False)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            auto_clean=True,
        )

        assert any("null" in a.lower() or "median" in a.lower() for a in result.cleaning_actions)
        assert "rating" in result.columns_added

    def test_source_registered(self, tmp_path):
        """Source is registered in source_registry.json."""
        csv_path = tmp_path / "data.csv"
        rng = np.random.default_rng(42)
        pd.DataFrame({
            "id": np.arange(50),
            "result": rng.integers(0, 2, size=50),
            "feat": rng.standard_normal(50),
        }).to_csv(csv_path, index=False)

        result = ingest_dataset(
            project_dir=tmp_path,
            data_path=str(csv_path),
            auto_clean=False,
        )

        assert result.source_registered is True
        import json
        registry = json.loads((tmp_path / "data" / "source_registry.json").read_text())
        assert len(registry["sources"]) == 1
        assert registry["sources"][0]["name"] == "data"


class TestIngestResultFormat:
    """Test IngestResult.format_summary()."""

    def test_format_summary_produces_markdown(self):
        result = IngestResult(
            name="coaching_data",
            columns_added=["coach_win_pct", "coach_tenure"],
            rows_matched=180,
            rows_total=200,
            null_rates={"coach_win_pct": 0.05, "coach_tenure": 0.15},
            correlation_preview=[
                ("coach_win_pct", 0.23),
                ("coach_tenure", 0.18),
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
        assert "coach_tenure" in md

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

    def test_format_summary_bootstrap(self):
        result = IngestResult(
            name="initial",
            columns_added=["a", "b"],
            rows_matched=100,
            rows_total=100,
            null_rates={},
            correlation_preview=[],
            is_bootstrap=True,
        )
        md = result.format_summary()
        assert "Bootstrap" in md

    def test_format_summary_cleaning_actions(self):
        result = IngestResult(
            name="cleaned",
            columns_added=["feat"],
            rows_matched=100,
            rows_total=100,
            null_rates={},
            correlation_preview=[],
            cleaning_actions=["Dropped 5 exact duplicate rows"],
        )
        md = result.format_summary()
        assert "Auto-Clean" in md
        assert "duplicate" in md


class TestValidateDataset:
    """Test validate_dataset preview tool."""

    def test_shows_schema(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({
            "id": [1, 2, 3],
            "val": [1.0, 2.0, 3.0],
        }).to_csv(csv_path, index=False)

        report = validate_dataset(tmp_path, str(csv_path))
        assert "Dataset Preview" in report
        assert "id" in report
        assert "val" in report
        assert "Rows" in report

    def test_detects_issues(self, tmp_path):
        # Write a file where pandas will keep a column as string
        # Use a parquet file with explicit string dtype
        parquet_path = tmp_path / "dirty.parquet"
        df = pd.DataFrame({
            "num_as_str": pd.array(["1.0", "2.0", "3.0"], dtype="string"),
            "half_null": [1.0, np.nan, np.nan],
        })
        df.to_parquet(parquet_path, index=False)

        report = validate_dataset(tmp_path, str(parquet_path))
        assert "Potential Issues" in report
        assert "numeric but stored as string" in report

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate_dataset(tmp_path, str(tmp_path / "nope.csv"))


class TestFillNulls:
    """Test fill_nulls tool."""

    def test_fills_with_median(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0], "result": [0, 1, 0, 1, 0]})
        df.to_parquet(feat_dir / "features.parquet", index=False)

        msg = fill_nulls(tmp_path, "x", strategy="median")
        assert "1" in msg  # 1 null filled

        updated = pd.read_parquet(feat_dir / "features.parquet")
        assert updated["x"].isna().sum() == 0

    def test_fills_with_value(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        df.to_parquet(feat_dir / "features.parquet", index=False)

        msg = fill_nulls(tmp_path, "x", strategy="value", value=999)
        updated = pd.read_parquet(feat_dir / "features.parquet")
        assert updated["x"].iloc[1] == 999

    def test_no_nulls_returns_message(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        df.to_parquet(feat_dir / "features.parquet", index=False)

        msg = fill_nulls(tmp_path, "x")
        assert "no nulls" in msg.lower()

    def test_invalid_column_raises(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        pd.DataFrame({"x": [1.0]}).to_parquet(feat_dir / "features.parquet", index=False)

        with pytest.raises(ValueError, match="not found"):
            fill_nulls(tmp_path, "nonexistent")


class TestDropDuplicates:
    """Test drop_duplicates tool."""

    def test_drops_exact_duplicates(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        df = pd.DataFrame({"x": [1, 1, 2, 3], "y": [10, 10, 20, 30]})
        df.to_parquet(feat_dir / "features.parquet", index=False)

        msg = drop_duplicates(tmp_path)
        assert "1" in msg  # 1 duplicate dropped

        updated = pd.read_parquet(feat_dir / "features.parquet")
        assert len(updated) == 3

    def test_drops_on_subset(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        df = pd.DataFrame({"x": [1, 1, 2], "y": [10, 20, 30]})
        df.to_parquet(feat_dir / "features.parquet", index=False)

        msg = drop_duplicates(tmp_path, columns=["x"])
        assert "1" in msg

    def test_no_duplicates_message(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        df = pd.DataFrame({"x": [1, 2, 3]})
        df.to_parquet(feat_dir / "features.parquet", index=False)

        msg = drop_duplicates(tmp_path)
        assert "No duplicates" in msg


class TestRenameColumns:
    """Test rename_columns tool."""

    def test_renames_columns(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        df = pd.DataFrame({"old_name": [1, 2], "keep": [3, 4]})
        df.to_parquet(feat_dir / "features.parquet", index=False)

        msg = rename_columns(tmp_path, {"old_name": "new_name"})
        assert "old_name" in msg
        assert "new_name" in msg

        updated = pd.read_parquet(feat_dir / "features.parquet")
        assert "new_name" in updated.columns
        assert "old_name" not in updated.columns

    def test_missing_column_raises(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        pd.DataFrame({"x": [1]}).to_parquet(feat_dir / "features.parquet", index=False)

        with pytest.raises(ValueError, match="not found"):
            rename_columns(tmp_path, {"nonexistent": "new"})
