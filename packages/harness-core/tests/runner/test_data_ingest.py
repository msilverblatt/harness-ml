"""Tests for declarative data ingestion."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harnessml.core.runner.data_ingest import (
    IngestResult,
    ingest_dataset,
    validate_dataset,
    fill_nulls,
    drop_duplicates,
    drop_rows,
    rename_columns,
    derive_column,
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
        from harnessml.core.runner.data_ingest import _detect_join_keys

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
        from harnessml.core.runner.data_ingest import _detect_join_keys

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
        assert "Null Summary" in md
        assert "coach_tenure" in md
        assert "fill_nulls" in md

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
            auto_clean=True,
        )
        md = result.format_summary()
        assert "Auto-Clean Applied" in md
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


class TestDeriveColumn:
    """Test derive_column tool."""

    def _setup_features(self, tmp_path, df):
        """Helper to create a features parquet in the standard location."""
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        df.to_parquet(feat_dir / "features.parquet", index=False)

    def test_simple_expression(self, tmp_path):
        """close - open arithmetic."""
        df = pd.DataFrame({
            "open": [10.0, 20.0, 30.0],
            "close": [12.0, 18.0, 35.0],
        })
        self._setup_features(tmp_path, df)

        msg = derive_column(tmp_path, "spread", "close - open")

        updated = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
        assert "spread" in updated.columns
        pd.testing.assert_series_equal(
            updated["spread"],
            pd.Series([2.0, -2.0, 5.0], name="spread"),
        )

    def test_grouped_shift(self, tmp_path):
        """close.shift(-1) / close - 1 with group_by=ticker."""
        df = pd.DataFrame({
            "ticker": ["A", "A", "A", "B", "B", "B"],
            "close": [100.0, 110.0, 121.0, 50.0, 55.0, 60.0],
        })
        self._setup_features(tmp_path, df)

        msg = derive_column(
            tmp_path, "fwd_return",
            "close.shift(-1) / close - 1",
            group_by="ticker",
        )

        updated = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
        assert "fwd_return" in updated.columns
        # For ticker A: 110/100 - 1 = 0.1, 121/110 - 1 = 0.1, NaN
        assert updated["fwd_return"].iloc[0] == pytest.approx(0.1)
        assert updated["fwd_return"].iloc[1] == pytest.approx(0.1)
        assert pd.isna(updated["fwd_return"].iloc[2])
        # For ticker B: 55/50 - 1 = 0.1, 60/55 - 1 ~= 0.0909, NaN
        assert updated["fwd_return"].iloc[3] == pytest.approx(0.1)
        assert updated["fwd_return"].iloc[4] == pytest.approx(60.0 / 55.0 - 1)
        assert pd.isna(updated["fwd_return"].iloc[5])

    def test_binary_threshold(self, tmp_path):
        """(value > 0).astype(int)."""
        df = pd.DataFrame({
            "value": [-1.0, 0.0, 1.0, 2.0, -0.5],
        })
        self._setup_features(tmp_path, df)

        msg = derive_column(tmp_path, "is_positive", "(value > 0).astype(int)")

        updated = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
        assert "is_positive" in updated.columns
        expected = pd.Series([0, 0, 1, 1, 0], name="is_positive")
        pd.testing.assert_series_equal(updated["is_positive"], expected)

    def test_datetime_extract(self, tmp_path):
        """date.dt.year."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-15", "2021-06-20", "2022-12-01"]),
        })
        self._setup_features(tmp_path, df)

        msg = derive_column(tmp_path, "year", "date.dt.year")

        updated = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
        assert "year" in updated.columns
        assert list(updated["year"]) == [2020, 2021, 2022]

    def test_dtype_cast(self, tmp_path):
        """Verify dtype parameter casts the result."""
        df = pd.DataFrame({
            "value": [1.5, 2.7, 3.1],
        })
        self._setup_features(tmp_path, df)

        msg = derive_column(tmp_path, "rounded", "value", dtype="int")

        updated = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
        assert updated["rounded"].dtype in [np.int64, np.int32, int]

    def test_missing_features_file_raises(self, tmp_path):
        """Raises FileNotFoundError if features parquet doesn't exist."""
        with pytest.raises(FileNotFoundError):
            derive_column(tmp_path, "x", "1 + 1")

    def test_bad_expression_raises(self, tmp_path):
        """Raises ValueError on invalid expression."""
        df = pd.DataFrame({"x": [1.0, 2.0]})
        self._setup_features(tmp_path, df)

        with pytest.raises(ValueError, match="expression"):
            derive_column(tmp_path, "bad", "nonexistent_col + 1")


class TestDeriveColumnTargetEngineering:
    """Integration test: derive_column for target engineering (forward return + binary target)."""

    def _setup_features(self, tmp_path, df):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(feat_dir / "features.parquet", index=False)

    def test_forward_return_then_binary_target(self, tmp_path):
        """Create forward return with group_by, then derive binary target from it."""
        df = pd.DataFrame({
            "ticker": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "close": [100.0, 110.0, 105.0, 120.0, 50.0, 55.0, 52.0, 60.0],
        })
        self._setup_features(tmp_path, df)

        # Step 1: derive forward return grouped by ticker
        derive_column(
            tmp_path, "fwd_return",
            "close.shift(-1) / close - 1",
            group_by="ticker",
        )

        updated = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
        assert "fwd_return" in updated.columns
        # Ticker A row 0: 110/100 - 1 = 0.1
        assert updated["fwd_return"].iloc[0] == pytest.approx(0.1)
        # Last row per group should be NaN
        assert pd.isna(updated["fwd_return"].iloc[3])
        assert pd.isna(updated["fwd_return"].iloc[7])

        # Step 2: derive binary target from forward return
        derive_column(
            tmp_path, "target",
            "(fwd_return > 0).astype(int)",
        )

        final = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
        assert "target" in final.columns
        # Ticker A: fwd_return = [0.1, -0.0455, 0.1429, NaN] -> target = [1, 0, 1, NaN-cast-to-int]
        assert final["target"].iloc[0] == 1
        assert final["target"].iloc[1] == 0
        assert final["target"].iloc[2] == 1


class TestEndToEndConfigDrivenPath:
    """Full path: DataConfig -> ingest -> profiler -> guards all use same config."""

    def test_custom_features_file_works_e2e(self, tmp_path):
        """Ingest, profile, and guard check all work with custom features_file."""
        # Setup project structure
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text(
            "data:\n"
            "  features_dir: data/features\n"
            "  features_file: my_dataset.parquet\n"
            "  target_column: outcome\n"
            "  key_columns: [id, period]\n"
            "  time_column: period\n"
        )
        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)

        # Create initial data
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "period": [2020] * 5 + [2021] * 5,
            "outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "feat_a": np.random.default_rng(42).normal(0, 1, 10).tolist(),
        })
        df.to_parquet(features_dir / "my_dataset.parquet", index=False)

        # Verify config loads correctly
        from harnessml.core.runner.data_utils import load_data_config, get_features_path

        config = load_data_config(tmp_path)
        assert config.features_file == "my_dataset.parquet"
        assert config.target_column == "outcome"

        parquet_path = get_features_path(tmp_path, config)
        assert parquet_path.exists()

        # Verify profiler works with config
        from harnessml.core.runner.data_profiler import profile_dataset

        profile = profile_dataset(parquet_path, config=config)
        assert profile.n_rows == 10
        assert profile.label_column == "outcome"
        assert profile.time_column == "period"
        # Key columns should not appear in feature_columns
        feature_names = [c.name for c in profile.feature_columns]
        assert "id" not in feature_names
        assert "period" not in feature_names
        assert "outcome" not in feature_names
        assert "feat_a" in feature_names

        # Verify guards work with config
        from harnessml.core.runner.stage_guards import PipelineGuards

        guards = PipelineGuards(config, tmp_path)
        guards.guard_train()  # Should not raise
        guards.guard_backtest()  # Should not raise


class TestDropRows:
    """Test drop_rows tool."""

    def _setup_features(self, tmp_path, df):
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(feat_dir / "features.parquet", index=False)

    def test_drop_nulls_in_column(self, tmp_path):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, np.nan, 30.0, np.nan, 50.0],
        })
        self._setup_features(tmp_path, df)

        msg = drop_rows(tmp_path, column="b", condition="null")
        assert "2" in msg  # 2 rows dropped
        assert "3" in msg  # 3 rows remaining

        updated = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
        assert len(updated) == 3
        assert updated["b"].isna().sum() == 0
        assert list(updated["a"]) == [1.0, 3.0, 5.0]

    def test_drop_by_expression(self, tmp_path):
        df = pd.DataFrame({
            "value": [0.5, 1.5, -0.3, 2.0, 0.8],
        })
        self._setup_features(tmp_path, df)

        msg = drop_rows(tmp_path, condition="value < 1")
        assert "Dropped" in msg

        updated = pd.read_parquet(tmp_path / "data" / "features" / "features.parquet")
        assert len(updated) == 2
        assert all(v >= 1 for v in updated["value"])

    def test_drop_null_requires_column(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        self._setup_features(tmp_path, df)

        msg = drop_rows(tmp_path, condition="null")
        assert "Error" in msg
        assert "column" in msg.lower()

    def test_no_rows_matched(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        self._setup_features(tmp_path, df)

        msg = drop_rows(tmp_path, column="x", condition="null")
        assert "nothing dropped" in msg.lower()

    def test_invalid_expression_raises(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, 2.0]})
        self._setup_features(tmp_path, df)

        with pytest.raises(ValueError, match="condition"):
            drop_rows(tmp_path, condition="nonexistent_col > 0")


class TestSampleData:
    def test_sample_fraction(self, tmp_path):
        from harnessml.core.runner.data_ingest import sample_data

        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)
        df = pd.DataFrame({"a": range(1000), "target": [0, 1] * 500})
        df.to_parquet(features_dir / "features.parquet", index=False)

        result = sample_data(tmp_path, fraction=0.1, seed=42)
        assert "Sampled" in result

        df2 = pd.read_parquet(features_dir / "features.parquet")
        assert 80 <= len(df2) <= 120  # ~100 rows

    def test_sample_creates_backup(self, tmp_path):
        from harnessml.core.runner.data_ingest import sample_data

        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)
        df = pd.DataFrame({"a": range(100)})
        df.to_parquet(features_dir / "features.parquet", index=False)

        sample_data(tmp_path, fraction=0.5, seed=42)
        assert (features_dir / "features_full.parquet").exists()
        df_full = pd.read_parquet(features_dir / "features_full.parquet")
        assert len(df_full) == 100

    def test_sample_restore(self, tmp_path):
        from harnessml.core.runner.data_ingest import sample_data, restore_full_data

        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)
        df = pd.DataFrame({"a": range(100)})
        df.to_parquet(features_dir / "features.parquet", index=False)

        sample_data(tmp_path, fraction=0.5, seed=42)
        result = restore_full_data(tmp_path)
        assert "Restored" in result

        df2 = pd.read_parquet(features_dir / "features.parquet")
        assert len(df2) == 100
        assert not (features_dir / "features_full.parquet").exists()

    def test_sample_no_features(self, tmp_path):
        from harnessml.core.runner.data_ingest import sample_data
        result = sample_data(tmp_path, fraction=0.1)
        assert "Error" in result

    def test_restore_no_backup(self, tmp_path):
        from harnessml.core.runner.data_ingest import restore_full_data
        result = restore_full_data(tmp_path)
        assert "Error" in result
