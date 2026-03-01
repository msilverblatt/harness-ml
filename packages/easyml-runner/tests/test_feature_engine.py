"""Tests for declarative feature creation engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.runner.feature_engine import (
    FeatureResult,
    create_feature,
    create_features_batch,
    _resolve_formula,
    _topological_sort_features,
)


def _make_features_parquet(path: Path, n: int = 200) -> None:
    """Create a base matchup_features.parquet."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "season": rng.choice([2022, 2023, 2024], size=n),
        "result": rng.integers(0, 2, size=n),
        "diff_x": rng.standard_normal(n),
        "diff_y": rng.standard_normal(n),
        "diff_seed_num": rng.integers(-15, 16, size=n).astype(float),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


class TestResolveFormula:
    """Test safe formula evaluation."""

    def test_simple_arithmetic(self):
        df = pd.DataFrame({"diff_x": [1.0, 2.0, 3.0], "diff_y": [4.0, 5.0, 6.0]})
        result = _resolve_formula("diff_x + diff_y", df)
        np.testing.assert_array_almost_equal(result.values, [5.0, 7.0, 9.0])

    def test_multiplication(self):
        df = pd.DataFrame({"diff_x": [2.0, 3.0], "diff_y": [4.0, 5.0]})
        result = _resolve_formula("diff_x * diff_y", df)
        np.testing.assert_array_almost_equal(result.values, [8.0, 15.0])

    def test_log_function(self):
        df = pd.DataFrame({"diff_x": [1.0, 2.0, 3.0]})
        result = _resolve_formula("log(diff_x)", df)
        expected = np.log(np.abs([1.0, 2.0, 3.0]) + 1)
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_sqrt_function(self):
        df = pd.DataFrame({"diff_x": [4.0, 9.0, -16.0]})
        result = _resolve_formula("sqrt(diff_x)", df)
        expected = np.sqrt(np.abs([4.0, 9.0, -16.0]))
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_cbrt_function(self):
        df = pd.DataFrame({"diff_x": [8.0, -27.0]})
        result = _resolve_formula("cbrt(diff_x)", df)
        expected = np.cbrt([8.0, -27.0])
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_feature_reference(self):
        df = pd.DataFrame({"diff_x": [1.0, 2.0]})
        created = {"my_feat": pd.Series([10.0, 20.0])}
        result = _resolve_formula("@my_feat + diff_x", df, created_features=created)
        np.testing.assert_array_almost_equal(result.values, [11.0, 22.0])

    def test_missing_column_raises(self):
        df = pd.DataFrame({"diff_x": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            _resolve_formula("nonexistent_col + 1", df)

    def test_missing_feature_ref_raises(self):
        df = pd.DataFrame({"diff_x": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            _resolve_formula("@nonexistent + 1", df)

    def test_division_with_constant(self):
        df = pd.DataFrame({"diff_x": [10.0, 20.0]})
        result = _resolve_formula("diff_x / 2", df)
        np.testing.assert_array_almost_equal(result.values, [5.0, 10.0])

    def test_power_operator(self):
        df = pd.DataFrame({"diff_x": [2.0, 3.0]})
        result = _resolve_formula("diff_x ** 2", df)
        np.testing.assert_array_almost_equal(result.values, [4.0, 9.0])


class TestCreateFeature:
    """Test end-to-end feature creation."""

    def test_simple_feature(self, tmp_path):
        """Create a feature from simple formula."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "matchup_features.parquet")

        result = create_feature(
            project_dir=tmp_path,
            name="x_times_y",
            formula="diff_x * diff_y",
        )

        assert result.column_added == "diff_x_times_y"
        assert isinstance(result.correlation, float)
        assert "mean" in result.stats
        assert "std" in result.stats

        # Verify saved to parquet
        df = pd.read_parquet(feat_dir / "matchup_features.parquet")
        assert "diff_x_times_y" in df.columns

    def test_feature_with_diff_prefix(self, tmp_path):
        """Feature name already has diff_ prefix."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "matchup_features.parquet")

        result = create_feature(
            project_dir=tmp_path,
            name="diff_custom",
            formula="diff_x + diff_y",
        )

        assert result.column_added == "diff_custom"

    def test_log_transformation(self, tmp_path):
        """Create log-transformed feature."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "matchup_features.parquet")

        result = create_feature(
            project_dir=tmp_path,
            name="log_x",
            formula="log(diff_x)",
        )

        assert result.column_added == "diff_log_x"
        assert result.null_rate < 1.0  # log(|x|+1) should not produce NaN

    def test_correlation_computed(self, tmp_path):
        """Correlation with target is computed."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "matchup_features.parquet")

        result = create_feature(
            project_dir=tmp_path,
            name="x_feat",
            formula="diff_x",
        )

        assert result.abs_correlation == abs(result.correlation)

    def test_save_false_no_parquet_change(self, tmp_path):
        """save=False does not modify parquet."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "matchup_features.parquet"
        _make_features_parquet(parquet_path)

        df_before = pd.read_parquet(parquet_path)

        create_feature(
            project_dir=tmp_path,
            name="temp_feat",
            formula="diff_x * 2",
            save=False,
        )

        df_after = pd.read_parquet(parquet_path)
        assert list(df_before.columns) == list(df_after.columns)

    def test_missing_parquet_raises(self, tmp_path):
        """Missing parquet raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            create_feature(
                project_dir=tmp_path,
                name="feat",
                formula="diff_x + 1",
            )

    def test_bad_formula_raises(self, tmp_path):
        """Invalid formula raises ValueError."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "matchup_features.parquet")

        with pytest.raises(ValueError):
            create_feature(
                project_dir=tmp_path,
                name="bad",
                formula="nonexistent_column * 2",
            )

    def test_null_handling(self, tmp_path):
        """Features computed from columns with NaN handle them."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "matchup_features.parquet"
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "season": rng.choice([2022, 2023], size=n),
            "result": rng.integers(0, 2, size=n),
            "diff_x": [np.nan if i % 5 == 0 else rng.standard_normal() for i in range(n)],
        })
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

        result = create_feature(
            project_dir=tmp_path,
            name="x_doubled",
            formula="diff_x * 2",
        )

        assert result.null_rate > 0  # should have some nulls


class TestCreateFeaturesBatch:
    """Test batch feature creation."""

    def test_batch_creates_multiple(self, tmp_path):
        """Creates multiple features in one call."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "matchup_features.parquet")

        features = [
            {"name": "sum_xy", "formula": "diff_x + diff_y"},
            {"name": "prod_xy", "formula": "diff_x * diff_y"},
        ]

        results = create_features_batch(tmp_path, features)

        assert len(results) == 2
        assert results[0].column_added == "diff_sum_xy"
        assert results[1].column_added == "diff_prod_xy"

        df = pd.read_parquet(feat_dir / "matchup_features.parquet")
        assert "diff_sum_xy" in df.columns
        assert "diff_prod_xy" in df.columns

    def test_batch_with_dependencies(self, tmp_path):
        """Features can reference other features in the batch with @."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "matchup_features.parquet")

        features = [
            {"name": "sum_xy", "formula": "diff_x + diff_y"},
            {"name": "scaled_sum", "formula": "@sum_xy * 2"},
        ]

        results = create_features_batch(tmp_path, features)

        assert len(results) == 2
        # sum_xy should be created before scaled_sum
        df = pd.read_parquet(feat_dir / "matchup_features.parquet")
        assert "diff_sum_xy" in df.columns
        assert "diff_scaled_sum" in df.columns

    def test_circular_dependency_raises(self, tmp_path):
        """Circular @-references raise ValueError."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "matchup_features.parquet")

        features = [
            {"name": "a", "formula": "@b + 1"},
            {"name": "b", "formula": "@a + 1"},
        ]

        with pytest.raises(ValueError, match="Circular"):
            create_features_batch(tmp_path, features)


class TestTopologicalSort:
    """Test topological sort of feature dependencies."""

    def test_no_deps(self):
        features = [
            {"name": "a", "formula": "x + 1"},
            {"name": "b", "formula": "y + 2"},
        ]
        ordered = _topological_sort_features(features, {"a", "b"})
        assert len(ordered) == 2

    def test_respects_deps(self):
        features = [
            {"name": "derived", "formula": "@base * 2"},
            {"name": "base", "formula": "x + 1"},
        ]
        ordered = _topological_sort_features(features, {"base", "derived"})
        names = [f["name"] for f in ordered]
        assert names.index("base") < names.index("derived")


class TestFeatureResultFormat:
    """Test FeatureResult.format_summary()."""

    def test_format_produces_markdown(self):
        result = FeatureResult(
            name="efficiency_ratio",
            column_added="diff_efficiency_ratio",
            correlation=0.31,
            abs_correlation=0.31,
            null_rate=0.02,
            stats={"mean": 0.05, "std": 1.2, "min": -4.1, "max": 3.8},
            description="Offensive/defensive efficiency ratio",
        )
        md = result.format_summary()
        assert "diff_efficiency_ratio" in md
        assert "+0.3100" in md
        assert "2.0%" in md

    def test_format_shows_redundancy_warning(self):
        result = FeatureResult(
            name="dup_feat",
            column_added="diff_dup_feat",
            correlation=0.2,
            abs_correlation=0.2,
            redundant_with=["diff_x", "diff_y"],
        )
        md = result.format_summary()
        assert "Redundant" in md
        assert "diff_x" in md
