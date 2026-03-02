"""Tests for declarative feature creation engine."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.runner.feature_engine import (
    FeatureResult,
    _resolve_formula,
    _topological_sort_features,
)


def _make_features_parquet(path: Path, n: int = 200) -> None:
    """Create a base features.parquet."""
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
