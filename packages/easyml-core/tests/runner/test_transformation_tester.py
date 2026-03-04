"""Tests for transformation testing tool."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.core.runner.schema import FeatureDef, FeatureType
from easyml.core.runner.transformation_tester import (
    TransformationReport,
    TransformationResult,
    run_transformation_tests,
    _compute_correlation,
    _find_top_interaction_partners,
)


def _make_features_parquet(path: Path, n: int = 300) -> None:
    """Create a base features.parquet with several features."""
    rng = np.random.default_rng(42)
    result = rng.integers(0, 2, size=n)
    df = pd.DataFrame({
        "season": rng.choice([2022, 2023, 2024], size=n),
        "result": result,
        "diff_x": rng.standard_normal(n),
        "diff_y": rng.standard_normal(n),
        "diff_z": result.astype(float) + rng.normal(0, 0.5, n),  # correlated with target
        "diff_seed_num": rng.integers(-15, 16, size=n).astype(float),
        "diff_adj_em": rng.standard_normal(n) * 5,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


class TestComputeCorrelation:
    """Test correlation computation helper."""

    def test_perfect_positive(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        t = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert _compute_correlation(s, t) == pytest.approx(1.0, abs=0.01)

    def test_handles_nan(self):
        s = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        t = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        corr = _compute_correlation(s, t)
        assert isinstance(corr, float)
        assert corr > 0.9

    def test_too_few_samples(self):
        s = pd.Series([1.0, 2.0])
        t = pd.Series([1.0, 2.0])
        assert _compute_correlation(s, t) == 0.0


class TestFindInteractionPartners:
    """Test interaction partner discovery."""

    def test_finds_partners(self):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "diff_x": rng.standard_normal(n),
            "diff_y": rng.standard_normal(n),
            "diff_z": rng.standard_normal(n),
            "result": rng.integers(0, 2, size=n),
        })
        partners = _find_top_interaction_partners(df, "diff_x", "result", n=2)
        assert len(partners) <= 2
        assert "diff_x" not in partners
        assert "result" not in partners

    def test_respects_feature_columns_filter(self):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "diff_x": rng.standard_normal(n),
            "season": rng.choice([2022, 2023], size=n).astype(float),
            "other_col": rng.standard_normal(n),
            "result": rng.integers(0, 2, size=n),
        })
        # When feature_columns is specified, only those are considered as partners
        partners = _find_top_interaction_partners(
            df, "diff_x", "result", feature_columns=["diff_x", "other_col"]
        )
        assert "season" not in partners
        assert "result" not in partners


class TestTestTransformations:
    """Test end-to-end transformation testing."""

    def test_standard_transformations(self, tmp_path):
        """Tests all standard transformations on a feature."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x"],
            test_interactions=False,
        )

        assert isinstance(report, TransformationReport)
        assert len(report.results) > 0
        # Should test raw, log, sqrt, cbrt, square, reciprocal, rank, zscore
        trans_names = {r.transformation for r in report.results}
        assert "raw" in trans_names
        assert "log" in trans_names
        assert "sqrt" in trans_names

    def test_interactions_tested(self, tmp_path):
        """With interactions enabled, tests multiply/divide/subtract with partners."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x"],
            test_interactions=True,
            top_interaction_partners=2,
        )

        # Should have some interaction results
        interaction_results = [
            r for r in report.results
            if "multiply" in r.transformation or "divide" in r.transformation
        ]
        assert len(interaction_results) > 0

    def test_improvement_calculation(self, tmp_path):
        """Improvement is calculated vs raw baseline."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x"],
            test_interactions=False,
        )

        raw_result = next(
            r for r in report.results if r.transformation == "raw"
        )
        assert raw_result.improvement == pytest.approx(0.0, abs=1e-6)

        # Other transformations have improvement != 0 (in general)
        non_raw = [r for r in report.results if r.transformation != "raw"]
        assert len(non_raw) > 0

    def test_best_per_feature_populated(self, tmp_path):
        """best_per_feature dict is populated."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x", "diff_y"],
            test_interactions=False,
        )

        assert "diff_x" in report.best_per_feature
        assert "diff_y" in report.best_per_feature

    def test_suggested_features_populated(self, tmp_path):
        """Suggested features include non-raw improvements."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        # diff_z is correlated with target, so transformations may improve it
        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_z"],
            test_interactions=False,
        )

        # If best is not raw, should have a suggestion
        if report.best_per_feature.get("diff_z") and \
           report.best_per_feature["diff_z"].transformation != "raw":
            assert len(report.suggested_features) > 0

    def test_format_summary_produces_markdown(self, tmp_path):
        """format_summary produces markdown table."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x"],
            test_interactions=False,
        )

        md = report.format_summary()
        assert "Transformation Test Results" in md
        assert "| Feature" in md
        assert "diff_x" in md

    def test_get_create_commands(self, tmp_path):
        """get_create_commands returns valid feature defs."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_z"],
            test_interactions=False,
        )

        commands = report.get_create_commands()
        # Commands should match suggested_features
        assert len(commands) == len(report.suggested_features)
        for cmd in commands:
            assert "name" in cmd
            assert "formula" in cmd

    def test_zero_safe_transformations(self, tmp_path):
        """Transformations handle zeros safely."""
        feat_dir = tmp_path / "data" / "features"
        parquet_path = feat_dir / "features.parquet"
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "season": rng.choice([2022, 2023], size=n),
            "result": rng.integers(0, 2, size=n),
            "diff_zeros": np.zeros(n),
        })
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet_path, index=False)

        # Should not raise
        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_zeros"],
            test_interactions=False,
        )
        assert isinstance(report, TransformationReport)

    def test_subset_transformations(self, tmp_path):
        """Can specify a subset of transformations."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x"],
            transformations=["raw", "log"],
            test_interactions=False,
        )

        trans_names = {r.transformation for r in report.results}
        assert trans_names == {"raw", "log"}

    def test_missing_feature_skipped(self, tmp_path):
        """Missing features are skipped with a warning."""
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x", "nonexistent_feature"],
            test_interactions=False,
        )

        # Only diff_x results should be present
        features_tested = {r.feature for r in report.results}
        assert "diff_x" in features_tested
        assert "nonexistent_feature" not in features_tested

    def test_missing_parquet_raises(self, tmp_path):
        """Missing parquet file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            run_transformation_tests(
                project_dir=tmp_path,
                features=["diff_x"],
            )


class TestTransformationReportFormat:
    """Test TransformationReport.format_summary() edge cases."""

    def test_empty_report(self):
        report = TransformationReport()
        md = report.format_summary()
        assert "No transformation results" in md

    def test_with_suggestions(self):
        report = TransformationReport(
            results=[
                TransformationResult(
                    feature="diff_x", transformation="log",
                    formula="log(diff_x)", correlation=0.35,
                    abs_correlation=0.35, improvement=0.05, null_rate=0.0,
                ),
            ],
            best_per_feature={
                "diff_x": TransformationResult(
                    feature="diff_x", transformation="log",
                    formula="log(diff_x)", correlation=0.35,
                    abs_correlation=0.35, improvement=0.05, null_rate=0.0,
                ),
            },
            suggested_features=[{
                "name": "log_x",
                "formula": "log(diff_x)",
                "correlation": 0.35,
            }],
        )
        md = report.format_summary()
        assert "Suggested Features" in md
        assert "log_x" in md


# -----------------------------------------------------------------------
# Store-aware tests (feature_defs parameter)
# -----------------------------------------------------------------------

class TestStoreAwareTransformations:
    """Tests that feature_defs annotates results with feature type."""

    @pytest.fixture()
    def sample_defs(self):
        return {
            "diff_x": FeatureDef(
                name="diff_x", type=FeatureType.PAIRWISE,
                formula="x_a - x_b", category="general",
            ),
            "diff_z": FeatureDef(
                name="diff_z", type=FeatureType.PAIRWISE,
                formula="z_a - z_b", category="efficiency",
            ),
        }

    def test_results_annotated_with_type(self, tmp_path, sample_defs):
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x"],
            test_interactions=False,
            feature_defs=sample_defs,
        )

        for r in report.results:
            assert r.feature_type == "pairwise"

    def test_best_per_feature_has_type(self, tmp_path, sample_defs):
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x"],
            test_interactions=False,
            feature_defs=sample_defs,
        )

        assert report.best_per_feature["diff_x"].feature_type == "pairwise"

    def test_format_summary_shows_type_column(self, tmp_path, sample_defs):
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x"],
            test_interactions=False,
            feature_defs=sample_defs,
        )

        md = report.format_summary()
        assert "| Type |" in md
        assert "pairwise" in md

    def test_without_defs_no_type_column(self, tmp_path):
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_x"],
            test_interactions=False,
        )

        md = report.format_summary()
        assert "| Type |" not in md

    def test_unregistered_feature_empty_type(self, tmp_path, sample_defs):
        feat_dir = tmp_path / "data" / "features"
        _make_features_parquet(feat_dir / "features.parquet")

        # diff_y is not in sample_defs
        report = run_transformation_tests(
            project_dir=tmp_path,
            features=["diff_y"],
            test_interactions=False,
            feature_defs=sample_defs,
        )

        for r in report.results:
            assert r.feature_type == ""
