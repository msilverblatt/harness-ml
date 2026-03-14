"""Tests for usability warnings and user-facing messages.

Covers: duplicate source re-ingestion, feature discovery target labels,
exploration budget warnings, view-only feature warnings, and
feature-aware fingerprint cache invalidation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml


class TestFeatureDiscoveryTargetLabel:
    """Correlations should be labeled with the target column name."""

    def test_correlation_df_has_target_attr(self):
        from harnessml.core.runner.features.discovery import compute_feature_correlations

        df = pd.DataFrame({
            "feat_a": [1, 2, 3, 4, 5],
            "feat_b": [5, 4, 3, 2, 1],
            "my_target": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        result = compute_feature_correlations(df, target_col="my_target")
        assert result.attrs.get("target_column") == "my_target"

    def test_report_includes_target_name(self):
        from harnessml.core.runner.features.discovery import (
            compute_feature_correlations,
            format_discovery_report,
        )

        df = pd.DataFrame({
            "feat_a": [1, 2, 3, 4, 5],
            "feat_b": [5, 4, 3, 2, 1],
            "LogSalePrice": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        corr = compute_feature_correlations(df, target_col="LogSalePrice")
        report = format_discovery_report(
            corr,
            pd.DataFrame(columns=["feature", "importance"]),
            [],
            {},
        )
        assert "LogSalePrice" in report

    def test_report_without_target_attr_still_works(self):
        from harnessml.core.runner.features.discovery import format_discovery_report

        corr = pd.DataFrame({
            "feature": ["a", "b"],
            "correlation": [0.5, -0.3],
            "abs_correlation": [0.5, 0.3],
        })
        # No attrs set — should not crash
        report = format_discovery_report(
            corr,
            pd.DataFrame(columns=["feature", "importance"]),
            [],
            {},
        )
        assert "Target Correlations" in report


class TestExplorationBudgetWarning:
    """Exploration should warn when budget is too low for the search space."""

    def test_budget_warning_in_report(self):
        from unittest.mock import MagicMock

        from harnessml.core.runner.optimization.exploration import (
            ExplorationSpace,
            format_exploration_report,
        )

        space = ExplorationSpace(
            axes=[
                {"key": f"models.xgb.params.p{i}", "type": "continuous", "low": 0.0, "high": 1.0}
                for i in range(8)
            ],
            budget=10,  # 10 < 5*8=40
            primary_metric="rmse",
        )

        # Mock study
        study = MagicMock()
        study.best_trial = None
        study.trials = []

        report = format_exploration_report(
            exploration_id="expl-001",
            space=space,
            study=study,
            baseline_metrics={},
            param_importance={},
            trial_results=[],
        )
        assert "Warning" in report
        assert "insufficient" in report.lower()
        assert "40" in report  # recommended minimum

    def test_no_warning_when_budget_sufficient(self):
        from unittest.mock import MagicMock

        from harnessml.core.runner.optimization.exploration import (
            ExplorationSpace,
            format_exploration_report,
        )

        space = ExplorationSpace(
            axes=[
                {"key": "models.xgb.params.lr", "type": "continuous", "low": 0.01, "high": 0.3},
                {"key": "models.xgb.params.depth", "type": "integer", "low": 3, "high": 10},
            ],
            budget=20,  # 20 >= 5*2=10
            primary_metric="rmse",
        )

        study = MagicMock()
        study.best_trial = None
        study.trials = []

        report = format_exploration_report(
            exploration_id="expl-002",
            space=space,
            study=study,
            baseline_metrics={},
            param_importance={},
            trial_results=[],
        )
        assert "insufficient" not in report.lower()


class TestDuplicateSourceWarning:
    """Re-ingesting a source with same columns should give a clear message."""

    def test_reingestion_warns_about_existing_source(self, tmp_path):
        from harnessml.core.runner.data.ingest import ingest_dataset

        project_dir = tmp_path / "project"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        features_dir = project_dir / "data" / "features"
        features_dir.mkdir(parents=True)

        (config_dir / "pipeline.yaml").write_text(
            "data:\n  features_dir: data/features\n  features_file: features.parquet\n"
            "  target_column: target\n  key_columns: [id]\n"
        )

        # Create initial features
        df = pd.DataFrame({"id": [1, 2, 3], "feat_a": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        df.to_parquet(features_dir / "features.parquet", index=False)

        # Create source registry showing "mydata" was previously ingested
        registry = {"mydata": {"columns": ["feat_a"], "rows": 3}}
        (config_dir / "source_registry.json").write_text(json.dumps(registry))

        # Write CSV with same columns
        csv_path = tmp_path / "mydata.csv"
        df.to_csv(csv_path, index=False)

        result = ingest_dataset(project_dir, str(csv_path), name="mydata", join_on=["id"])
        assert "previously ingested" in result.warnings[0].lower() or "already exist" in result.warnings[0].lower()

    def test_new_source_no_new_columns_generic_warning(self, tmp_path):
        from harnessml.core.runner.data.ingest import ingest_dataset

        project_dir = tmp_path / "project"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        features_dir = project_dir / "data" / "features"
        features_dir.mkdir(parents=True)

        (config_dir / "pipeline.yaml").write_text(
            "data:\n  features_dir: data/features\n  features_file: features.parquet\n"
            "  target_column: target\n  key_columns: [id]\n"
        )

        df = pd.DataFrame({"id": [1, 2, 3], "feat_a": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        df.to_parquet(features_dir / "features.parquet", index=False)

        csv_path = tmp_path / "newdata.csv"
        df.to_csv(csv_path, index=False)

        result = ingest_dataset(project_dir, str(csv_path), name="newdata", join_on=["id"])
        assert len(result.warnings) > 0
        assert "already exist" in result.warnings[0].lower()


class TestViewOnlyFeatureWarning:
    """Model update should note when features only exist in views."""

    def test_warns_about_view_derived_features(self, tmp_path):
        from harnessml.core.runner.config_writer.models import update_model

        project_dir = tmp_path / "project"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        features_dir = project_dir / "data" / "features"
        features_dir.mkdir(parents=True)

        (config_dir / "pipeline.yaml").write_text(
            "data:\n  features_dir: data/features\n  features_file: features.parquet\n"
            "  target_column: target\n"
        )

        # Base features — only has feat_a and feat_b
        df = pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0], "target": [0]})
        df.to_parquet(features_dir / "features.parquet", index=False)

        # Models config with existing model
        (config_dir / "models.yaml").write_text(yaml.dump({
            "models": {
                "xgb": {
                    "type": "xgboost",
                    "features": ["feat_a", "feat_b"],
                }
            }
        }))

        # Append a view-derived feature that's not in the base store
        result = update_model(
            project_dir, "xgb",
            append_features=["SFxQualScore", "QualVsNeighQual"],
        )
        assert "view-derived" in result.lower() or "not in base" in result.lower()
        assert "SFxQualScore" in result

    def test_no_warning_when_all_features_in_base(self, tmp_path):
        from harnessml.core.runner.config_writer.models import update_model

        project_dir = tmp_path / "project"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        features_dir = project_dir / "data" / "features"
        features_dir.mkdir(parents=True)

        (config_dir / "pipeline.yaml").write_text(
            "data:\n  features_dir: data/features\n  features_file: features.parquet\n"
            "  target_column: target\n"
        )

        df = pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0], "feat_c": [3.0], "target": [0]})
        df.to_parquet(features_dir / "features.parquet", index=False)

        (config_dir / "models.yaml").write_text(yaml.dump({
            "models": {
                "xgb": {
                    "type": "xgboost",
                    "features": ["feat_a"],
                }
            }
        }))

        result = update_model(project_dir, "xgb", append_features=["feat_b", "feat_c"])
        assert "view-derived" not in result.lower()
        assert "not in base" not in result.lower()


class TestFeatureAwareFingerprints:
    """Fingerprints should change when features change, invalidating cache."""

    def test_different_feature_schema_different_fingerprint(self):
        from harnessml.core.runner.training.fingerprint import compute_fingerprint

        config = {"type": "xgboost", "params": {"max_depth": 6}}
        schema_a = {"columns": ["feat_a", "feat_b"], "dtypes": {"feat_a": "double", "feat_b": "double"}, "row_count": 100}
        schema_b = {"columns": ["feat_a", "feat_b", "feat_c"], "dtypes": {"feat_a": "double", "feat_b": "double", "feat_c": "double"}, "row_count": 100}

        fp_a = compute_fingerprint(model_config=config, feature_schema=schema_a)
        fp_b = compute_fingerprint(model_config=config, feature_schema=schema_b)
        assert fp_a != fp_b

    def test_same_feature_schema_same_fingerprint(self):
        from harnessml.core.runner.training.fingerprint import compute_fingerprint

        config = {"type": "xgboost", "params": {"max_depth": 6}}
        schema = {"columns": ["feat_a", "feat_b"], "dtypes": {"feat_a": "double", "feat_b": "double"}, "row_count": 100}

        fp1 = compute_fingerprint(model_config=config, feature_schema=schema)
        fp2 = compute_fingerprint(model_config=config, feature_schema=schema)
        assert fp1 == fp2

    def test_no_schema_vs_with_schema_different_fingerprint(self):
        from harnessml.core.runner.training.fingerprint import compute_fingerprint

        config = {"type": "xgboost", "params": {"max_depth": 6}}
        schema = {"columns": ["feat_a"], "dtypes": {"feat_a": "double"}, "row_count": 50}

        fp_none = compute_fingerprint(model_config=config, feature_schema=None)
        fp_with = compute_fingerprint(model_config=config, feature_schema=schema)
        assert fp_none != fp_with
