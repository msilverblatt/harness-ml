"""End-to-end integration test: MCP-driven ML experimentation workflow.

Simulates the exact workflow an AI agent would use via tool calls:
1. Scaffold a project
2. Add a dataset
3. Discover features
4. Test transformations
5. Create a custom feature
6. Add a model with preset
7. Configure backtest
8. Run experiment + overlay
9. Show config / list models
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.runner import config_writer
from easyml.runner.data_ingest import (
    drop_duplicates,
    fill_nulls,
    ingest_dataset,
    rename_columns,
    validate_dataset,
)
from easyml.runner.data_utils import get_feature_columns, get_features_path, load_data_config
from easyml.runner.feature_discovery import (
    compute_feature_correlations,
    compute_feature_importance,
    detect_redundant_features,
    format_discovery_report,
    suggest_feature_groups,
    suggest_features,
)
from easyml.runner.feature_engine import create_feature, create_features_batch
from easyml.runner.pipeline_planner import plan_execution
from easyml.runner.presets import list_presets
from easyml.runner.scaffold import scaffold_project
from easyml.runner.schema import ProjectConfig
from easyml.runner.transformation_tester import run_transformation_tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(path: Path, *, n_rows: int = 200, n_seasons: int = 3) -> Path:
    """Create a synthetic matchup-style dataset."""
    rng = np.random.default_rng(42)
    seasons = list(range(2022, 2022 + n_seasons))

    rows = []
    for season in seasons:
        for _ in range(n_rows // n_seasons):
            rows.append({
                "Season": season,
                "team_id": rng.integers(1, 50),
                "opp_id": rng.integers(1, 50),
                "diff_adj_em": rng.normal(0, 10),
                "diff_barthag": rng.normal(0, 0.2),
                "diff_seed_num": rng.integers(-15, 16),
                "diff_tempo": rng.normal(0, 3),
                "diff_off_rtg": rng.normal(0, 8),
                "result": rng.integers(0, 2),
            })

    df = pd.DataFrame(rows)
    csv_path = path / "matchup_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _scaffold_with_data(tmp_path: Path) -> tuple[Path, Path]:
    """Scaffold a project and add initial data."""
    project_dir = tmp_path / "my_project"
    scaffold_project(project_dir, "test-project")

    # Create the features parquet
    features_dir = project_dir / "data" / "features"
    csv_path = _make_dataset(tmp_path)
    df = pd.read_csv(csv_path)
    df.to_parquet(features_dir / "features.parquet", index=False)

    return project_dir, csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMCPWorkflow:
    """End-to-end MCP-driven workflow tests."""

    def test_scaffold_creates_valid_project(self, tmp_path):
        """Step 0: Scaffold a project."""
        project_dir = tmp_path / "ml_project"
        scaffold_project(project_dir, "test-project")

        assert (project_dir / "config" / "pipeline.yaml").exists()
        assert (project_dir / "config" / "models.yaml").exists()
        assert (project_dir / "config" / "ensemble.yaml").exists()
        assert (project_dir / "config" / "server.yaml").exists()
        assert (project_dir / "config" / "sources.yaml").exists()
        assert (project_dir / "CLAUDE.md").exists()

        # Server.yaml includes new inspection tools
        import yaml

        server = yaml.safe_load(
            (project_dir / "config" / "server.yaml").read_text()
        )
        inspection = server["server"]["inspection"]
        assert "discover_features" in inspection
        assert "list_presets" in inspection
        assert "profile_data" in inspection

    def test_add_dataset_via_ingest(self, tmp_path):
        """Step 1: 'Here's my data' → ingest_dataset."""
        project_dir, csv_path = _scaffold_with_data(tmp_path)

        # Create a separate dataset to ingest
        rng = np.random.default_rng(99)
        extra = pd.DataFrame({
            "Season": [2022] * 30 + [2023] * 30 + [2024] * 30,
            "team_id": rng.integers(1, 50, size=90),
            "opp_id": rng.integers(1, 50, size=90),
            "coaching_rating": rng.normal(50, 10, size=90),
            "recruiting_score": rng.normal(70, 15, size=90),
        })
        extra_path = tmp_path / "coaching_data.csv"
        extra.to_csv(extra_path, index=False)

        features_dir = str(project_dir / "data" / "features")
        result = ingest_dataset(
            project_dir,
            str(extra_path),
            features_dir=features_dir,
        )

        assert result.rows_matched > 0
        assert "coaching_rating" in result.columns_added
        assert "recruiting_score" in result.columns_added
        summary = result.format_summary()
        assert "Ingested" in summary

    def test_discover_features(self, tmp_path):
        """Step 2: 'What features look useful?' → feature discovery."""
        project_dir, _ = _scaffold_with_data(tmp_path)
        features_dir = project_dir / "data" / "features"
        df = pd.read_parquet(features_dir / "features.parquet")

        correlations = compute_feature_correlations(df)
        importance = compute_feature_importance(df, method="mutual_info")
        redundant = detect_redundant_features(df)
        groups = suggest_feature_groups(df)

        assert len(correlations) > 0
        assert len(importance) > 0
        assert "feature" in correlations.columns

        report = format_discovery_report(correlations, importance, redundant, groups)
        assert "Feature Discovery" in report or "Correlation" in report

        # Also test suggest_features
        suggested = suggest_features(df, count=3, method="mutual_info")
        assert len(suggested) <= 3
        assert all(isinstance(f, str) for f in suggested)

    def test_transformation_testing(self, tmp_path):
        """Step 3: 'Test transformations on top features' → run_transformation_tests."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        report = run_transformation_tests(
            project_dir,
            features=["diff_adj_em", "diff_barthag"],
            features_dir=str(project_dir / "data" / "features"),
            test_interactions=False,
        )

        assert len(report.results) > 0
        assert "diff_adj_em" in report.best_per_feature

        summary = report.format_summary()
        assert "Transformation" in summary or "raw" in summary

        # get_create_commands returns dicts ready for batch creation
        commands = report.get_create_commands()
        assert isinstance(commands, list)

    def test_create_custom_feature(self, tmp_path):
        """Step 4: 'Create this custom feature' → create_feature with formula."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        result = create_feature(
            project_dir,
            name="em_barthag_product",
            formula="diff_adj_em * diff_barthag",
            features_dir=str(project_dir / "data" / "features"),
        )

        assert result.column_added == "em_barthag_product"
        assert isinstance(result.correlation, float)
        summary = result.format_summary()
        assert "em_barthag_product" in summary

        # Verify feature was persisted
        df = pd.read_parquet(
            project_dir / "data" / "features" / "features.parquet"
        )
        assert "em_barthag_product" in df.columns

    def test_batch_feature_creation_with_deps(self, tmp_path):
        """Step 4b: Batch create with @-references."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        features = [
            {"name": "adj_em_sq", "formula": "diff_adj_em ** 2"},
            {"name": "em_tempo", "formula": "@adj_em_sq * diff_tempo"},
        ]

        results = create_features_batch(
            project_dir,
            features,
            features_dir=str(project_dir / "data" / "features"),
        )

        assert len(results) == 2
        # First feature should be computed first due to dependency
        assert results[0].name == "adj_em_sq"
        assert results[1].name == "em_tempo"

    def test_add_model_with_preset(self, tmp_path):
        """Step 5: 'Add XGBoost with best features' → add_model with preset."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        # Verify presets are available
        presets = list_presets()
        assert len(presets) > 0
        assert "xgboost_classifier" in presets

        # Add model via config_writer
        result = config_writer.add_model(
            project_dir,
            "xgb_v1",
            preset="xgboost_classifier",
            features=["diff_adj_em", "diff_barthag", "diff_seed_num"],
        )

        assert "Added model" in result
        assert "xgb_v1" in result

        # Verify it shows up in models list
        models_md = config_writer.show_models(project_dir)
        assert "xgb_v1" in models_md

    def test_configure_backtest(self, tmp_path):
        """Step 6: Configure backtest seasons and metrics."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        result = config_writer.configure_backtest(
            project_dir,
            seasons=[2022, 2023, 2024],
            metrics=["brier", "accuracy", "log_loss"],
        )

        assert "Updated backtest config" in result
        assert "2022" in result

    def test_experiment_create_and_overlay(self, tmp_path):
        """Step 7: Create experiment + write overlay."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        # Create experiment
        result = config_writer.experiment_create(
            project_dir,
            description="Test higher learning rate",
            hypothesis="Higher LR converges faster with small data",
        )

        assert "Created experiment" in result
        assert "exp-" in result

        # Extract experiment ID from result
        exp_dir = project_dir / "experiments"
        exp_ids = sorted(d.name for d in exp_dir.iterdir() if d.is_dir())
        assert len(exp_ids) == 1

        # Write overlay
        overlay_result = config_writer.write_overlay(
            project_dir,
            exp_ids[0],
            {
                "models": {
                    "xgb_v1": {
                        "params": {"learning_rate": 0.2},
                    }
                }
            },
        )

        assert "Overlay written" in overlay_result

    def test_show_config_roundtrip(self, tmp_path):
        """Step 8: Show full config after modifications."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        # Add a model
        config_writer.add_model(
            project_dir,
            "lr_baseline",
            model_type="logistic_regression",
            features=["diff_adj_em"],
        )

        # Remove the scaffolded default
        config_writer.remove_model(project_dir, "logreg_baseline")

        # Show full config
        config_md = config_writer.show_config(project_dir)
        assert "lr_baseline" in config_md
        assert "logreg_baseline" not in config_md

    def test_pipeline_planner_detects_changes(self, tmp_path):
        """Step 9: plan_execution detects config diffs for smart re-runs."""
        from easyml.runner.schema import (
            BacktestConfig,
            DataConfig,
            EnsembleDef,
            ModelDef,
        )

        data = DataConfig(
            raw_dir="data/raw",
            processed_dir="data/processed",
            features_dir="data/features",
        )
        base_models = {
            "xgb": ModelDef(
                type="xgboost",
                features=["a", "b"],
                params={"lr": 0.1},
                active=True,
            ),
        }
        new_models = {
            "xgb": ModelDef(
                type="xgboost",
                features=["a", "b", "c"],
                params={"lr": 0.1},
                active=True,
            ),
        }

        current = ProjectConfig(
            data=data,
            models=base_models,
            ensemble=EnsembleDef(method="stacked"),
            backtest=BacktestConfig(cv_strategy="leave_one_season_out"),
        )
        new = ProjectConfig(
            data=data,
            models=new_models,
            ensemble=EnsembleDef(method="stacked"),
            backtest=BacktestConfig(cv_strategy="leave_one_season_out"),
        )

        plan = plan_execution(current, new)
        assert not plan.is_empty
        assert "xgb" in plan.models_to_retrain
        summary = plan.format_summary()
        assert "xgb" in summary

    def test_show_presets_markdown(self):
        """Verify presets display as formatted markdown."""
        result = config_writer.show_presets()
        assert "Presets" in result
        assert "xgboost" in result.lower()

    def test_configure_ensemble(self, tmp_path):
        """Configure ensemble settings."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        result = config_writer.configure_ensemble(
            project_dir,
            method="stacked",
            temperature=1.5,
        )

        assert "Updated ensemble config" in result
        assert "stacked" in result

    def test_full_workflow_sequence(self, tmp_path):
        """Full end-to-end sequence from scaffold to experiment.

        This is the integration test that verifies the complete
        ML experimentation workflow works without errors.
        """
        # 1. Scaffold
        project_dir = tmp_path / "full_flow"
        scaffold_project(project_dir, "full-flow-test")
        features_dir = project_dir / "data" / "features"

        # 2. Create initial data
        raw_data_dir = tmp_path / "raw_data"
        raw_data_dir.mkdir()
        csv_path = _make_dataset(raw_data_dir)
        df = pd.read_csv(csv_path)
        df.to_parquet(features_dir / "features.parquet", index=False)

        # 3. Discover features
        correlations = compute_feature_correlations(df)
        assert len(correlations) > 0
        top_features = suggest_features(df, count=3, method="mutual_info")
        assert len(top_features) > 0

        # 4. Test transformations
        report = run_transformation_tests(
            project_dir,
            features=top_features[:2],
            features_dir=str(features_dir),
            test_interactions=False,
        )
        assert len(report.results) > 0

        # 5. Create a custom feature
        result = create_feature(
            project_dir,
            name="custom_combo",
            formula="diff_adj_em * diff_barthag",
            features_dir=str(features_dir),
        )
        assert result.column_added == "custom_combo"

        # 6. Add model with preset
        config_writer.add_model(
            project_dir,
            "xgb_main",
            preset="xgboost_classifier",
            features=top_features + [result.column_added],
        )

        # 7. Configure backtest
        config_writer.configure_backtest(
            project_dir,
            seasons=[2022, 2023, 2024],
            metrics=["brier", "accuracy"],
        )

        # 8. Create experiment
        config_writer.experiment_create(
            project_dir,
            description="Baseline run with auto-discovered features",
            hypothesis="XGBoost with top features should beat logistic baseline",
        )

        # 9. Verify everything is readable
        config_md = config_writer.show_config(project_dir)
        assert "xgb_main" in config_md

        models_md = config_writer.show_models(project_dir)
        assert "xgb_main" in models_md


class TestScaffoldMLProblemDefinition:
    """Verify scaffold generates DataConfig with ML problem definition."""

    def test_scaffold_with_custom_target(self, tmp_path):
        project_dir = tmp_path / "churn_project"
        scaffold_project(
            project_dir,
            "churn-model",
            task="classification",
            target_column="churned",
            key_columns=["customer_id"],
            time_column="cohort_month",
        )

        import yaml
        pipeline = yaml.safe_load(
            (project_dir / "config" / "pipeline.yaml").read_text()
        )
        data = pipeline["data"]
        assert data["task"] == "classification"
        assert data["target_column"] == "churned"
        assert data["key_columns"] == ["customer_id"]
        assert data["time_column"] == "cohort_month"

    def test_scaffold_default_values(self, tmp_path):
        project_dir = tmp_path / "default_project"
        scaffold_project(project_dir, "default-test")

        import yaml
        pipeline = yaml.safe_load(
            (project_dir / "config" / "pipeline.yaml").read_text()
        )
        data = pipeline["data"]
        assert data["task"] == "classification"
        assert data["target_column"] == "result"


class TestBootstrapIngestion:
    """Verify bootstrap ingestion when no features parquet exists."""

    def test_first_dataset_creates_parquet(self, tmp_path):
        project_dir = tmp_path / "fresh_project"
        scaffold_project(project_dir, "fresh")
        features_dir = project_dir / "data" / "features"

        # Create a CSV dataset
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "customer_id": range(100),
            "revenue": rng.normal(500, 100, 100),
            "age": rng.integers(18, 80, 100),
            "churned": rng.integers(0, 2, 100),
        })
        csv_path = tmp_path / "customers.csv"
        df.to_csv(csv_path, index=False)

        # Ingest — should bootstrap (no existing parquet)
        result = ingest_dataset(
            project_dir,
            str(csv_path),
            features_dir=str(features_dir),
        )

        assert result.is_bootstrap
        assert result.rows_total == 100
        assert "customer_id" in result.columns_added
        assert "revenue" in result.columns_added

        # Verify parquet was created
        assert (features_dir / "features.parquet").exists()
        loaded = pd.read_parquet(features_dir / "features.parquet")
        assert len(loaded) == 100
        assert "revenue" in loaded.columns


class TestAutoCleanIngestion:
    """Verify auto-clean actions during ingestion."""

    def test_auto_clean_fills_nulls_and_drops_dupes(self, tmp_path):
        project_dir = tmp_path / "clean_project"
        scaffold_project(project_dir, "clean")
        features_dir = project_dir / "data" / "features"

        # Create data with nulls and duplicates
        df = pd.DataFrame({
            "id": [1, 2, 3, 3, 4, 5],
            "value": [10.0, np.nan, 30.0, 30.0, 40.0, np.nan],
            "target": [0, 1, 0, 0, 1, 1],
        })
        path = tmp_path / "dirty.parquet"
        df.to_parquet(path, index=False)

        result = ingest_dataset(
            project_dir,
            str(path),
            features_dir=str(features_dir),
            auto_clean=True,
        )

        assert result.is_bootstrap
        assert len(result.cleaning_actions) > 0

        # Check actions include null filling and/or dedup
        actions_text = " ".join(result.cleaning_actions)
        assert "null" in actions_text.lower() or "duplicate" in actions_text.lower()


class TestFeatureToolsGenericColumns:
    """Verify feature tools work with non-diff_ column names."""

    def test_create_feature_no_prefix(self, tmp_path):
        """Feature names should NOT be auto-prefixed with diff_."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        result = create_feature(
            project_dir,
            name="revenue_growth",
            formula="diff_adj_em * 2",
            features_dir=str(project_dir / "data" / "features"),
        )

        # Should be "revenue_growth", not "diff_revenue_growth"
        assert result.column_added == "revenue_growth"

        df = pd.read_parquet(
            project_dir / "data" / "features" / "features.parquet"
        )
        assert "revenue_growth" in df.columns
        assert "diff_revenue_growth" not in df.columns

    def test_feature_discovery_generic_columns(self, tmp_path):
        """Feature discovery should work on non-diff_ columns."""
        project_dir = tmp_path / "generic_project"
        scaffold_project(project_dir, "generic")
        features_dir = project_dir / "data" / "features"

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "user_age": rng.normal(35, 10, n),
            "market_vix": rng.normal(20, 5, n),
            "revenue": rng.normal(500, 100, n),
            "result": rng.integers(0, 2, n),
        })
        df.to_parquet(features_dir / "features.parquet", index=False)

        correlations = compute_feature_correlations(df)
        assert len(correlations) > 0
        # All three features should appear (not filtered by diff_ prefix)
        all_features = set(correlations["feature"].tolist())
        assert "user_age" in all_features
        assert "market_vix" in all_features
        assert "revenue" in all_features

    def test_get_feature_columns_by_exclusion(self, tmp_path):
        """get_feature_columns identifies features by excluding keys/target/time."""
        project_dir = tmp_path / "excl_project"
        scaffold_project(
            project_dir,
            "excl",
            target_column="churned",
            key_columns=["customer_id"],
            time_column="cohort_month",
        )

        config = load_data_config(project_dir)
        df = pd.DataFrame({
            "customer_id": [1, 2, 3],
            "cohort_month": [202301, 202302, 202303],
            "revenue": [100.0, 200.0, 300.0],
            "age": [25.0, 35.0, 45.0],
            "churned": [0, 1, 0],
        })

        features = get_feature_columns(df, config)
        assert "revenue" in features
        assert "age" in features
        assert "customer_id" not in features
        assert "cohort_month" not in features
        assert "churned" not in features


class TestGranularDataTools:
    """Test validate_dataset, fill_nulls, drop_duplicates, rename_columns."""

    def test_validate_dataset_reports_schema(self, tmp_path):
        project_dir = tmp_path / "val_project"
        scaffold_project(project_dir, "val")

        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": ["x", "y", "z"],
        })
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        result = validate_dataset(project_dir, str(csv_path))
        assert "Dataset Preview" in result
        assert "Rows" in result
        assert "Columns" in result

    def test_fill_nulls_and_drop_duplicates(self, tmp_path):
        project_dir = tmp_path / "granular_project"
        scaffold_project(project_dir, "granular")
        features_dir = project_dir / "data" / "features"

        # Create features with nulls and duplicates
        df = pd.DataFrame({
            "id": [1, 2, 3, 3],
            "score": [10.0, np.nan, 30.0, 30.0],
            "result": [0, 1, 0, 0],
        })
        df.to_parquet(features_dir / "features.parquet", index=False)

        # Fill nulls
        fill_result = fill_nulls(project_dir, "score", strategy="median")
        assert "Filled" in fill_result
        assert "1" in fill_result  # 1 null filled

        # Drop duplicates
        drop_result = drop_duplicates(project_dir)
        assert "Dropped" in drop_result or "No duplicates" in drop_result

    def test_rename_columns_persists(self, tmp_path):
        project_dir = tmp_path / "rename_project"
        scaffold_project(project_dir, "rename")
        features_dir = project_dir / "data" / "features"

        df = pd.DataFrame({
            "old_name": [1.0, 2.0, 3.0],
            "result": [0, 1, 0],
        })
        df.to_parquet(features_dir / "features.parquet", index=False)

        result = rename_columns(project_dir, {"old_name": "new_name"})
        assert "Renamed" in result

        loaded = pd.read_parquet(features_dir / "features.parquet")
        assert "new_name" in loaded.columns
        assert "old_name" not in loaded.columns


class TestInitAction:
    """Test configure(action='init') for MCP-driven project scaffolding."""

    def test_init_creates_project(self, tmp_path):
        """init action should scaffold a project and return confirmation."""
        result = config_writer.scaffold_init(
            project_dir=tmp_path / "new_project",
            project_name="test-init",
            task="classification",
            target_column="result",
            key_columns=["game_id"],
            time_column="season",
        )

        assert "Initialized project" in result
        assert "test-init" in result
        assert (tmp_path / "new_project" / "config" / "pipeline.yaml").exists()
        assert (tmp_path / "new_project" / "config" / "models.yaml").exists()

        import yaml
        pipeline = yaml.safe_load(
            (tmp_path / "new_project" / "config" / "pipeline.yaml").read_text()
        )
        assert pipeline["data"]["target_column"] == "result"
        assert pipeline["data"]["key_columns"] == ["game_id"]
        assert pipeline["data"]["time_column"] == "season"

    def test_init_rejects_existing_project(self, tmp_path):
        """init should error if directory already has content."""
        project_dir = tmp_path / "existing"
        project_dir.mkdir()
        (project_dir / "some_file.txt").write_text("content")

        result = config_writer.scaffold_init(
            project_dir=project_dir,
            project_name="existing",
        )
        assert "Error" in result


class TestPredictAction:
    """Test pipeline(action='predict') for MCP-driven predictions."""

    def test_predict_returns_formatted_results(self, tmp_path):
        """predict action should return prediction summary markdown."""
        project_dir, csv_path = _scaffold_with_data(tmp_path)

        # Add a simple model
        config_writer.add_model(
            project_dir, "lr_test",
            model_type="logistic_regression",
            features=["diff_adj_em", "diff_barthag", "diff_seed_num"],
        )
        config_writer.remove_model(project_dir, "logreg_baseline")
        config_writer.configure_backtest(
            project_dir, seasons=[2022, 2023, 2024],
        )

        result = config_writer.run_predict(project_dir, season=2024)

        assert "Prediction" in result or "predict" in result.lower()
        assert any(word in result.lower() for word in ["matchup", "prediction", "row"])

    def test_predict_no_data_for_season(self, tmp_path):
        """predict should handle missing season gracefully."""
        project_dir, _ = _scaffold_with_data(tmp_path)

        config_writer.add_model(
            project_dir, "lr_test",
            model_type="logistic_regression",
            features=["diff_adj_em", "diff_barthag"],
        )
        config_writer.remove_model(project_dir, "logreg_baseline")

        result = config_writer.run_predict(project_dir, season=2099)

        assert "Error" in result or "No" in result or "0" in result or "failed" in result.lower()


class TestTransformationTesterGeneric:
    """Verify transformation tester works with non-diff_ columns."""

    def test_interaction_partners_found(self, tmp_path):
        """Interaction partners should be found without diff_ filter."""
        project_dir = tmp_path / "transform_project"
        scaffold_project(project_dir, "transform")
        features_dir = project_dir / "data" / "features"

        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "revenue": rng.normal(500, 100, n),
            "cost": rng.normal(300, 50, n),
            "margin": rng.normal(200, 80, n),
            "result": rng.integers(0, 2, n),
        })
        df.to_parquet(features_dir / "features.parquet", index=False)

        report = run_transformation_tests(
            project_dir,
            features=["revenue"],
            features_dir=str(features_dir),
            test_interactions=True,
        )

        assert len(report.results) > 0


class TestGeneratePairwiseAction:
    """Test manage_data(action='generate_pairwise')."""

    def test_generate_pairwise_from_entity_data(self, tmp_path):
        """generate_pairwise should create pairwise diff features."""
        project_dir = tmp_path / "pairwise_project"
        scaffold_project(
            project_dir, "pairwise-test",
            target_column="result",
            key_columns=["period", "pair_id"],
            time_column="period",
        )
        features_dir = project_dir / "data" / "features"

        rng = np.random.default_rng(42)
        entities = list(range(1, 11))  # 10 entities
        periods = [2022, 2023]
        rows = []
        for period in periods:
            for entity_id in entities:
                rows.append({
                    "entity_id": entity_id,
                    "period": period,
                    "strength": rng.normal(0, 10),
                    "speed": rng.normal(68, 3),
                    "rating": rng.normal(100, 8),
                })
        entity_df = pd.DataFrame(rows)
        entity_df.to_parquet(features_dir / "features.parquet", index=False)

        result = config_writer.generate_pairwise_features(
            project_dir,
            entity_col="entity_id",
            group_col="period",
            feature_cols=["strength", "speed", "rating"],
        )

        assert "pairwise" in result.lower()
        # C(10,2) = 45 per period, 90 total
        assert "90" in result or "45" in result

        df = pd.read_parquet(features_dir / "features.parquet")
        assert len(df) == 90
        assert any(c.startswith("diff_") for c in df.columns)

    def test_generate_pairwise_with_filter(self, tmp_path):
        """generate_pairwise with filter_path should limit to filtered entities."""
        project_dir = tmp_path / "filtered_project"
        scaffold_project(project_dir, "filtered-test")
        features_dir = project_dir / "data" / "features"

        rng = np.random.default_rng(42)
        entity_df = pd.DataFrame({
            "entity_id": list(range(1, 11)),
            "period": [2022] * 10,
            "strength": rng.normal(0, 10, 10),
        })
        entity_df.to_parquet(features_dir / "features.parquet", index=False)

        # Filter: only 4 entities qualify
        filter_df = pd.DataFrame({
            "entity_id": [1, 2, 3, 4],
            "period": [2022] * 4,
        })
        filter_path = tmp_path / "qualified.csv"
        filter_df.to_csv(filter_path, index=False)

        result = config_writer.generate_pairwise_features(
            project_dir,
            entity_col="entity_id",
            group_col="period",
            feature_cols=["strength"],
            filter_path=str(filter_path),
        )

        assert "pairwise" in result.lower()
        df = pd.read_parquet(features_dir / "features.parquet")
        assert len(df) == 6  # C(4,2) = 6 pairs from 4 entities
