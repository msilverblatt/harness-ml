"""Tests for config writer — pure functions for config mutations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from harnessml.core.runner.config_writer import (
    _load_yaml,
    _save_yaml,
    add_feature,
    add_model,
    add_target,
    available_features,
    configure_backtest,
    configure_denylist,
    configure_ensemble,
    configure_exclude_columns,
    experiment_create,
    inspect_data,
    list_runs,
    list_targets,
    log_experiment_result,
    remove_model,
    set_active_target,
    show_config,
    show_journal,
    show_model,
    show_models,
    show_presets,
    update_data_config,
    update_model,
    write_overlay,
)


def _setup_project(tmp_path: Path) -> Path:
    """Create a minimal project structure for testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True)

    # pipeline.yaml
    pipeline = {
        "data": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "features_dir": str(feat_dir),
        },
        "backtest": {
            "cv_strategy": "leave_one_out",
            "fold_values": [2022, 2023, 2024],
            "metrics": ["brier", "accuracy"],
            "min_train_folds": 1,
        },
    }
    (config_dir / "pipeline.yaml").write_text(
        yaml.dump(pipeline, default_flow_style=False)
    )

    # models.yaml
    models = {
        "models": {
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        },
    }
    (config_dir / "models.yaml").write_text(
        yaml.dump(models, default_flow_style=False)
    )

    # ensemble.yaml
    (config_dir / "ensemble.yaml").write_text(
        yaml.dump({"ensemble": {"method": "average"}}, default_flow_style=False)
    )

    # sources.yaml (minimal — needed for denylist tests)
    (config_dir / "sources.yaml").write_text(
        yaml.dump({"guardrails": {}}, default_flow_style=False)
    )

    # Create matchup features parquet
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "season": rng.choice([2022, 2023, 2024], size=n),
        "result": rng.integers(0, 2, size=n),
        "diff_x": rng.standard_normal(n),
        "diff_y": rng.standard_normal(n),
    })
    df.to_parquet(feat_dir / "features.parquet", index=False)

    return tmp_path


class TestAddModel:
    """Test add_model."""

    def test_add_with_type(self, tmp_path):
        project = _setup_project(tmp_path)
        result = add_model(project, "xgb_new", model_type="xgboost",
                          features=["diff_x", "diff_y"])
        assert "Added model" in result
        assert "xgb_new" in result

        # Verify YAML updated
        data = _load_yaml(project / "config" / "models.yaml")
        assert "xgb_new" in data["models"]
        assert data["models"]["xgb_new"]["type"] == "xgboost"

    def test_add_with_preset(self, tmp_path):
        project = _setup_project(tmp_path)
        result = add_model(project, "xgb_preset", preset="xgboost_classifier",
                          features=["diff_x"])
        assert "Added model" in result
        assert "preset" in result

    def test_add_duplicate_returns_error(self, tmp_path):
        project = _setup_project(tmp_path)
        result = add_model(project, "logreg", model_type="logistic_regression")
        assert "Error" in result
        assert "already exists" in result

    def test_add_without_type_or_preset_returns_error(self, tmp_path):
        project = _setup_project(tmp_path)
        result = add_model(project, "new_model")
        assert "Error" in result

    def test_add_model_with_cdf_scale(self, tmp_path):
        project = _setup_project(tmp_path)
        result = add_model(project, "xgb_cdf", model_type="xgboost",
                          features=["diff_x"], cdf_scale=15.0)
        assert "Added model" in result

        data = _load_yaml(project / "config" / "models.yaml")
        assert data["models"]["xgb_cdf"]["cdf_scale"] == 15.0

    def test_add_model_with_zero_fill(self, tmp_path):
        project = _setup_project(tmp_path)
        result = add_model(project, "xgb_zf", model_type="xgboost",
                          features=["diff_x"], zero_fill_features=["feat_a"])
        assert "Added model" in result

        data = _load_yaml(project / "config" / "models.yaml")
        assert data["models"]["xgb_zf"]["zero_fill_features"] == ["feat_a"]

    def test_add_model_without_optional_params_omits_them(self, tmp_path):
        project = _setup_project(tmp_path)
        add_model(project, "xgb_plain", model_type="xgboost", features=["diff_x"])

        data = _load_yaml(project / "config" / "models.yaml")
        assert "cdf_scale" not in data["models"]["xgb_plain"]
        assert "zero_fill_features" not in data["models"]["xgb_plain"]


class TestUpdateModel:
    """Test update_model."""

    def test_update_model_cdf_scale(self, tmp_path):
        project = _setup_project(tmp_path)
        result = update_model(project, "logreg", cdf_scale=12.0)
        assert "Updated model" in result

        data = _load_yaml(project / "config" / "models.yaml")
        assert data["models"]["logreg"]["cdf_scale"] == 12.0

    def test_update_model_zero_fill(self, tmp_path):
        project = _setup_project(tmp_path)
        result = update_model(project, "logreg", zero_fill_features=["feat_a", "feat_b"])
        assert "Updated model" in result

        data = _load_yaml(project / "config" / "models.yaml")
        assert data["models"]["logreg"]["zero_fill_features"] == ["feat_a", "feat_b"]


class TestRemoveModel:
    """Test remove_model."""

    def test_remove_existing_soft_disables_by_default(self, tmp_path):
        project = _setup_project(tmp_path)
        result = remove_model(project, "logreg")
        assert "Disabled" in result

        data = _load_yaml(project / "config" / "models.yaml")
        assert "logreg" in data["models"], "model should still exist after soft-disable"
        assert data["models"]["logreg"]["active"] is False
        assert data["models"]["logreg"]["include_in_ensemble"] is False

    def test_remove_existing_purge_deletes(self, tmp_path):
        project = _setup_project(tmp_path)
        result = remove_model(project, "logreg", purge=True)
        assert "Removed" in result

        data = _load_yaml(project / "config" / "models.yaml")
        assert "logreg" not in data["models"]

    def test_remove_nonexistent(self, tmp_path):
        project = _setup_project(tmp_path)
        result = remove_model(project, "nonexistent")
        assert "Error" in result


class TestShowModels:
    """Test show_models."""

    def test_shows_all_models(self, tmp_path):
        project = _setup_project(tmp_path)
        result = show_models(project)
        assert "logreg" in result
        assert "logistic_regression" in result

    def test_empty_models(self, tmp_path):
        project = _setup_project(tmp_path)
        (project / "config" / "models.yaml").write_text(
            yaml.dump({"models": {}}, default_flow_style=False)
        )
        result = show_models(project)
        assert "No models" in result


class TestShowPresets:
    """Test show_presets."""

    def test_lists_presets(self):
        result = show_presets()
        assert "Presets" in result
        # Should include known presets
        assert "xgboost_classifier" in result
        assert "logistic_regression" in result


class TestConfigureEnsemble:
    """Test configure_ensemble."""

    def test_update_method(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_ensemble(project, method="stacked")
        assert "Updated" in result
        assert "stacked" in result

        data = _load_yaml(project / "config" / "ensemble.yaml")
        assert data["ensemble"]["method"] == "stacked"

    def test_update_temperature(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_ensemble(project, temperature=0.5)
        assert "Updated" in result

    def test_configure_ensemble_prior_feature(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_ensemble(project, prior_feature="seed_diff")
        assert "Updated" in result

        data = _load_yaml(project / "config" / "ensemble.yaml")
        assert data["ensemble"]["prior_feature"] == "seed_diff"

    def test_configure_ensemble_spline_params(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_ensemble(project, spline_prob_max=0.95, spline_n_bins=15)
        assert "Updated" in result

        data = _load_yaml(project / "config" / "ensemble.yaml")
        assert data["ensemble"]["spline_prob_max"] == 0.95
        assert data["ensemble"]["spline_n_bins"] == 15

    def test_configure_ensemble_without_optional_params_omits_them(self, tmp_path):
        project = _setup_project(tmp_path)
        configure_ensemble(project, method="stacked")

        data = _load_yaml(project / "config" / "ensemble.yaml")
        assert "prior_feature" not in data["ensemble"]
        assert "spline_prob_max" not in data["ensemble"]
        assert "spline_n_bins" not in data["ensemble"]


class TestConfigureBacktest:
    """Test configure_backtest."""

    def test_update_fold_values(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_backtest(project, fold_values=[2022, 2023, 2024, 2025])
        assert "Updated" in result
        assert "2025" in result

    def test_update_cv_strategy(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_backtest(project, cv_strategy="expanding_window")
        assert "expanding_window" in result

    def test_configure_backtest_fold_column(self, tmp_path):
        from harnessml.core.runner.config_writer import scaffold_init
        scaffold_init(tmp_path, "test_proj", task="binary")
        result = configure_backtest(tmp_path, fold_column="year")
        pipeline = yaml.safe_load((tmp_path / "config" / "pipeline.yaml").read_text())
        assert pipeline["backtest"]["fold_column"] == "year"
        assert "year" in result


class TestExperimentCreate:
    """Test experiment_create."""

    def test_creates_experiment(self, tmp_path):
        project = _setup_project(tmp_path)
        result = experiment_create(project, "Test hypothesis about coaching data")
        assert "Created experiment" in result
        assert "exp-" in result

        # Verify directory created
        exp_dirs = list((project / "experiments").iterdir())
        assert len(exp_dirs) == 1

    def test_creates_with_hypothesis(self, tmp_path):
        project = _setup_project(tmp_path)
        result = experiment_create(
            project,
            "Test coaching data",
            hypothesis="Adding coaching features will improve Brier by 0.002",
        )
        assert "Created experiment" in result

        exp_dirs = list((project / "experiments").iterdir())
        assert len(exp_dirs) == 1
        exp_dir = exp_dirs[0]
        assert (exp_dir / "hypothesis.txt").exists()
        assert "coaching" in (exp_dir / "hypothesis.txt").read_text()


class TestWriteOverlay:
    """Test write_overlay."""

    def test_writes_overlay(self, tmp_path):
        project = _setup_project(tmp_path)
        # Create experiment dir first
        exp_dir = project / "experiments" / "exp-001"
        exp_dir.mkdir(parents=True)

        overlay = {"models": {"logreg": {"params": {"C": 0.5}}}}
        result = write_overlay(project, "exp-001", overlay)
        assert "Overlay written" in result

        # Verify overlay file
        data = yaml.safe_load((exp_dir / "overlay.yaml").read_text())
        assert data["models"]["logreg"]["params"]["C"] == 0.5

    def test_nonexistent_experiment(self, tmp_path):
        project = _setup_project(tmp_path)
        result = write_overlay(project, "exp-999", {"foo": "bar"})
        assert "Error" in result


class TestShowConfig:
    """Test show_config."""

    def test_shows_config(self, tmp_path):
        project = _setup_project(tmp_path)
        result = show_config(project)
        assert "Project Configuration" in result
        assert "logreg" in result
        assert "average" in result


class TestAvailableFeatures:
    """Test available_features."""

    def test_lists_features(self, tmp_path):
        project = _setup_project(tmp_path)
        result = available_features(project)
        assert "diff_x" in result
        assert "diff_y" in result

    def test_prefix_filter(self, tmp_path):
        project = _setup_project(tmp_path)
        result = available_features(project, prefix="diff_")
        assert "diff_x" in result
        assert "season" not in result


class TestLoadSaveYaml:
    """Test _load_yaml and _save_yaml helpers."""

    def test_load_nonexistent(self, tmp_path):
        result = _load_yaml(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        result = _load_yaml(path)
        assert result == {}

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "test.yaml"
        data = {"key": "value", "nested": {"a": 1}}
        _save_yaml(path, data)
        loaded = _load_yaml(path)
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c.yaml"
        _save_yaml(path, {"x": 1})
        assert path.exists()
        assert _load_yaml(path) == {"x": 1}


# -----------------------------------------------------------------------
# Helper for declarative feature tests
# -----------------------------------------------------------------------


def _setup_declarative_project(tmp_path: Path) -> Path:
    """Create a project with entity-level and matchup-level data for declarative features."""
    project_dir = tmp_path / "decl_project"
    project_dir.mkdir()

    config_dir = project_dir / "config"
    config_dir.mkdir()

    # Entity-level data (team stats)
    rng = np.random.default_rng(42)
    n_teams = 20
    entities = []
    for season in [2022, 2023, 2024]:
        for team_id in range(1, n_teams + 1):
            entities.append({
                "entity_id": team_id,
                "period_id": season,
                "adj_em": rng.standard_normal() * 10,
                "adj_tempo": rng.standard_normal() * 5 + 65,
                "win_rate": rng.uniform(0.2, 0.9),
            })
    raw_dir = project_dir / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(entities).to_parquet(raw_dir / "kenpom.parquet", index=False)

    # Matchup-level data
    n_matchups = 200
    features_dir = project_dir / "data" / "features"
    features_dir.mkdir(parents=True)
    matchups = []
    for i in range(n_matchups):
        season = rng.choice([2022, 2023, 2024])
        team_a = rng.integers(1, n_teams + 1)
        team_b = rng.integers(1, n_teams + 1)
        while team_b == team_a:
            team_b = rng.integers(1, n_teams + 1)
        matchups.append({
            "entity_a_id": int(team_a),
            "entity_b_id": int(team_b),
            "period_id": int(season),
            "result": int(rng.integers(0, 2)),
            "day_num": int(rng.integers(1, 155)),
            "is_neutral": int(rng.integers(0, 2)),
        })
    pd.DataFrame(matchups).to_parquet(features_dir / "features.parquet", index=False)

    # pipeline.yaml with sources configured
    pipeline = {
        "data": {
            "features_dir": "data/features",
            "features_file": "features.parquet",
            "target_column": "result",
            "raw_dir": "data/raw",
            "sources": {
                "kenpom": {
                    "name": "kenpom",
                    "path": "data/raw/kenpom.parquet",
                    "format": "parquet",
                },
            },
            "feature_store": {
                "cache_dir": "data/features/cache",
                "auto_pairwise": True,
                "default_pairwise_mode": "diff",
            },
        },
        "backtest": {
            "cv_strategy": "leave_one_out",
            "fold_values": [2022, 2023, 2024],
            "metrics": ["brier"],
            "min_train_folds": 1,
        },
    }
    (config_dir / "pipeline.yaml").write_text(
        yaml.dump(pipeline, default_flow_style=False)
    )

    # models.yaml (minimal)
    (config_dir / "models.yaml").write_text(
        yaml.dump({"models": {}}, default_flow_style=False)
    )
    # ensemble.yaml (minimal)
    (config_dir / "ensemble.yaml").write_text(
        yaml.dump({"ensemble": {"method": "average"}}, default_flow_style=False)
    )

    return project_dir


def _setup_declarative_project_with_defs(tmp_path: Path) -> Path:
    """Create a project that already has feature_defs in pipeline.yaml."""
    project_dir = _setup_declarative_project(tmp_path)

    # Re-write pipeline.yaml with feature_defs included
    pipeline_path = project_dir / "config" / "pipeline.yaml"
    pipeline = yaml.safe_load(pipeline_path.read_text())
    pipeline["data"]["feature_defs"] = {
        "adj_em": {
            "name": "adj_em",
            "type": "entity",
            "source": "kenpom",
            "column": "adj_em",
            "category": "efficiency",
            "description": "Adjusted efficiency margin",
        },
        "late_season": {
            "name": "late_season",
            "type": "regime",
            "condition": "day_num > 100",
            "category": "temporal",
            "description": "Late season flag",
        },
    }
    pipeline_path.write_text(yaml.dump(pipeline, default_flow_style=False))

    return project_dir


# -----------------------------------------------------------------------
# add_feature MCP tool tests
# -----------------------------------------------------------------------


class TestAddFeatureDeclarative:
    """Test add_feature with declarative type= parameter (FeatureStore path)."""

    def test_add_entity_feature_via_type(self, tmp_path):
        """add_feature(type='entity') uses FeatureStore and mentions auto-pairwise."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "adj_em",
            type="entity", source="kenpom", column="adj_em",
        )
        assert "Added entity feature" in result
        assert "adj_em" in result
        # Should mention auto-generated pairwise
        assert "Auto-generated pairwise" in result
        assert "diff_adj_em" in result

    def test_add_entity_feature_with_description(self, tmp_path):
        """Description is included in output."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "adj_em",
            type="entity", source="kenpom", column="adj_em",
            description="Adjusted efficiency margin",
        )
        assert "Adjusted efficiency margin" in result

    def test_add_regime_feature_via_type(self, tmp_path):
        """add_feature(type='regime', condition='...') works."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "late_season",
            type="regime", condition="day_num > 100",
        )
        assert "Added regime feature" in result
        assert "late_season" in result
        assert "Correlation" in result

    def test_add_instance_feature_via_type(self, tmp_path):
        """add_feature(type='instance', column='...') works."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "day_feat",
            type="instance", column="day_num",
        )
        assert "Added instance feature" in result
        assert "Correlation" in result

    def test_add_pairwise_formula_feature(self, tmp_path):
        """add_feature(type='pairwise', formula='...') works."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "day_x_neutral",
            type="pairwise", formula="day_num * is_neutral",
        )
        assert "Added pairwise feature" in result
        assert "Correlation" in result

    def test_add_entity_feature_includes_stats(self, tmp_path):
        """Entity feature output includes null rate, correlation, category."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "adj_em",
            type="entity", source="kenpom", column="adj_em",
            category="efficiency",
        )
        assert "Null rate" in result
        assert "Category" in result
        assert "efficiency" in result

    def test_no_type_no_formula_raises(self, tmp_path):
        """Calling add_feature without type or formula raises ValueError."""
        project = _setup_declarative_project(tmp_path)
        with pytest.raises(ValueError, match="Must provide type, formula, condition, or source"):
            add_feature(project, "bad_feature")


class TestAddFeatureBackwardCompat:
    """Test add_feature backward compatibility — formula-only path.

    When no type= is specified but formula is given, type defaults to pairwise
    and the feature is created via the FeatureStore.
    """

    def test_formula_only_infers_pairwise(self, tmp_path):
        """add_feature(name, formula) infers type=pairwise."""
        project = _setup_project(tmp_path)
        result = add_feature(project, "combo", formula="diff_x * diff_y")
        assert "combo" in result
        assert "pairwise" in result.lower()
        assert "Correlation" in result

    def test_formula_with_description(self, tmp_path):
        """Formula path includes description."""
        project = _setup_project(tmp_path)
        result = add_feature(
            project, "combo",
            formula="diff_x + diff_y",
            description="Sum of diffs",
        )
        assert "Sum of diffs" in result

    def test_formula_persists_to_yaml(self, tmp_path):
        """Formula-only features are persisted to pipeline.yaml as feature_defs."""
        project = _setup_project(tmp_path)
        add_feature(project, "combo", formula="diff_x * diff_y")

        pipeline = yaml.safe_load(
            (project / "config" / "pipeline.yaml").read_text()
        )
        feature_defs = pipeline.get("data", {}).get("feature_defs", {})
        assert "combo" in feature_defs
        assert feature_defs["combo"]["type"] == "pairwise"
        assert feature_defs["combo"]["formula"] == "diff_x * diff_y"


# -----------------------------------------------------------------------
# available_features MCP tool tests
# -----------------------------------------------------------------------


class TestAvailableFeaturesDeclarative:
    """Test available_features with declarative feature store."""

    def test_fallback_column_listing(self, tmp_path):
        """When no feature_defs exist, falls back to column listing."""
        project = _setup_project(tmp_path)
        result = available_features(project)
        assert "Available Features" in result
        assert "diff_x" in result
        assert "diff_y" in result

    def test_shows_types_with_feature_defs(self, tmp_path):
        """When feature_defs exist, shows features grouped by type."""
        project = _setup_declarative_project_with_defs(tmp_path)
        result = available_features(project)
        assert "Declarative Features" in result
        assert "Entity" in result
        assert "adj_em" in result

    def test_type_filter(self, tmp_path):
        """type_filter limits to a single feature type."""
        project = _setup_declarative_project_with_defs(tmp_path)
        result = available_features(project, type_filter="regime")
        assert "Declarative Features" in result
        assert "late_season" in result
        # Should not include entity features when filtering by regime
        assert "Entity" not in result

    def test_prefix_filter_still_works(self, tmp_path):
        """Prefix filter still works on fallback column listing."""
        project = _setup_project(tmp_path)
        result = available_features(project, prefix="diff_")
        assert "diff_x" in result
        assert "season" not in result


class TestUpdateDataConfig:
    """Test update_data_config."""

    def test_update_data_config(self, tmp_path):
        """update_data_config should update target_column, key_columns, time_column in pipeline.yaml."""
        from harnessml.core.runner.config_writer import scaffold_init

        scaffold_init(tmp_path, "test_proj", task="binary", target_column="result")

        result = update_data_config(
            tmp_path,
            target_column="outcome",
            key_columns=["id", "date"],
            time_column="year",
        )

        pipeline = yaml.safe_load((tmp_path / "config" / "pipeline.yaml").read_text())
        assert pipeline["data"]["target_column"] == "outcome"
        assert pipeline["data"]["key_columns"] == ["id", "date"]
        assert pipeline["data"]["time_column"] == "year"
        assert "outcome" in result

    def test_update_data_config_partial(self, tmp_path):
        """update_data_config with only target_column should not touch other fields."""
        from harnessml.core.runner.config_writer import scaffold_init

        scaffold_init(tmp_path, "test_proj", task="binary", target_column="result",
                      key_columns=["a", "b"], time_column="season")

        update_data_config(tmp_path, target_column="label")

        pipeline = yaml.safe_load((tmp_path / "config" / "pipeline.yaml").read_text())
        assert pipeline["data"]["target_column"] == "label"
        assert pipeline["data"]["key_columns"] == ["a", "b"]
        assert pipeline["data"]["time_column"] == "season"

    def test_update_data_config_no_changes(self, tmp_path):
        """update_data_config with no arguments returns no-change message."""
        project = _setup_project(tmp_path)
        result = update_data_config(project)
        assert "No changes" in result


class TestConfigureExcludeColumns:
    """Test configure_exclude_columns."""

    def test_add_columns(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_exclude_columns(project, add_columns=["margin", "actual_score"])
        assert "margin" in result
        data = yaml.safe_load((project / "config" / "pipeline.yaml").read_text())
        assert "margin" in data["data"]["exclude_columns"]
        assert "actual_score" in data["data"]["exclude_columns"]

    def test_remove_columns(self, tmp_path):
        project = _setup_project(tmp_path)
        configure_exclude_columns(project, add_columns=["margin"])
        configure_exclude_columns(project, remove_columns=["margin"])
        data = yaml.safe_load((project / "config" / "pipeline.yaml").read_text())
        assert "margin" not in data["data"].get("exclude_columns", [])

    def test_no_duplicates(self, tmp_path):
        project = _setup_project(tmp_path)
        configure_exclude_columns(project, add_columns=["margin"])
        configure_exclude_columns(project, add_columns=["margin"])
        data = yaml.safe_load((project / "config" / "pipeline.yaml").read_text())
        assert data["data"]["exclude_columns"].count("margin") == 1


class TestConfigureDenylist:
    """Test configure_denylist."""

    def test_add_columns(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_denylist(project, add_columns=["leaked_col"])
        assert "leaked_col" in result
        data = yaml.safe_load((project / "config" / "sources.yaml").read_text())
        assert "leaked_col" in data["guardrails"]["feature_leakage_denylist"]

    def test_remove_columns(self, tmp_path):
        project = _setup_project(tmp_path)
        configure_denylist(project, add_columns=["leaked_col"])
        configure_denylist(project, remove_columns=["leaked_col"])
        data = yaml.safe_load((project / "config" / "sources.yaml").read_text())
        assert "leaked_col" not in data["guardrails"].get("feature_leakage_denylist", [])

    def test_no_duplicates(self, tmp_path):
        project = _setup_project(tmp_path)
        configure_denylist(project, add_columns=["leaked_col"])
        configure_denylist(project, add_columns=["leaked_col"])
        data = yaml.safe_load((project / "config" / "sources.yaml").read_text())
        assert data["guardrails"]["feature_leakage_denylist"].count("leaked_col") == 1


def test_discover_features_on_progress(tmp_path):
    """on_progress callback is invoked with correlation and importance messages."""
    from harnessml.core.runner.config_writer import discover_features

    project = _setup_project(tmp_path)
    # Add target_column to pipeline so discovery can compute correlations/importance
    pipeline_path = project / "config" / "pipeline.yaml"
    pipeline = yaml.safe_load(pipeline_path.read_text())
    pipeline["data"]["target_column"] = "result"
    pipeline_path.write_text(yaml.dump(pipeline, default_flow_style=False))

    messages = []

    def progress_cb(step, total, msg):
        messages.append(msg)

    discover_features(project, on_progress=progress_cb)

    assert len(messages) >= 3
    assert any("correlation" in m.lower() for m in messages)
    assert any("importance" in m.lower() for m in messages)


def test_discover_features_respects_denylist(tmp_path):
    """Columns in the denylist should not appear in feature discovery results."""
    from harnessml.core.runner.config_writer import discover_features
    project = _setup_project(tmp_path)
    # Add a leaky column to the feature parquet
    feat_path = project / "data" / "features" / "features.parquet"
    df = pd.read_parquet(feat_path)
    df["leaky_col"] = np.random.rand(len(df))
    df.to_parquet(feat_path, index=False)
    # Add leaky_col to denylist
    configure_denylist(project, add_columns=["leaky_col"])
    # Run discovery — denylist column should not appear
    result = discover_features(project)
    assert "leaky_col" not in result


# -----------------------------------------------------------------------
# add_feature redundancy warning tests
# -----------------------------------------------------------------------


class TestAddFeatureRedundancyWarning:
    """Test that adding a feature with an identical formula warns the user."""

    def test_add_feature_redundancy_warning(self, tmp_path):
        """Adding a feature with identical formula to existing should include a warning."""
        project = _setup_project(tmp_path)

        # Add first feature
        add_feature(project, "momentum_10d", formula="diff_x * diff_y")

        # Add feature with same formula — should warn
        result = add_feature(project, "return_10d", formula="diff_x * diff_y")
        assert "warning" in result.lower() or "identical" in result.lower()

    def test_add_feature_no_warning_different_formula(self, tmp_path):
        """Adding a feature with different formula should not warn."""
        project = _setup_project(tmp_path)

        add_feature(project, "feat_a", formula="diff_x + diff_y")
        result = add_feature(project, "feat_b", formula="diff_x - diff_y")
        assert "warning" not in result.lower() and "identical" not in result.lower()


def _setup_inspect_project(tmp_path: Path) -> Path:
    """Create a project with a features parquet suitable for inspect tests."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True)

    pipeline = {
        "data": {
            "target_column": "target",
            "features_dir": str(feat_dir),
        },
    }
    (config_dir / "pipeline.yaml").write_text(
        yaml.dump(pipeline, default_flow_style=False)
    )

    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, float("nan"), 5.0],
        "feat_b": ["cat", "dog", "cat", "bird", "dog"],
        "target": [0, 1, 1, 0, 1],
    })
    df.to_parquet(feat_dir / "features.parquet", index=False)
    return tmp_path


class TestInspectData:
    """Tests for inspect_data."""

    def test_inspect_all(self, tmp_path):
        project = _setup_inspect_project(tmp_path)
        result = inspect_data(project)
        assert "5" in result  # 5 rows
        assert "3 columns" in result
        assert "feat_a" in result
        assert "feat_b" in result
        assert "target" in result
        # feat_a has 1 null
        assert "1" in result
        assert "20.0%" in result

    def test_inspect_specific_column_numeric(self, tmp_path):
        project = _setup_inspect_project(tmp_path)
        result = inspect_data(project, column="feat_a")
        assert "feat_a" in result
        assert "Mean" in result
        assert "Std" in result
        assert "Min" in result
        assert "25%" in result
        assert "50%" in result
        assert "75%" in result
        assert "Max" in result

    def test_inspect_specific_column_categorical(self, tmp_path):
        project = _setup_inspect_project(tmp_path)
        result = inspect_data(project, column="target")
        assert "Value Counts" in result
        assert "0" in result
        assert "1" in result

    def test_inspect_unknown_column(self, tmp_path):
        project = _setup_inspect_project(tmp_path)
        result = inspect_data(project, column="nonexistent")
        assert "Error" in result
        assert "nonexistent" in result
        assert "feat_a" in result  # available columns listed


# -----------------------------------------------------------------------
# Target profiles
# -----------------------------------------------------------------------

class TestTargetProfiles:
    """Tests for add_target, list_targets, set_active_target."""

    def test_add_target(self, tmp_path):
        project = _setup_project(tmp_path)
        result = add_target(project, "winner", column="result", task="binary", metrics=["brier", "accuracy"])
        assert "winner" in result

        data = _load_yaml(tmp_path / "config" / "pipeline.yaml")
        target = data["data"]["targets"]["winner"]
        assert target["column"] == "result"
        assert target["task"] == "binary"
        assert target["metrics"] == ["brier", "accuracy"]

    def test_add_multiple_targets(self, tmp_path):
        project = _setup_project(tmp_path)
        add_target(project, "winner", column="result", task="binary")
        add_target(project, "spread", column="margin", task="regression", metrics=["rmse"])

        data = _load_yaml(tmp_path / "config" / "pipeline.yaml")
        assert "winner" in data["data"]["targets"]
        assert "spread" in data["data"]["targets"]
        assert data["data"]["targets"]["spread"]["column"] == "margin"
        assert data["data"]["targets"]["spread"]["task"] == "regression"

    def test_list_targets(self, tmp_path):
        project = _setup_project(tmp_path)
        add_target(project, "winner", column="result", task="binary", metrics=["brier"])
        add_target(project, "spread", column="margin", task="regression")

        result = list_targets(project)
        assert "winner" in result
        assert "spread" in result
        assert "result" in result
        assert "margin" in result

    def test_set_active_target(self, tmp_path):
        project = _setup_project(tmp_path)
        add_target(project, "spread", column="margin", task="regression", metrics=["rmse", "mae"])
        result = set_active_target(project, "spread")
        assert "spread" in result

        data = _load_yaml(tmp_path / "config" / "pipeline.yaml")
        assert data["data"]["target_column"] == "margin"
        assert data["data"]["task"] == "regression"
        assert data["backtest"]["metrics"] == ["rmse", "mae"]

    def test_set_active_target_unknown(self, tmp_path):
        project = _setup_project(tmp_path)
        result = set_active_target(project, "nonexistent")
        assert "Error" in result
        assert "nonexistent" in result


class TestListRuns:
    def test_list_runs_with_metrics(self, tmp_path):
        """list_runs should include metrics when available."""
        import json

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text("data:\n  outputs_dir: outputs\n")

        for run_id, brier, acc in [("20260301_100000", 0.25, 0.55), ("20260302_100000", 0.22, 0.60)]:
            run_dir = tmp_path / "outputs" / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "pooled_metrics.json").write_text(
                json.dumps({"brier": brier, "accuracy": acc})
            )

        result = list_runs(tmp_path)
        assert "0.2500" in result  # brier formatted to 4 decimal places
        assert "0.6000" in result  # accuracy formatted
        assert "20260301_100000" in result
        assert "20260302_100000" in result
        assert "Brier" in result  # metric names title-cased in header
        assert "Accuracy" in result

    def test_list_runs_without_metrics(self, tmp_path):
        """list_runs falls back to bullet list when no metrics files exist."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text("data:\n  outputs_dir: outputs\n")

        run_dir = tmp_path / "outputs" / "20260301_100000"
        run_dir.mkdir(parents=True)

        result = list_runs(tmp_path)
        assert "20260301_100000" in result
        assert "|" not in result  # no table formatting


class TestShowModel:
    def test_show_model(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        models_data = {
            "models": {
                "xgb_core": {
                    "type": "xgboost",
                    "active": True,
                    "include_in_ensemble": True,
                    "features": ["feat_a", "feat_b", "feat_c"],
                    "params": {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.05},
                }
            }
        }
        (config_dir / "models.yaml").write_text(yaml.dump(models_data))

        result = show_model(tmp_path, "xgb_core")
        assert "xgb_core" in result
        assert "xgboost" in result
        assert "feat_a" in result
        assert "feat_b" in result
        assert "n_estimators" in result
        assert "200" in result

    def test_show_model_not_found(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "models.yaml").write_text(yaml.dump({"models": {}}))
        result = show_model(tmp_path, "nonexistent")
        assert "Error" in result

    def test_show_model_extra_keys(self, tmp_path):
        """Extra config keys beyond standard ones are shown in Other Settings."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        models_data = {
            "models": {
                "lgb_test": {
                    "type": "lightgbm",
                    "features": ["f1"],
                    "params": {},
                    "cdf_scale": 1.5,
                }
            }
        }
        (config_dir / "models.yaml").write_text(yaml.dump(models_data))

        result = show_model(tmp_path, "lgb_test")
        assert "cdf_scale" in result
        assert "1.5" in result


class TestExperimentJournal:
    def test_log_experiment_result(self, tmp_path):
        import json

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text("data:\n  target_column: result\n")

        log_experiment_result(
            tmp_path,
            experiment_id="exp-001",
            description="Test longer horizon",
            hypothesis="20-day target will have more signal",
            metrics={"brier": 0.2481, "accuracy": 0.5308, "auc_roc": 0.5132},
            verdict="improved",
        )

        journal_path = tmp_path / "experiments" / "journal.jsonl"
        assert journal_path.exists()
        entry = json.loads(journal_path.read_text().strip())
        assert entry["experiment_id"] == "exp-001"
        assert entry["metrics"]["brier"] == 0.2481
        assert entry["verdict"] == "improved"

    def test_show_journal(self, tmp_path):
        (tmp_path / "experiments").mkdir(parents=True, exist_ok=True)

        log_experiment_result(tmp_path, experiment_id="exp-001",
            description="Baseline", metrics={"brier": 0.25, "accuracy": 0.53}, verdict="baseline")
        log_experiment_result(tmp_path, experiment_id="exp-002",
            description="Add macro features", metrics={"brier": 0.24, "accuracy": 0.55}, verdict="improved")

        result = show_journal(tmp_path)
        assert "exp-001" in result
        assert "exp-002" in result
        assert "improved" in result

    def test_show_journal_empty(self, tmp_path):
        result = show_journal(tmp_path)
        assert "No experiment journal" in result

    def test_show_journal_last_n(self, tmp_path):
        for i in range(5):
            log_experiment_result(tmp_path, experiment_id=f"exp-{i:03d}",
                description=f"Test {i}", metrics={"brier": 0.25 - i * 0.01})

        result = show_journal(tmp_path, last_n=2)
        assert "exp-003" in result
        assert "exp-004" in result
        assert "exp-000" not in result


class TestFormatTargetComparison:
    def test_format_target_comparison(self):
        from harnessml.core.runner.config_writer import format_target_comparison

        results = {
            "short": {"brier": 0.2497, "accuracy": 0.5389, "auc_roc": 0.4686},
            "medium": {"brier": 0.2481, "accuracy": 0.5308, "auc_roc": 0.5132},
            "long": {"brier": 0.2300, "accuracy": 0.5800, "auc_roc": 0.5500},
        }
        output = format_target_comparison(results)
        assert "short" in output
        assert "medium" in output
        assert "long" in output
        assert "0.5132" in output
        assert "Best per metric" in output
        # brier is lower-better, so "long" should be best
        assert "**long**" in output

    def test_format_target_comparison_empty(self):
        from harnessml.core.runner.config_writer import format_target_comparison
        assert "No results" in format_target_comparison({})

    def test_format_target_comparison_with_error(self):
        from harnessml.core.runner.config_writer import format_target_comparison
        results = {
            "short": {"brier": 0.25, "accuracy": 0.55},
            "broken": {"error": "something failed"},
        }
        output = format_target_comparison(results)
        assert "short" in output
        assert "broken" in output


class TestFetchUrl:
    def test_fetch_url(self, tmp_path, monkeypatch):
        from harnessml.core.runner.config_writer import fetch_url

        # Mock urllib.request.urlretrieve to just write a file
        def mock_urlretrieve(url, dest):
            Path(dest).write_text("a,b\n1,2\n3,4\n")

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlretrieve", mock_urlretrieve)

        result = fetch_url(tmp_path, "https://example.com/data.csv")
        assert (tmp_path / "data" / "raw" / "data.csv").exists()
        assert "data.csv" in result
        assert "Downloaded" in result

    def test_fetch_url_custom_filename(self, tmp_path, monkeypatch):
        from harnessml.core.runner.config_writer import fetch_url

        def mock_urlretrieve(url, dest):
            Path(dest).write_text("test data")

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlretrieve", mock_urlretrieve)

        result = fetch_url(tmp_path, "https://example.com/some/path", filename="custom.csv")
        assert (tmp_path / "data" / "raw" / "custom.csv").exists()
        assert "custom.csv" in result

    def test_fetch_url_auto_filename(self, tmp_path, monkeypatch):
        from harnessml.core.runner.config_writer import fetch_url

        def mock_urlretrieve(url, dest):
            Path(dest).write_text("test data")

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlretrieve", mock_urlretrieve)

        result = fetch_url(tmp_path, "https://example.com/datasets/sp500_data.parquet")
        assert (tmp_path / "data" / "raw" / "sp500_data.parquet").exists()


class TestInspectPredictions:
    def test_inspect_predictions_worst(self, tmp_path):
        from harnessml.core.runner.config_writer import inspect_predictions

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text(
            "data:\n  target_column: target\n  outputs_dir: outputs\n  key_columns: [id]\n"
        )

        run_dir = tmp_path / "outputs" / "20260306_100000" / "predictions"
        run_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "id": ["a", "b", "c", "d", "e"],
            "target": [0, 1, 0, 1, 0],
            "ensemble_prob": [0.9, 0.1, 0.2, 0.8, 0.95],
        })
        df.to_parquet(run_dir / "fold_1.parquet", index=False)

        result = inspect_predictions(tmp_path, mode="worst", top_n=3)
        # Most confident wrong: id=e (0.95 predicted 1, actual 0),
        # id=a (0.9 predicted 1, actual 0), id=b (0.1 predicted 0, actual 1)
        assert "0.9500" in result or "0.95" in result
        assert "Confident Wrong" in result

    def test_inspect_predictions_best(self, tmp_path):
        from harnessml.core.runner.config_writer import inspect_predictions

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text(
            "data:\n  target_column: target\n  outputs_dir: outputs\n"
        )

        run_dir = tmp_path / "outputs" / "20260306_100000" / "predictions"
        run_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "target": [1, 0, 1],
            "ensemble_prob": [0.9, 0.1, 0.6],
        })
        df.to_parquet(run_dir / "fold_1.parquet", index=False)

        result = inspect_predictions(tmp_path, mode="best", top_n=2)
        assert "Confident Correct" in result

    def test_inspect_predictions_uncertain(self, tmp_path):
        from harnessml.core.runner.config_writer import inspect_predictions

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text(
            "data:\n  target_column: target\n  outputs_dir: outputs\n"
        )

        run_dir = tmp_path / "outputs" / "20260306_100000" / "predictions"
        run_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "target": [1, 0, 1],
            "ensemble_prob": [0.51, 0.1, 0.9],
        })
        df.to_parquet(run_dir / "fold_1.parquet", index=False)

        result = inspect_predictions(tmp_path, mode="uncertain", top_n=1)
        assert "Uncertain" in result
        assert "0.5100" in result or "0.51" in result

    def test_inspect_predictions_no_runs(self, tmp_path):
        from harnessml.core.runner.config_writer import inspect_predictions

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text(
            "data:\n  target_column: target\n  outputs_dir: outputs\n"
        )
        (tmp_path / "outputs").mkdir()
        result = inspect_predictions(tmp_path)
        assert "Error" in result
