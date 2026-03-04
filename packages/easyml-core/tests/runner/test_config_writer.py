"""Tests for config writer — pure functions for config mutations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.core.runner.config_writer import (
    _load_yaml,
    _save_yaml,
    add_feature,
    add_model,
    available_features,
    configure_backtest,
    configure_denylist,
    configure_ensemble,
    configure_exclude_columns,
    experiment_create,
    remove_model,
    show_config,
    show_models,
    show_presets,
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
            "cv_strategy": "leave_one_season_out",
            "seasons": [2022, 2023, 2024],
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


class TestConfigureBacktest:
    """Test configure_backtest."""

    def test_update_seasons(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_backtest(project, seasons=[2022, 2023, 2024, 2025])
        assert "Updated" in result
        assert "2025" in result

    def test_update_cv_strategy(self, tmp_path):
        project = _setup_project(tmp_path)
        result = configure_backtest(project, cv_strategy="expanding_window")
        assert "expanding_window" in result


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
            "cv_strategy": "leave_one_season_out",
            "seasons": [2022, 2023, 2024],
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
            "type": "team",
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

    def test_add_team_feature_via_type(self, tmp_path):
        """add_feature(type='team') uses FeatureStore and mentions auto-pairwise."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "adj_em",
            type="team", source="kenpom", column="adj_em",
        )
        assert "Added team feature" in result
        assert "adj_em" in result
        # Should mention auto-generated pairwise
        assert "Auto-generated pairwise" in result
        assert "diff_adj_em" in result

    def test_add_team_feature_with_description(self, tmp_path):
        """Description is included in output."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "adj_em",
            type="team", source="kenpom", column="adj_em",
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

    def test_add_matchup_feature_via_type(self, tmp_path):
        """add_feature(type='matchup', column='...') works."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "day_feat",
            type="matchup", column="day_num",
        )
        assert "Added matchup feature" in result
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

    def test_add_team_feature_includes_stats(self, tmp_path):
        """Team feature output includes null rate, correlation, category."""
        project = _setup_declarative_project(tmp_path)
        result = add_feature(
            project, "adj_em",
            type="team", source="kenpom", column="adj_em",
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
        assert "Team" in result
        assert "adj_em" in result

    def test_type_filter(self, tmp_path):
        """type_filter limits to a single feature type."""
        project = _setup_declarative_project_with_defs(tmp_path)
        result = available_features(project, type_filter="regime")
        assert "Declarative Features" in result
        assert "late_season" in result
        # Should not include team features when filtering by regime
        assert "Team" not in result

    def test_prefix_filter_still_works(self, tmp_path):
        """Prefix filter still works on fallback column listing."""
        project = _setup_project(tmp_path)
        result = available_features(project, prefix="diff_")
        assert "diff_x" in result
        assert "season" not in result


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


def test_discover_features_respects_denylist(tmp_path):
    """Columns in the denylist should not appear in feature discovery results."""
    from easyml.core.runner.config_writer import discover_features
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
