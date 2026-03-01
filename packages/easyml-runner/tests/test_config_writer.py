"""Tests for config writer — pure functions for config mutations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.runner.config_writer import (
    _load_yaml,
    _save_yaml,
    add_model,
    available_features,
    configure_backtest,
    configure_ensemble,
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

    # Create matchup features parquet
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "season": rng.choice([2022, 2023, 2024], size=n),
        "result": rng.integers(0, 2, size=n),
        "diff_x": rng.standard_normal(n),
        "diff_y": rng.standard_normal(n),
    })
    df.to_parquet(feat_dir / "matchup_features.parquet", index=False)

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

    def test_remove_existing(self, tmp_path):
        project = _setup_project(tmp_path)
        result = remove_model(project, "logreg")
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
