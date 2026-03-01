"""Tests for PipelineRunner — wires library APIs from YAML config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.runner.pipeline import PipelineRunner


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _make_matchup_parquet(path: Path, n_rows: int = 200, n_seasons: int = 3) -> None:
    """Create a mock matchup_features.parquet with diff_x feature and season column."""
    rng = np.random.default_rng(42)
    seasons = rng.choice(list(range(2022, 2022 + n_seasons)), size=n_rows)
    df = pd.DataFrame({
        "season": seasons,
        "diff_x": rng.standard_normal(n_rows),
        "result": rng.integers(0, 2, size=n_rows),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _setup_project(tmp_path: Path, models: dict | None = None) -> Path:
    """Create a minimal project config and data for PipelineRunner tests.

    Returns the config directory path.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Pipeline config
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)
    _make_matchup_parquet(features_dir / "matchup_features.parquet")

    _write_yaml(
        config_dir / "pipeline.yaml",
        {
            "data": {
                "raw_dir": str(tmp_path / "data" / "raw"),
                "processed_dir": str(tmp_path / "data" / "processed"),
                "features_dir": str(features_dir),
            },
            "backtest": {
                "cv_strategy": "leave_one_season_out",
                "seasons": [2022, 2023, 2024],
                "metrics": ["brier", "accuracy"],
                "min_train_folds": 1,
            },
        },
    )

    if models is None:
        models = {
            "logreg_seed": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            }
        }
    _write_yaml(config_dir / "models.yaml", {"models": models})
    _write_yaml(config_dir / "ensemble.yaml", {"ensemble": {"method": "average"}})

    return config_dir


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestLoad:
    """PipelineRunner.load() validates config successfully."""

    def test_load(self, tmp_path):
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        assert runner.config is not None
        assert "logreg_seed" in runner.config.models


class TestTrain:
    """PipelineRunner.train() trains models."""

    def test_train(self, tmp_path):
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.train()
        assert result["status"] == "success"
        assert "logreg_seed" in result["models_trained"]


class TestBacktest:
    """PipelineRunner.backtest() returns metrics."""

    def test_backtest(self, tmp_path):
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()
        assert result["status"] == "success"
        assert "metrics" in result
        assert "brier" in result["metrics"]
        assert "accuracy" in result["metrics"]
        assert "per_fold" in result


class TestRunFull:
    """PipelineRunner.run_full() runs load + train + backtest."""

    def test_run_full(self, tmp_path):
        config_dir = _setup_project(tmp_path)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        result = runner.run_full()
        assert result["status"] == "success"
        assert "metrics" in result


class TestTwoModels:
    """PipelineRunner trains multiple models."""

    def test_two_models(self, tmp_path):
        models = {
            "logreg_seed": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "logreg_v2": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 500, "C": 0.5},
                "active": True,
            },
        }
        config_dir = _setup_project(tmp_path, models=models)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.train()
        assert result["status"] == "success"
        assert "logreg_seed" in result["models_trained"]
        assert "logreg_v2" in result["models_trained"]


class TestInactiveModelSkipped:
    """Inactive models are not trained."""

    def test_inactive_skipped(self, tmp_path):
        models = {
            "logreg_seed": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "logreg_inactive": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {},
                "active": False,
            },
        }
        config_dir = _setup_project(tmp_path, models=models)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.train()
        assert "logreg_seed" in result["models_trained"]
        assert "logreg_inactive" not in result["models_trained"]
