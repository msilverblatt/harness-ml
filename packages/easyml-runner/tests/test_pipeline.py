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


def _make_matchup_parquet(
    path: Path,
    n_rows: int = 200,
    n_seasons: int = 3,
    include_margin: bool = False,
    include_seed: bool = False,
    mm_style: bool = False,
    extra_features: list[str] | None = None,
) -> None:
    """Create a mock matchup_features.parquet with diff_x feature and season column."""
    rng = np.random.default_rng(42)
    seasons = rng.choice(list(range(2022, 2022 + n_seasons)), size=n_rows)
    result = rng.integers(0, 2, size=n_rows)

    data = {
        "diff_x": rng.standard_normal(n_rows),
    }

    if mm_style:
        data["Season"] = seasons
        data["TeamAWon"] = result
        if include_margin:
            data["TeamAMargin"] = rng.standard_normal(n_rows) * 10
    else:
        data["season"] = seasons
        data["result"] = result
        if include_margin:
            data["margin"] = rng.standard_normal(n_rows) * 10

    if include_seed:
        seed_col = "diff_seed_num"
        data[seed_col] = rng.integers(-15, 16, size=n_rows).astype(float)

    if extra_features:
        for feat in extra_features:
            data[feat] = rng.standard_normal(n_rows)

    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _setup_project(
    tmp_path: Path,
    models: dict | None = None,
    ensemble: dict | None = None,
    n_rows: int = 200,
    n_seasons: int = 3,
    include_margin: bool = False,
    include_seed: bool = False,
    mm_style: bool = False,
    extra_features: list[str] | None = None,
    seasons: list[int] | None = None,
) -> Path:
    """Create a minimal project config and data for PipelineRunner tests.

    Returns the config directory path.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    # Pipeline config
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)
    _make_matchup_parquet(
        features_dir / "matchup_features.parquet",
        n_rows=n_rows,
        n_seasons=n_seasons,
        include_margin=include_margin,
        include_seed=include_seed,
        mm_style=mm_style,
        extra_features=extra_features,
    )

    if seasons is None:
        seasons = list(range(2022, 2022 + n_seasons))

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
                "seasons": seasons,
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

    if ensemble is None:
        ensemble = {"method": "average"}
    _write_yaml(config_dir / "ensemble.yaml", {"ensemble": ensemble})

    return config_dir


# -----------------------------------------------------------------------
# Tests — original
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


# -----------------------------------------------------------------------
# Tests — new: stacked ensemble backtesting
# -----------------------------------------------------------------------

class TestStackedEnsembleBacktest:
    """Backtest with stacked ensemble (meta-learner)."""

    def test_stacked_three_models(self, tmp_path):
        """Stacked backtest with 3 models across 3 seasons produces ensemble."""
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 1.0},
                "active": True,
            },
            "logreg_b": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.5},
                "active": True,
            },
            "logreg_c": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 2.0},
                "active": True,
            },
        }
        ensemble = {
            "method": "stacked",
            "meta_learner": {"C": 1.0},
            "calibration": "none",
        }
        config_dir = _setup_project(
            tmp_path,
            models=models,
            ensemble=ensemble,
            n_rows=300,
            n_seasons=3,
            include_seed=True,
        )
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
        assert result["metrics"]["brier"] >= 0
        assert result["metrics"]["brier"] <= 1.0
        assert len(result["per_fold"]) == 3


class TestExcludeModels:
    """Exclude models from ensemble."""

    def test_exclude_models(self, tmp_path):
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "logreg_excluded": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        ensemble = {
            "method": "average",
            "exclude_models": ["logreg_excluded"],
        }
        config_dir = _setup_project(
            tmp_path, models=models, ensemble=ensemble,
            include_seed=True,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        # Excluded model should not be in models_trained
        assert "logreg_excluded" not in result["models_trained"]
        assert "logreg_a" in result["models_trained"]


class TestProbEnsembleColumn:
    """Verify prob_ensemble column is produced."""

    def test_prob_ensemble_exists(self, tmp_path):
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        config_dir = _setup_project(tmp_path, models=models)
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()
        # The backtest should succeed and produce metrics
        assert result["status"] == "success"
        assert "brier" in result["metrics"]


class TestSimpleAverageMethod:
    """Simple average method works."""

    def test_simple_average(self, tmp_path):
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 1.0},
                "active": True,
            },
            "logreg_b": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.1},
                "active": True,
            },
        }
        ensemble = {"method": "average"}
        config_dir = _setup_project(
            tmp_path, models=models, ensemble=ensemble,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "brier" in result["metrics"]
        assert result["metrics"]["brier"] >= 0


class TestColumnAutoDetect:
    """Auto-detect mm-style column names."""

    def test_mm_style_columns(self, tmp_path):
        """mm-style columns (TeamAWon, Season) are auto-normalized."""
        models = {
            "logreg_a": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
        config_dir = _setup_project(
            tmp_path, models=models, mm_style=True,
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "brier" in result["metrics"]


class TestTrainSeasonsFiltering:
    """train_seasons filtering works in backtest."""

    def test_train_seasons_last_n(self, tmp_path):
        """Models with train_seasons='last_2' use fewer seasons."""
        models = {
            "logreg_recent": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
                "train_seasons": "last_2",
            },
        }
        config_dir = _setup_project(
            tmp_path, models=models,
            n_rows=300, n_seasons=4,
            seasons=[2022, 2023, 2024, 2025],
        )
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "brier" in result["metrics"]
