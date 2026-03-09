"""Tests for error recovery — partial results when individual models fail."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from harnessml.core.runner.pipeline import PipelineRunner

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _setup_project(tmp_path: Path) -> Path:
    """Create a minimal project with two models, one of which will be patched to fail."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    n_rows = 200
    n_seasons = 4
    fold_vals = rng.choice(list(range(2022, 2022 + n_seasons)), size=n_rows)
    result = rng.integers(0, 2, size=n_rows)
    df = pd.DataFrame({
        "season": fold_vals,
        "result": result,
        "diff_x": rng.standard_normal(n_rows),
        "diff_y": rng.standard_normal(n_rows),
    })
    df.to_parquet(features_dir / "features.parquet", index=False)

    _write_yaml(config_dir / "pipeline.yaml", {
        "data": {
            "raw_dir": str(tmp_path / "data" / "raw"),
            "processed_dir": str(tmp_path / "data" / "processed"),
            "features_dir": str(features_dir),
        },
        "backtest": {
            "cv_strategy": "leave_one_out",
            "fold_column": "season",
            "fold_values": list(range(2022, 2022 + n_seasons)),
            "metrics": ["brier", "accuracy"],
            "min_train_folds": 1,
        },
    })

    _write_yaml(config_dir / "models.yaml", {
        "models": {
            "good_model": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "bad_model": {
                "type": "logistic_regression",
                "features": ["diff_y"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
    })

    _write_yaml(config_dir / "ensemble.yaml", {
        "ensemble": {"method": "average"},
    })

    return config_dir


def test_partial_results_on_model_failure(tmp_path):
    """If one model fails, backtest should still return results from other models."""
    config_dir = _setup_project(tmp_path)
    runner = PipelineRunner(
        project_dir=tmp_path,
        config_dir=config_dir,
        enable_guards=False,
    )
    runner.load()

    # Patch train_single_model to fail for 'bad_model'
    original_train = __import__(
        "harnessml.core.runner.training.trainer", fromlist=["train_single_model"]
    ).train_single_model

    def _failing_train(model_name, *args, **kwargs):
        if model_name == "bad_model":
            raise RuntimeError("Simulated model failure")
        return original_train(model_name, *args, **kwargs)

    with patch(
        "harnessml.core.runner.pipeline.train_single_model",
        side_effect=_failing_train,
    ):
        result = runner.backtest()

    # Backtest should succeed with partial results
    assert result is not None
    assert result["status"] == "success"

    # bad_model should be in the failed list
    assert "bad_model" in result["models_failed"]

    # good_model should have produced predictions
    assert "good_model" not in result["models_failed"]


def test_all_models_fail_raises(tmp_path):
    """If all models fail, backtest should raise ValueError."""
    config_dir = _setup_project(tmp_path)
    runner = PipelineRunner(
        project_dir=tmp_path,
        config_dir=config_dir,
        enable_guards=False,
    )
    runner.load()

    with patch(
        "harnessml.core.runner.pipeline.train_single_model",
        side_effect=RuntimeError("All models fail"),
    ):
        with pytest.raises(ValueError, match="No valid holdout folds"):
            runner.backtest()


def test_failed_models_list_in_result(tmp_path):
    """Result should include failed_models even when empty."""
    config_dir = _setup_project(tmp_path)
    runner = PipelineRunner(
        project_dir=tmp_path,
        config_dir=config_dir,
        enable_guards=False,
    )
    runner.load()
    result = runner.backtest()

    # No failures expected — list should be empty
    assert result["models_failed"] == []
