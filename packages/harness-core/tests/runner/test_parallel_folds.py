"""Tests for fold-level parallelization."""
from __future__ import annotations

from pathlib import Path

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


def _setup_project(tmp_path: Path, n_seasons: int = 4) -> Path:
    """Create a minimal project config and data."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    n_rows = 200
    fold_vals = rng.choice(list(range(2022, 2022 + n_seasons)), size=n_rows)
    result = rng.integers(0, 2, size=n_rows)
    df = pd.DataFrame({
        "season": fold_vals,
        "result": result,
        "diff_x": rng.standard_normal(n_rows),
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
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }
    })

    _write_yaml(config_dir / "ensemble.yaml", {
        "ensemble": {"method": "average"},
    })

    return config_dir


def test_parallel_folds_produce_same_results_as_sequential(tmp_path):
    """Parallel and sequential fold execution should produce equivalent results."""
    config_dir = _setup_project(tmp_path)

    # Run sequential
    runner_seq = PipelineRunner(
        project_dir=tmp_path,
        config_dir=config_dir,
        enable_guards=False,
        parallel_folds=False,
    )
    runner_seq.load()
    results_seq = runner_seq.backtest()

    # Run parallel
    runner_par = PipelineRunner(
        project_dir=tmp_path,
        config_dir=config_dir,
        enable_guards=False,
        parallel_folds=True,
        max_workers=2,
    )
    runner_par.load()
    results_par = runner_par.backtest()

    # Compare key metrics
    seq_metrics = results_seq["metrics"]
    par_metrics = results_par["metrics"]

    for metric in seq_metrics:
        assert abs(seq_metrics[metric] - par_metrics[metric]) < 1e-6, (
            f"Metric {metric} differs: seq={seq_metrics[metric]}, par={par_metrics[metric]}"
        )


def test_parallel_folds_with_max_workers(tmp_path):
    """parallel_folds with explicit max_workers should succeed."""
    config_dir = _setup_project(tmp_path)

    runner = PipelineRunner(
        project_dir=tmp_path,
        config_dir=config_dir,
        enable_guards=False,
        parallel_folds=True,
        max_workers=1,  # single worker — effectively sequential
    )
    runner.load()
    result = runner.backtest()

    assert result["status"] == "success"
    assert "metrics" in result


def test_parallel_false_is_default(tmp_path):
    """Default should be sequential (parallel_folds=False)."""
    config_dir = _setup_project(tmp_path)

    runner = PipelineRunner(
        project_dir=tmp_path,
        config_dir=config_dir,
        enable_guards=False,
    )
    assert runner.parallel_folds is False
