"""Tests for JSON diagnostics output format."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from harnessml.core.runner.analysis.diagnostics import build_diagnostics_json


def _make_fold_predictions(n_folds: int = 3, n_rows: int = 50) -> list[pd.DataFrame]:
    """Create mock fold predictions DataFrames."""
    rng = np.random.default_rng(42)
    folds = []
    for _ in range(n_folds):
        result = rng.integers(0, 2, size=n_rows)
        df = pd.DataFrame({
            "result": result,
            "prob_model_a": rng.random(n_rows),
            "prob_model_b": rng.random(n_rows),
            "prob_ensemble": rng.random(n_rows),
        })
        folds.append(df)
    return folds


def test_diagnostics_json_binary():
    """Binary diagnostics JSON should contain metrics, calibration, and per_fold."""
    fold_preds = _make_fold_predictions()
    result = build_diagnostics_json(fold_preds, target_column="result", task="binary")

    # Verify JSON-serializable
    json_str = json.dumps(result)
    data = json.loads(json_str)

    assert data["task"] == "binary"
    assert data["n_folds"] == 3
    assert data["n_samples"] == 150
    assert "metrics" in data
    assert "calibration" in data
    assert "per_fold" in data

    # Should have metrics for each model
    assert "model_a" in data["metrics"]
    assert "ensemble" in data["metrics"]

    # Calibration should have bin data
    for model_name, cal in data["calibration"].items():
        assert "mean_predicted" in cal
        assert "mean_actual" in cal
        assert "bin_counts" in cal


def test_diagnostics_json_multiclass():
    """Multiclass diagnostics JSON should contain multiclass metrics."""
    rng = np.random.default_rng(42)
    n_rows = 50
    n_classes = 3
    folds = []
    for _ in range(2):
        result = rng.integers(0, n_classes, size=n_rows)
        data = {"result": result}
        for model in ["model_a", "model_b"]:
            probs = rng.random((n_rows, n_classes))
            probs = probs / probs.sum(axis=1, keepdims=True)
            for ci in range(n_classes):
                data[f"prob_{model}_c{ci}"] = probs[:, ci]
        folds.append(pd.DataFrame(data))

    result = build_diagnostics_json(folds, target_column="result", task="multiclass")
    json_str = json.dumps(result)
    data = json.loads(json_str)

    assert data["task"] == "multiclass"
    assert "metrics" in data
    assert "calibration" not in data  # Not computed for multiclass


def test_diagnostics_json_regression():
    """Regression diagnostics JSON should contain regression metrics."""
    rng = np.random.default_rng(42)
    n_rows = 50
    folds = []
    for _ in range(2):
        target = rng.standard_normal(n_rows)
        df = pd.DataFrame({
            "result": target,
            "prob_model_a": target + rng.standard_normal(n_rows) * 0.1,
        })
        folds.append(df)

    result = build_diagnostics_json(folds, target_column="result", task="regression")
    json_str = json.dumps(result)
    data = json.loads(json_str)

    assert data["task"] == "regression"
    assert "metrics" in data
    assert "model_a" in data["metrics"]
    assert "rmse" in data["metrics"]["model_a"]


def test_diagnostics_json_empty():
    """Empty fold list should return minimal JSON."""
    result = build_diagnostics_json([], target_column="result", task="binary")
    assert result["n_folds"] == 0
    assert result["n_samples"] == 0
    assert result["metrics"] == {}
