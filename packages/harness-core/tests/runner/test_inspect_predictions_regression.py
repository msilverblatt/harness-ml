"""Tests for inspect_predictions regression support.

inspect_predictions uses absolute residuals for regression tasks and
confidence scores for classification tasks.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest
import yaml
from harnessml.core.runner.config_writer.pipeline import inspect_predictions

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _setup_project(tmp_path, task, target_col, preds_df):
    """Create a minimal project with a run containing predictions."""
    project_dir = tmp_path / "project"
    config_dir = project_dir / "config"
    config_dir.mkdir(parents=True)

    pipeline_config = {
        "data": {
            "outputs_dir": "runs",
            "target_column": target_col,
            "task": task,
            "key_columns": ["id"] if "id" in preds_df.columns else [],
        },
    }
    (config_dir / "pipeline.yaml").write_text(yaml.dump(pipeline_config))

    run_dir = project_dir / "runs" / "run_001"
    preds_dir = run_dir / "predictions"
    preds_dir.mkdir(parents=True)
    preds_df.to_parquet(preds_dir / "predictions.parquet", index=False)
    (run_dir / "status.json").write_text(json.dumps({"status": "completed"}))

    return project_dir


# -----------------------------------------------------------------------
# Regression inspection
# -----------------------------------------------------------------------

class TestRegressionInspection:
    @pytest.fixture()
    def project(self, tmp_path):
        df = pd.DataFrame({
            "target": [10.0, 20.0, 30.0, 40.0, 50.0],
            "prediction": [12.0, 18.0, 35.0, 39.0, 42.0],
            "id": ["a", "b", "c", "d", "e"],
        })
        return _setup_project(tmp_path, "regression", "target", df)

    def test_worst_shows_largest_residuals(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="worst", top_n=3)
        assert "Largest Prediction Errors" in result
        # 50-42=8 is largest residual, should appear first
        lines = result.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("|") and "target" not in l and "---" not in l]
        assert len(data_lines) == 3
        assert "e" in data_lines[0]
        assert "8.0000" in data_lines[0]

    def test_best_shows_smallest_residuals(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="best", top_n=2)
        assert "Most Accurate" in result
        lines = result.strip().split("\n")
        data_lines = [l for l in lines if l.startswith("|") and "target" not in l and "---" not in l]
        assert len(data_lines) == 2
        # Smallest residual is |40-39|=1 (id=d)
        assert "d" in data_lines[0]
        assert "1.0000" in data_lines[0]

    def test_uncertain_shows_median_residuals(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="uncertain", top_n=2)
        assert "Median Error" in result

    def test_unknown_mode_returns_error(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="invalid")
        assert "Error" in result
        assert "invalid" in result

    def test_residual_column_in_output(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="worst")
        assert "_residual" in result

    def test_no_confidence_column_in_regression(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="worst")
        assert "_confidence" not in result

    def test_shows_count(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="worst", top_n=2)
        assert "2 of 5" in result


# -----------------------------------------------------------------------
# Classification inspection (preserved behavior)
# -----------------------------------------------------------------------

class TestClassificationInspection:
    @pytest.fixture()
    def project(self, tmp_path):
        df = pd.DataFrame({
            "target": [1, 0, 1, 0, 1],
            "ensemble_prob": [0.9, 0.8, 0.3, 0.2, 0.6],
            "id": ["a", "b", "c", "d", "e"],
        })
        return _setup_project(tmp_path, "binary", "target", df)

    def test_worst_shows_confident_wrong(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="worst", top_n=3)
        assert "Confident Wrong" in result
        # id=b has prob=0.8, predicted=1, actual=0 -> wrong with high confidence
        assert "b" in result

    def test_best_shows_confident_correct(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="best", top_n=3)
        assert "Confident Correct" in result
        # id=a has prob=0.9, predicted=1, actual=1 -> correct with high confidence
        assert "a" in result

    def test_uncertain_shows_near_half(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="uncertain", top_n=2)
        assert "Uncertain" in result

    def test_confidence_column_in_classification(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="worst")
        assert "_confidence" in result

    def test_no_residual_column_in_classification(self, project):
        result = inspect_predictions(project, run_id="run_001", mode="worst")
        assert "_residual" not in result
