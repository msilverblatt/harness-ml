"""Tests for model prediction correlation matrix."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.config_writer.pipeline import _load_predictions, model_correlation


class TestModelCorrelation:
    """Model correlation from prediction DataFrames."""

    def _setup_run(self, tmp_path, preds_df):
        project_dir = tmp_path / "project"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        outputs_dir = project_dir / "outputs" / "run-001"
        preds_dir = outputs_dir / "predictions"
        preds_dir.mkdir(parents=True)

        (config_dir / "pipeline.yaml").write_text(
            "data:\n  outputs_dir: outputs\n  target_column: target\n"
        )
        preds_df.to_parquet(preds_dir / "predictions.parquet", index=False)
        return project_dir

    def test_identical_predictions_have_correlation_1(self, tmp_path):
        rng = np.random.default_rng(42)
        preds = rng.standard_normal(100)
        df = pd.DataFrame({
            "prob_model_a": preds,
            "prob_model_b": preds,
            "target": rng.integers(0, 2, 100),
        })
        project_dir = self._setup_run(tmp_path, df)
        result = model_correlation(project_dir, run_id="run-001")
        assert "1.000" in result
        assert "Highly correlated" in result

    def test_uncorrelated_predictions(self, tmp_path):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "prob_model_a": rng.standard_normal(100),
            "prob_model_b": rng.standard_normal(100),
            "target": rng.integers(0, 2, 100),
        })
        project_dir = self._setup_run(tmp_path, df)
        result = model_correlation(project_dir, run_id="run-001")
        assert "Highly correlated" not in result

    def test_needs_at_least_2_models(self, tmp_path):
        df = pd.DataFrame({
            "prob_model_a": [0.5, 0.6, 0.7],
            "target": [0, 1, 0],
        })
        project_dir = self._setup_run(tmp_path, df)
        result = model_correlation(project_dir, run_id="run-001")
        assert "Error" in result
