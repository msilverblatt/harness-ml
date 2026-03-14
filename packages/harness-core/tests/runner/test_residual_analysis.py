"""Tests for residual analysis by feature bin."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.config_writer.pipeline import residual_analysis


class TestResidualAnalysis:
    """Residual analysis binned by feature values."""

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

    def test_detects_systematic_bias(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 200
        feature = np.linspace(0, 10, n)
        target = feature * 0.5
        # Predictions that systematically underpredict for high feature values
        predictions = target.copy()
        predictions[feature > 7] -= 5.0

        df = pd.DataFrame({
            "quality": feature,
            "prob_ensemble": predictions,
            "target": target,
        })
        project_dir = self._setup_run(tmp_path, df)
        result = residual_analysis(project_dir, feature="quality", run_id="run-001")
        assert "Residuals by `quality`" in result
        assert "YES" in result  # bias detected in high bins

    def test_no_bias_in_clean_predictions(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 200
        feature = rng.standard_normal(n)
        target = feature * 2.0
        # Perfect predictions with tiny noise
        predictions = target + rng.standard_normal(n) * 0.01

        df = pd.DataFrame({
            "x": feature,
            "prob_ensemble": predictions,
            "target": target,
        })
        project_dir = self._setup_run(tmp_path, df)
        result = residual_analysis(project_dir, feature="x", run_id="run-001")
        assert "YES" not in result

    def test_auto_selects_features(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame({
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
            "prob_ensemble": rng.standard_normal(n),
            "target": rng.standard_normal(n),
        })
        project_dir = self._setup_run(tmp_path, df)
        result = residual_analysis(project_dir, run_id="run-001")
        assert "feat_a" in result or "feat_b" in result

    def test_missing_predictions(self, tmp_path):
        project_dir = tmp_path / "project"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "pipeline.yaml").write_text(
            "data:\n  outputs_dir: outputs\n  target_column: target\n"
        )
        result = residual_analysis(project_dir, run_id="nonexistent")
        assert "Error" in result
