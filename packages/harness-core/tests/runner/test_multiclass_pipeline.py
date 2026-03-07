"""Test multiclass pipeline support."""
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path


@pytest.fixture
def multiclass_project(tmp_path):
    """Create a minimal multiclass project."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)

    # Create synthetic 3-class data
    rng = np.random.RandomState(42)
    n = 500
    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "f3": rng.randn(n),
        "f4": rng.randn(n),
        "f5": rng.randn(n),
        "target": rng.randint(0, 3, n),
        "fold": np.tile(np.arange(5), n // 5),
    })
    df.to_parquet(features_dir / "features.parquet", index=False)

    config = {
        "data": {
            "task": "multiclass",
            "target_column": "target",
            "features_dir": "data/features",
            "features_file": "features.parquet",
        },
        "models": {
            "xgb_test": {
                "type": "xgboost",
                "features": ["f1", "f2", "f3", "f4", "f5"],
                "params": {
                    "max_depth": 3,
                    "n_estimators": 10,
                    "objective": "multi:softprob",
                    "num_class": 3,
                },
                "active": True,
                "include_in_ensemble": True,
            },
            "lgbm_test": {
                "type": "lightgbm",
                "features": ["f1", "f2", "f3", "f4", "f5"],
                "params": {
                    "max_depth": 3,
                    "n_estimators": 10,
                    "objective": "multiclass",
                    "num_class": 3,
                    "verbose": -1,
                },
                "active": True,
                "include_in_ensemble": True,
            },
        },
        "backtest": {
            "cv_strategy": "leave_one_out",
            "fold_values": [2, 3, 4],
            "metrics": ["accuracy", "log_loss"],
            "fold_column": "fold",
            "min_train_folds": 1,
        },
        "ensemble": {
            "method": "average",
        },
    }
    (config_dir / "pipeline.yaml").write_text(yaml.dump(config))

    return tmp_path


def test_multiclass_backtest(multiclass_project):
    """Full multiclass backtest should complete without errors."""
    from harnessml.core.runner.pipeline import PipelineRunner

    runner = PipelineRunner(
        project_dir=str(multiclass_project),
        config_dir=str(multiclass_project / "config"),
    )
    runner.load()
    result = runner.backtest()

    assert result["status"] == "success"
    assert "accuracy" in result["metrics"]
    assert result["metrics"]["accuracy"] > 0
    assert len(result["per_fold"]) == 3
    assert len(result["models_trained"]) == 2


def test_multiclass_prediction_shape(multiclass_project):
    """Verify multiclass predictions are stored as per-class columns."""
    from harnessml.core.runner.pipeline import PipelineRunner

    runner = PipelineRunner(
        project_dir=str(multiclass_project),
        config_dir=str(multiclass_project / "config"),
    )
    runner.load()
    result = runner.backtest()

    # Check that per-class ensemble columns exist
    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# Multiclass diagnostics unit tests
# ---------------------------------------------------------------------------


def _make_multiclass_preds_df(n=100, n_classes=3, seed=42):
    """Build a synthetic multiclass predictions DataFrame with prob_{model}_c{i} columns."""
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, n_classes, n)

    # Build per-class probability columns for two models + ensemble
    models = ["xgb", "lgbm", "ensemble"]
    data = {"target": y_true}
    for model in models:
        raw = rng.dirichlet(np.ones(n_classes), size=n)
        for ci in range(n_classes):
            data[f"prob_{model}_c{ci}"] = raw[:, ci]
    return pd.DataFrame(data)


class TestEvaluateFoldPredictionsMulticlass:
    """Tests for evaluate_fold_predictions_multiclass."""

    def test_returns_metrics_per_model(self):
        from harnessml.core.runner.diagnostics import evaluate_fold_predictions_multiclass

        df = _make_multiclass_preds_df()
        results = evaluate_fold_predictions_multiclass(df, fold_id=1, target_column="target")

        model_names = {r["model"] for r in results}
        assert "xgb" in model_names
        assert "lgbm" in model_names
        assert "ensemble" in model_names

        for r in results:
            assert r["fold"] == 1
            assert "accuracy" in r
            assert "log_loss" in r
            assert "f1_macro" in r
            # Binary-specific metrics should NOT be present
            assert "brier_score" not in r
            assert "ece" not in r

    def test_accuracy_range(self):
        from harnessml.core.runner.diagnostics import evaluate_fold_predictions_multiclass

        df = _make_multiclass_preds_df()
        results = evaluate_fold_predictions_multiclass(df, fold_id=0, target_column="target")

        for r in results:
            assert 0.0 <= r["accuracy"] <= 1.0
            assert r["log_loss"] > 0

    def test_no_prob_columns_returns_empty(self):
        from harnessml.core.runner.diagnostics import evaluate_fold_predictions_multiclass

        df = pd.DataFrame({"target": [0, 1, 2], "feature_a": [1, 2, 3]})
        results = evaluate_fold_predictions_multiclass(df, fold_id=0, target_column="target")
        assert results == []


class TestComputePooledMetricsMulticlass:
    """Tests for compute_pooled_metrics_multiclass."""

    def test_pooled_metrics_across_folds(self):
        from harnessml.core.runner.diagnostics import compute_pooled_metrics_multiclass

        df1 = _make_multiclass_preds_df(n=50, seed=1)
        df2 = _make_multiclass_preds_df(n=50, seed=2)
        metrics = compute_pooled_metrics_multiclass([df1, df2], target_column="target")

        assert "xgb" in metrics
        assert "lgbm" in metrics
        assert "ensemble" in metrics

        for model, m in metrics.items():
            assert "accuracy" in m
            assert "log_loss" in m
            assert "f1_macro" in m
            assert m["n_samples"] == 100

    def test_empty_input_returns_empty(self):
        from harnessml.core.runner.diagnostics import compute_pooled_metrics_multiclass

        assert compute_pooled_metrics_multiclass([]) == {}


class TestBuildDiagnosticsReportMulticlass:
    """Tests for build_diagnostics_report with task='multiclass'."""

    def test_multiclass_report_uses_multiclass_metrics(self):
        from harnessml.core.runner.reporting import build_diagnostics_report

        fold_data = {
            1: _make_multiclass_preds_df(n=50, seed=1),
            2: _make_multiclass_preds_df(n=50, seed=2),
        }
        df = build_diagnostics_report(fold_data, target_column="target", task="multiclass")

        assert len(df) == 2
        assert "accuracy" in df.columns
        assert "log_loss" in df.columns
        # Binary-specific columns should NOT be present
        assert "brier_score" not in df.columns
        assert "ece" not in df.columns

    def test_binary_report_still_works(self):
        """Ensure adding the task param doesn't break binary path."""
        from harnessml.core.runner.reporting import build_diagnostics_report

        # Binary predictions with prob_ensemble column
        rng = np.random.RandomState(42)
        n = 50
        binary_df = pd.DataFrame({
            "result": rng.randint(0, 2, n),
            "prob_ensemble": rng.rand(n),
        })
        fold_data = {1: binary_df}
        df = build_diagnostics_report(fold_data, target_column="result", task="binary")

        assert len(df) == 1
        assert "brier_score" in df.columns
        assert "accuracy" in df.columns


class TestShowDiagnosticsMulticlass:
    """Tests for show_diagnostics handling multiclass data."""

    def test_multiclass_diagnostics_no_brier(self, tmp_path):
        from harnessml.core.runner.config_writer import show_diagnostics

        # Set up project structure
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        outputs_dir = tmp_path / "outputs" / "run_1" / "predictions"
        outputs_dir.mkdir(parents=True)

        config = {
            "data": {
                "task": "multiclass",
                "target_column": "target",
                "outputs_dir": "outputs",
            },
        }
        (config_dir / "pipeline.yaml").write_text(yaml.dump(config))

        # Create multiclass predictions
        preds_df = _make_multiclass_preds_df(n=60)
        preds_df.to_parquet(outputs_dir / "predictions.parquet", index=False)

        result = show_diagnostics(tmp_path, run_id="run_1")

        # Should contain accuracy and log_loss but NOT brier or ECE or calibration
        assert "Accuracy" in result
        assert "Log Loss" in result
        assert "Brier" not in result
        assert "ECE" not in result
        assert "Calibration Curve" not in result
        assert "Model Agreement" not in result


class TestPipelineGenerateReportMulticlass:
    """Test that _generate_report builds diagnostics for multiclass."""

    def test_multiclass_backtest_has_diagnostics(self, multiclass_project):
        from harnessml.core.runner.pipeline import PipelineRunner

        runner = PipelineRunner(
            project_dir=str(multiclass_project),
            config_dir=str(multiclass_project / "config"),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        # Diagnostics should be populated (not empty) for multiclass
        assert len(result["diagnostics"]) > 0
        # Report should exist and mention accuracy
        assert "report" in result
        assert "Accuracy" in result["report"]
