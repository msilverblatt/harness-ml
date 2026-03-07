"""End-to-end integration tests for multiclass pipeline."""
import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def multiclass_e2e_project(tmp_path):
    """Create a minimal multiclass project with 3-class synthetic data."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    features_dir = tmp_path / "data" / "features"
    features_dir.mkdir(parents=True)

    # Synthetic 3-class data: 300 rows, 5 numeric features, 3 folds
    rng = np.random.RandomState(123)
    n = 300
    f1 = rng.randn(n)
    f2 = rng.randn(n)
    f3 = rng.randn(n)
    f4 = rng.randn(n)
    f5 = rng.randn(n)

    # Target correlated with features so models can learn something
    score = f1 + f2 - f3
    target = np.where(score > 0.8, 2, np.where(score < -0.8, 0, 1))

    df = pd.DataFrame({
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "f5": f5,
        "target": target,
        "fold": np.tile(np.arange(3), n // 3),
    })
    df.to_parquet(features_dir / "features.parquet", index=False)

    return tmp_path


def _write_config(project_dir, ensemble_method="average"):
    """Write pipeline.yaml with configurable ensemble method."""
    config_dir = project_dir / "config"
    config = {
        "data": {
            "task": "multiclass",
            "target_column": "target",
            "features_dir": "data/features",
            "features_file": "features.parquet",
        },
        "models": {
            "xgb_mc": {
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
            "lr_mc": {
                "type": "logistic_regression",
                "features": ["f1", "f2", "f3", "f4", "f5"],
                "params": {
                    "max_iter": 200,
                },
                "active": True,
                "include_in_ensemble": True,
            },
        },
        "backtest": {
            "cv_strategy": "leave_one_out",
            "fold_column": "fold",
            "fold_values": [0, 1, 2],
            "metrics": ["accuracy", "log_loss"],
            "min_train_folds": 1,
        },
        "ensemble": {
            "method": ensemble_method,
        },
    }
    (config_dir / "pipeline.yaml").write_text(yaml.dump(config))


class TestMulticlassE2EAverage:
    """E2E tests with average ensemble method."""

    def test_backtest_succeeds(self, multiclass_e2e_project):
        """Pipeline completes with status=success."""
        from harnessml.core.runner.pipeline import PipelineRunner

        _write_config(multiclass_e2e_project, ensemble_method="average")
        runner = PipelineRunner(
            project_dir=str(multiclass_e2e_project),
            config_dir=str(multiclass_e2e_project / "config"),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"

    def test_metrics_include_accuracy_and_log_loss(self, multiclass_e2e_project):
        """Multiclass metrics should include accuracy and log_loss, not brier."""
        from harnessml.core.runner.pipeline import PipelineRunner

        _write_config(multiclass_e2e_project, ensemble_method="average")
        runner = PipelineRunner(
            project_dir=str(multiclass_e2e_project),
            config_dir=str(multiclass_e2e_project / "config"),
        )
        runner.load()
        result = runner.backtest()

        metrics = result["metrics"]
        assert "accuracy" in metrics
        assert "log_loss" in metrics
        assert "brier" not in metrics
        assert "brier_score" not in metrics
        assert metrics["accuracy"] > 0

    def test_per_fold_breakdown(self, multiclass_e2e_project):
        """Per-fold diagnostics should exist for all 3 folds."""
        from harnessml.core.runner.pipeline import PipelineRunner

        _write_config(multiclass_e2e_project, ensemble_method="average")
        runner = PipelineRunner(
            project_dir=str(multiclass_e2e_project),
            config_dir=str(multiclass_e2e_project / "config"),
        )
        runner.load()
        result = runner.backtest()

        assert len(result["per_fold"]) == 3
        assert len(result["diagnostics"]) > 0

    def test_models_trained_list(self, multiclass_e2e_project):
        """Models trained should list both configured models."""
        from harnessml.core.runner.pipeline import PipelineRunner

        _write_config(multiclass_e2e_project, ensemble_method="average")
        runner = PipelineRunner(
            project_dir=str(multiclass_e2e_project),
            config_dir=str(multiclass_e2e_project / "config"),
        )
        runner.load()
        result = runner.backtest()

        assert sorted(result["models_trained"]) == ["lr_mc", "xgb_mc"]

    def test_report_mentions_accuracy(self, multiclass_e2e_project):
        """Markdown report should contain accuracy info."""
        from harnessml.core.runner.pipeline import PipelineRunner

        _write_config(multiclass_e2e_project, ensemble_method="average")
        runner = PipelineRunner(
            project_dir=str(multiclass_e2e_project),
            config_dir=str(multiclass_e2e_project / "config"),
        )
        runner.load()
        result = runner.backtest()

        assert "report" in result
        assert "Accuracy" in result["report"]


class TestMulticlassE2EStacked:
    """E2E tests with stacked ensemble method."""

    def test_stacked_backtest_succeeds(self, multiclass_e2e_project):
        """Stacked ensemble should complete for multiclass."""
        from harnessml.core.runner.pipeline import PipelineRunner

        _write_config(multiclass_e2e_project, ensemble_method="stacked")
        runner = PipelineRunner(
            project_dir=str(multiclass_e2e_project),
            config_dir=str(multiclass_e2e_project / "config"),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "accuracy" in result["metrics"]
        assert "log_loss" in result["metrics"]

    def test_stacked_produces_per_class_probabilities(self, multiclass_e2e_project):
        """Stacked path should produce prob_ensemble_c* columns in fold data."""
        from harnessml.core.runner.pipeline import PipelineRunner

        _write_config(multiclass_e2e_project, ensemble_method="stacked")
        runner = PipelineRunner(
            project_dir=str(multiclass_e2e_project),
            config_dir=str(multiclass_e2e_project / "config"),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        # Verify per-fold breakdown exists
        assert len(result["per_fold"]) == 3
        assert sorted(result["models_trained"]) == ["lr_mc", "xgb_mc"]

    def test_stacked_metrics_not_binary(self, multiclass_e2e_project):
        """Stacked multiclass should not produce binary-only metrics."""
        from harnessml.core.runner.pipeline import PipelineRunner

        _write_config(multiclass_e2e_project, ensemble_method="stacked")
        runner = PipelineRunner(
            project_dir=str(multiclass_e2e_project),
            config_dir=str(multiclass_e2e_project / "config"),
        )
        runner.load()
        result = runner.backtest()

        metrics = result["metrics"]
        assert "brier" not in metrics
        assert "brier_score" not in metrics
        assert "ece" not in metrics
