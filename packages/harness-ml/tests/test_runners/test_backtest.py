import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from harness.ml.runners.backtest import run_backtest, BacktestResult
from harness.ml.config.project import ProjectConfig, CVConfig
from harness.ml.config.models import ModelsConfig, SingleModelConfig
from harness.ml.config.ensemble import EnsembleConfig


@pytest.fixture
def simple_binary_data():
    """Simple binary data for backtest testing."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
        "feature_c": rng.rand(n) * 10,
    })
    df["target"] = (0.5 * df["feature_a"] - 0.3 * df["feature_b"] + rng.randn(n) * 0.5 > 0).astype(int)
    return df


class TestRunBacktest:
    def test_basic_backtest_single_model(self, simple_binary_data):
        result = run_backtest(
            data=simple_binary_data,
            project_config=ProjectConfig(
                task_type="binary", target_column="target",
                cv=CVConfig(strategy="kfold", n_folds=3),
                metrics=["brier", "accuracy"],
            ),
            models_config=ModelsConfig(models={
                "lr": SingleModelConfig(name="lr", model_type="logistic",
                                        features=["feature_a", "feature_b", "feature_c"]),
            }),
            ensemble_config=EnsembleConfig(method="average"),
        )
        assert isinstance(result, BacktestResult)
        assert "brier" in result.metrics
        assert "accuracy" in result.metrics
        assert result.metrics["accuracy"] > 0.5
        assert result.models_trained > 0
        assert len(result.per_fold_metrics) == 3
        assert result.predictions is not None
        assert len(result.predictions) == 100

    def test_backtest_two_models_ensemble(self, simple_binary_data):
        result = run_backtest(
            data=simple_binary_data,
            project_config=ProjectConfig(
                task_type="binary", target_column="target",
                cv=CVConfig(strategy="kfold", n_folds=3),
                metrics=["brier", "accuracy", "auroc"],
            ),
            models_config=ModelsConfig(models={
                "lr": SingleModelConfig(name="lr", model_type="logistic",
                                        features=["feature_a", "feature_b", "feature_c"]),
                "rf": SingleModelConfig(name="rf", model_type="random_forest",
                                        features=["feature_a", "feature_b", "feature_c"]),
            }),
            ensemble_config=EnsembleConfig(method="stacked"),
        )
        assert result.metrics["accuracy"] > 0.5
        assert result.metrics["auroc"] > 0.5
        assert result.models_trained >= 6  # 2 models x 3 folds

    def test_backtest_with_cache(self, simple_binary_data, tmp_path):
        config = ProjectConfig(
            task_type="binary", target_column="target",
            cv=CVConfig(strategy="kfold", n_folds=3),
            metrics=["brier"],
        )
        models = ModelsConfig(models={
            "lr": SingleModelConfig(name="lr", model_type="logistic",
                                    features=["feature_a", "feature_b", "feature_c"]),
        })
        ensemble = EnsembleConfig(method="average")

        # First run -- trains
        r1 = run_backtest(simple_binary_data, config, models, ensemble, cache_dir=tmp_path)
        assert r1.models_trained == 3
        assert r1.models_cached == 0

        # Second run -- should hit cache
        r2 = run_backtest(simple_binary_data, config, models, ensemble, cache_dir=tmp_path)
        assert r2.models_cached == 3
        assert r2.models_trained == 0

    def test_default_features_exclude_target(self):
        """The implicit feature path must not train on the answer column."""
        rng = np.random.RandomState(7)
        n = 400
        data = pd.DataFrame({
            "noise_a": rng.randn(n),
            "noise_b": rng.randn(n),
            "target": rng.randint(0, 2, n),
        })
        result = run_backtest(
            data=data,
            project_config=ProjectConfig(
                task_type="binary", target_column="target",
                cv=CVConfig(strategy="kfold", n_folds=5),
                metrics=["accuracy"],
            ),
            models_config=ModelsConfig(models={
                "lr": SingleModelConfig(name="lr", model_type="logistic"),
            }),
            ensemble_config=EnsembleConfig(method="average"),
        )
        assert result.metrics["accuracy"] < 0.7

    def test_explicit_target_feature_is_rejected(self, simple_binary_data):
        with pytest.raises(ValueError, match="forbidden feature"):
            run_backtest(
                data=simple_binary_data,
                project_config=ProjectConfig(target_column="target"),
                models_config=ModelsConfig(models={
                    "lr": SingleModelConfig(
                        name="lr", model_type="logistic", features=["target"]
                    ),
                }),
                ensemble_config=EnsembleConfig(method="average"),
            )

    def test_fold_and_excluded_columns_are_not_default_features(self):
        rng = np.random.RandomState(11)
        n = 200
        data = pd.DataFrame({
            "signal": rng.randn(n),
            "season": np.repeat(np.arange(5), 40),
            "row_id": np.arange(n),
        })
        data["target"] = (data["signal"] > 0).astype(int)
        result = run_backtest(
            data=data,
            project_config=ProjectConfig(
                target_column="target",
                cv=CVConfig(
                    strategy="expanding_window", fold_column="season", min_train_folds=2
                ),
                exclude_columns=["row_id"],
                metrics=["accuracy"],
            ),
            models_config=ModelsConfig(models={
                "lr": SingleModelConfig(name="lr", model_type="logistic"),
            }),
            ensemble_config=EnsembleConfig(method="average"),
        )
        assert result.models_trained == 3

    def test_changed_data_invalidates_cache(self, simple_binary_data, tmp_path):
        config = ProjectConfig(
            target_column="target", cv=CVConfig(strategy="kfold", n_folds=3)
        )
        models = ModelsConfig(models={
            "lr": SingleModelConfig(name="lr", model_type="logistic")
        })
        ensemble = EnsembleConfig(method="average")
        first = run_backtest(simple_binary_data, config, models, ensemble, cache_dir=tmp_path)
        assert first.models_trained == 3

        changed = simple_binary_data.copy()
        changed.loc[0, "feature_a"] += 1
        second = run_backtest(changed, config, models, ensemble, cache_dir=tmp_path)
        assert second.models_trained == 3
        assert second.models_cached == 0

    def test_backtest_inactive_model_skipped(self, simple_binary_data):
        result = run_backtest(
            data=simple_binary_data,
            project_config=ProjectConfig(
                task_type="binary", target_column="target",
                cv=CVConfig(strategy="kfold", n_folds=3),
                metrics=["brier"],
            ),
            models_config=ModelsConfig(models={
                "lr": SingleModelConfig(name="lr", model_type="logistic",
                                        features=["feature_a", "feature_b", "feature_c"]),
                "inactive": SingleModelConfig(name="inactive", model_type="logistic",
                                              features=["feature_a"], active=False),
            }),
            ensemble_config=EnsembleConfig(method="average"),
        )
        # Only lr should train (3 folds)
        assert result.models_trained == 3

    def test_backtest_model_failure_continues(self, simple_binary_data):
        result = run_backtest(
            data=simple_binary_data,
            project_config=ProjectConfig(
                task_type="binary", target_column="target",
                cv=CVConfig(strategy="kfold", n_folds=3),
                metrics=["brier"],
            ),
            models_config=ModelsConfig(models={
                "lr": SingleModelConfig(name="lr", model_type="logistic",
                                        features=["feature_a", "feature_b", "feature_c"]),
                "bad": SingleModelConfig(name="bad", model_type="nonexistent_model_type",
                                         features=["feature_a"]),
            }),
            ensemble_config=EnsembleConfig(method="average"),
        )
        assert result.models_trained > 0  # lr trained
        assert len(result.models_failed) > 0  # bad failed
        assert "brier" in result.metrics  # still got metrics from lr

    def test_all_models_failing_raises(self, simple_binary_data):
        with pytest.raises(RuntimeError, match="No model produced predictions"):
            run_backtest(
                data=simple_binary_data,
                project_config=ProjectConfig(target_column="target"),
                models_config=ModelsConfig(models={
                    "bad": SingleModelConfig(
                        name="bad", model_type="does_not_exist", features=["feature_a"]
                    )
                }),
                ensemble_config=EnsembleConfig(method="average"),
            )

    def test_prediction_artifact_preserves_row_and_fold_identity(self, simple_binary_data):
        result = run_backtest(
            data=simple_binary_data,
            project_config=ProjectConfig(
                target_column="target", cv=CVConfig(n_folds=3)
            ),
            models_config=ModelsConfig(models={
                "lr": SingleModelConfig(name="lr", model_type="logistic")
            }),
            ensemble_config=EnsembleConfig(method="average"),
        )
        assert {"row_position", "row_index", "fold_id", "y_true", "y_pred"}.issubset(
            result.predictions.columns
        )
        assert sorted(result.predictions["row_position"].tolist()) == list(range(100))

    def test_multiclass_backtest_outputs_per_class_probabilities(self):
        rng = np.random.RandomState(19)
        n = 180
        data = pd.DataFrame({"x": rng.randn(n), "z": rng.randn(n)})
        data["target"] = np.select(
            [data["x"] < -0.4, data["x"] > 0.4], [0, 2], default=1
        )
        result = run_backtest(
            data=data,
            project_config=ProjectConfig(
                task_type="multiclass", target_column="target",
                cv=CVConfig(strategy="stratified_kfold", n_folds=3),
                metrics=["accuracy"],
            ),
            models_config=ModelsConfig(models={
                "rf": SingleModelConfig(name="rf", model_type="random_forest")
            }),
            ensemble_config=EnsembleConfig(method="average"),
        )
        assert result.metrics["accuracy"] > 0.7
        assert {"y_pred_class_0", "y_pred_class_1", "y_pred_class_2"}.issubset(
            result.predictions.columns
        )

    def test_regression_predictions_are_not_probability_clipped(self):
        rng = np.random.RandomState(23)
        n = 150
        x = rng.randn(n)
        data = pd.DataFrame({"x": x, "target": 10 + 4 * x + rng.randn(n) * 0.1})
        result = run_backtest(
            data=data,
            project_config=ProjectConfig(
                task_type="regression", target_column="target",
                cv=CVConfig(n_folds=3), metrics=["r2", "rmse"],
            ),
            models_config=ModelsConfig(models={
                "rf": SingleModelConfig(name="rf", model_type="random_forest")
            }),
            ensemble_config=EnsembleConfig(method="stacked"),
        )
        assert result.metrics["r2"] > 0.8
        assert result.predictions["y_pred"].max() > 1.0

    def test_no_active_models_raises(self, simple_binary_data):
        with pytest.raises(ValueError, match="No active models"):
            run_backtest(
                data=simple_binary_data,
                project_config=ProjectConfig(task_type="binary", target_column="target"),
                models_config=ModelsConfig(models={}),
                ensemble_config=EnsembleConfig(),
            )
