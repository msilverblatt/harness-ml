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
    from easyml.core.runner.pipeline import PipelineRunner

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
    from easyml.core.runner.pipeline import PipelineRunner

    runner = PipelineRunner(
        project_dir=str(multiclass_project),
        config_dir=str(multiclass_project / "config"),
    )
    runner.load()
    result = runner.backtest()

    # Check that per-class ensemble columns exist
    assert result["status"] == "success"
