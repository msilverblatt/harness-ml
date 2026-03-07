"""Tests for pipeline execution planner."""
from __future__ import annotations

import pytest

from harnessml.core.runner.pipeline_planner import (
    PipelinePlan,
    PipelineStep,
    plan_execution,
)
from harnessml.core.runner.schema import (
    BacktestConfig,
    DataConfig,
    EnsembleDef,
    ModelDef,
    ProjectConfig,
)


def _make_config(
    models: dict[str, dict] | None = None,
    ensemble: dict | None = None,
    backtest: dict | None = None,
) -> ProjectConfig:
    """Create a minimal ProjectConfig for testing."""
    if models is None:
        models = {
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        }

    model_defs = {name: ModelDef(**m) for name, m in models.items()}

    if ensemble is None:
        ensemble = {"method": "average"}

    if backtest is None:
        backtest = {
            "cv_strategy": "leave_one_out",
            "fold_values": [2022, 2023, 2024],
        }

    return ProjectConfig(
        data=DataConfig(
            raw_dir="data/raw",
            processed_dir="data/processed",
            features_dir="data/features",
        ),
        models=model_defs,
        ensemble=EnsembleDef(**ensemble),
        backtest=BacktestConfig(**backtest),
    )


class TestNoChanges:
    """When configs are identical, nothing needs to run."""

    def test_identical_configs(self):
        config = _make_config()
        plan = plan_execution(config, config)
        assert plan.is_empty
        assert len(plan.cache_hits) > 0

    def test_format_empty_plan(self):
        config = _make_config()
        plan = plan_execution(config, config)
        md = plan.format_summary()
        assert "No changes" in md


class TestNewModel:
    """Adding a new model should train only that model + re-ensemble."""

    def test_new_model_trains_only_new(self):
        current = _make_config()
        new = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "xgb_new": {
                "type": "xgboost",
                "features": ["diff_x"],
                "params": {},
                "active": True,
            },
        })
        plan = plan_execution(current, new)
        assert not plan.is_empty
        assert not plan.ensemble_only

        train_targets = plan.models_to_retrain
        assert "xgb_new" in train_targets
        assert "logreg" not in train_targets
        assert "logreg" in plan.cache_hits

        # Should have ensemble step too
        ensemble_steps = [s for s in plan.steps if s.stage == "ensemble"]
        assert len(ensemble_steps) == 1


class TestModelRemoved:
    """Removing a model should only re-ensemble."""

    def test_removed_model_ensemble_only(self):
        current = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "xgb": {
                "type": "xgboost",
                "features": ["diff_x"],
                "params": {},
                "active": True,
            },
        })
        new = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        })
        plan = plan_execution(current, new)
        assert plan.ensemble_only
        assert "logreg" in plan.cache_hits


class TestModelParamsChanged:
    """Changing model params should retrain only that model."""

    def test_params_changed_retrains_one(self):
        current = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 1.0},
                "active": True,
            },
            "xgb": {
                "type": "xgboost",
                "features": ["diff_x"],
                "params": {"n_estimators": 100},
                "active": True,
            },
        })
        new = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.1},
                "active": True,
            },
            "xgb": {
                "type": "xgboost",
                "features": ["diff_x"],
                "params": {"n_estimators": 100},
                "active": True,
            },
        })
        plan = plan_execution(current, new)
        assert "logreg" in plan.models_to_retrain
        assert "xgb" not in plan.models_to_retrain
        assert "xgb" in plan.cache_hits


class TestModelFeaturesChanged:
    """Changing model features should retrain that model."""

    def test_features_changed(self):
        current = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        })
        new = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x", "diff_y"],
                "params": {"max_iter": 200},
                "active": True,
            },
        })
        plan = plan_execution(current, new)
        assert "logreg" in plan.models_to_retrain

    def test_feature_added_shows_in_reason(self):
        current = _make_config()
        new = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x", "diff_new_feat"],
                "params": {"max_iter": 200},
                "active": True,
            },
        })
        plan = plan_execution(current, new)
        train_step = next(s for s in plan.steps if s.target == "logreg")
        assert "+1 features" in train_step.reason


class TestEnsembleConfigChanged:
    """Ensemble config changes should be ensemble-only."""

    def test_ensemble_method_change(self):
        current = _make_config(ensemble={"method": "average"})
        new = _make_config(ensemble={"method": "stacked"})
        plan = plan_execution(current, new)
        assert plan.ensemble_only

    def test_ensemble_param_change(self):
        current = _make_config(ensemble={"method": "average", "temperature": 1.0})
        new = _make_config(ensemble={"method": "average", "temperature": 0.5})
        plan = plan_execution(current, new)
        assert plan.ensemble_only


class TestBacktestConfigChanged:
    """Backtest config changes should force full retrain."""

    def test_backtest_change_full_retrain(self):
        current = _make_config(
            backtest={"cv_strategy": "leave_one_out", "fold_values": [2022, 2023, 2024]}
        )
        new = _make_config(
            backtest={"cv_strategy": "leave_one_out", "fold_values": [2022, 2023, 2024, 2025]}
        )
        plan = plan_execution(current, new)
        # Should have a train "all" step
        assert any(s.stage == "train" and s.target == "all" for s in plan.steps)


class TestModelActivation:
    """Activating/deactivating models."""

    def test_activate_model(self):
        current = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": False,
            },
        })
        new = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
        })
        plan = plan_execution(current, new)
        assert "logreg" in plan.models_to_retrain

    def test_deactivate_model_ensemble_only(self):
        current = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": True,
            },
            "xgb": {
                "type": "xgboost",
                "features": ["diff_x"],
                "params": {},
                "active": True,
            },
        })
        new = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200},
                "active": False,
            },
            "xgb": {
                "type": "xgboost",
                "features": ["diff_x"],
                "params": {},
                "active": True,
            },
        })
        plan = plan_execution(current, new)
        assert plan.ensemble_only
        # xgb should be a cache hit
        assert "xgb" in plan.cache_hits


class TestFormatSummary:
    """Test format_summary produces useful markdown."""

    def test_format_with_steps(self):
        plan = PipelinePlan(
            steps=[
                PipelineStep("train", "xgb_new", "New model", "1 model"),
                PipelineStep("ensemble", "all", "Re-ensemble", "all"),
            ],
            cache_hits=["logreg", "catboost"],
            reason="New model: xgb_new",
        )
        md = plan.format_summary()
        assert "Execution Plan" in md
        assert "xgb_new" in md
        assert "Cache Hits" in md
        assert "logreg" in md

    def test_format_ensemble_only(self):
        plan = PipelinePlan(
            steps=[
                PipelineStep("ensemble", "all", "Config changed", "all"),
            ],
            reason="Ensemble config changed",
        )
        md = plan.format_summary()
        assert "Ensemble-only" in md


class TestMultipleModelChanges:
    """Multiple models changed simultaneously."""

    def test_two_models_changed(self):
        current = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 1.0},
                "active": True,
            },
            "xgb": {
                "type": "xgboost",
                "features": ["diff_x"],
                "params": {"n_estimators": 100},
                "active": True,
            },
            "catboost": {
                "type": "catboost",
                "features": ["diff_x"],
                "params": {},
                "active": True,
            },
        })
        new = _make_config(models={
            "logreg": {
                "type": "logistic_regression",
                "features": ["diff_x"],
                "params": {"max_iter": 200, "C": 0.1},
                "active": True,
            },
            "xgb": {
                "type": "xgboost",
                "features": ["diff_x", "diff_y"],
                "params": {"n_estimators": 100},
                "active": True,
            },
            "catboost": {
                "type": "catboost",
                "features": ["diff_x"],
                "params": {},
                "active": True,
            },
        })
        plan = plan_execution(current, new)
        assert "logreg" in plan.models_to_retrain
        assert "xgb" in plan.models_to_retrain
        assert "catboost" in plan.cache_hits
