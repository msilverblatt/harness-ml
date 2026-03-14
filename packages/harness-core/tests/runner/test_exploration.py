"""Tests for the Bayesian exploration engine."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from harnessml.core.runner.optimization.exploration import (
    AxisDef,
    ExplorationSpace,
    _apply_subset_to_overlay,
    _build_overlay,
    _next_exploration_dir,
    format_exploration_report,
)

# -----------------------------------------------------------------------
# Schema validation
# -----------------------------------------------------------------------

class TestAxisDef:
    def test_continuous_requires_low_high(self):
        with pytest.raises(ValueError, match="requires 'low' and 'high'"):
            AxisDef(key="x", type="continuous")

    def test_continuous_valid(self):
        axis = AxisDef(key="x", type="continuous", low=0.0, high=1.0)
        assert axis.low == 0.0

    def test_integer_valid(self):
        axis = AxisDef(key="x", type="integer", low=1, high=10)
        assert axis.high == 10

    def test_categorical_requires_values(self):
        with pytest.raises(ValueError, match="requires 'values'"):
            AxisDef(key="x", type="categorical", values=["only_one"])

    def test_categorical_valid(self):
        axis = AxisDef(key="x", type="categorical", values=["a", "b"])
        assert axis.values == ["a", "b"]

    def test_subset_requires_candidates(self):
        with pytest.raises(ValueError, match="requires 'candidates'"):
            AxisDef(key="x", type="subset", candidates=[])

    def test_subset_valid(self):
        axis = AxisDef(key="x", type="subset", candidates=["a", "b"], min_size=1)
        assert axis.candidates == ["a", "b"]

    def test_type_alias_int(self):
        axis = AxisDef(key="x", type="int", low=1, high=10)
        assert axis.type == "integer"

    def test_type_alias_float(self):
        axis = AxisDef(key="x", type="float", low=0.0, high=1.0)
        assert axis.type == "continuous"

    def test_choices_alias_for_values(self):
        axis = AxisDef(key="x", type="categorical", choices=["a", "b", "c"])
        assert axis.values == ["a", "b", "c"]

    def test_param_alias_for_key(self):
        axis = AxisDef(param="models.xgb.params.lr", type="continuous", low=0.01, high=0.3)
        assert axis.key == "models.xgb.params.lr"

    def test_name_alias_for_key(self):
        axis = AxisDef(name="models.xgb.params.depth", type="integer", low=3, high=10)
        assert axis.key == "models.xgb.params.depth"


class TestExplorationSpace:
    def test_from_dict(self):
        space = ExplorationSpace(**{
            "axes": [
                {"key": "lr", "type": "continuous", "low": 0.01, "high": 0.3},
                {"key": "depth", "type": "integer", "low": 3, "high": 10},
            ],
            "budget": 10,
            "primary_metric": "brier",
        })
        assert len(space.axes) == 2
        assert space.budget == 10

    def test_defaults(self):
        space = ExplorationSpace(axes=[
            AxisDef(key="x", type="continuous", low=0.0, high=1.0),
        ])
        assert space.budget == 20
        assert space.primary_metric == "brier"
        assert space.baseline is True

    def test_axes_as_dict(self):
        """Accept axes as {param_path: {low, high, type}} dict."""
        space = ExplorationSpace(**{
            "axes": {
                "models.xgb.params.lr": {"low": 0.01, "high": 0.3, "type": "float"},
                "models.xgb.params.depth": {"low": 3, "high": 10, "type": "int"},
            },
            "budget": 15,
            "primary_metric": "rmse",
        })
        assert len(space.axes) == 2
        keys = {a.key for a in space.axes}
        assert keys == {"models.xgb.params.lr", "models.xgb.params.depth"}
        assert space.budget == 15

    def test_axes_as_list_with_param_alias(self):
        """Accept 'param' instead of 'key' in list-style axes."""
        space = ExplorationSpace(**{
            "axes": [
                {"param": "models.xgb.params.lr", "low": 0.01, "high": 0.3, "type": "float"},
            ],
            "budget": 10,
            "primary_metric": "rmse",
        })
        assert space.axes[0].key == "models.xgb.params.lr"
        assert space.axes[0].type == "continuous"


# -----------------------------------------------------------------------
# Subset routing
# -----------------------------------------------------------------------

class TestSubsetRouting:
    def test_models_active_routing(self):
        axis = AxisDef(
            key="models.active", type="subset",
            candidates=["xgb", "lgbm", "catboost"], min_size=2,
        )
        overlay: dict = {}
        _apply_subset_to_overlay(overlay, axis, ["xgb", "catboost"])

        assert overlay["models"]["xgb"]["active"] is True
        assert overlay["models"]["lgbm"]["active"] is False
        assert overlay["models"]["catboost"]["active"] is True

    def test_features_include_routing(self):
        axis = AxisDef(
            key="features.include", type="subset",
            candidates=["diff_tempo", "diff_luck"], min_size=1,
        )
        overlay: dict = {}
        _apply_subset_to_overlay(overlay, axis, ["diff_tempo"])

        assert overlay["data"]["feature_defs"]["diff_tempo"]["enabled"] is True
        assert overlay["data"]["feature_defs"]["diff_luck"]["enabled"] is False

    def test_generic_subset(self):
        axis = AxisDef(
            key="custom.list", type="subset",
            candidates=["a", "b", "c"], min_size=1,
        )
        overlay: dict = {}
        _apply_subset_to_overlay(overlay, axis, ["a", "c"])
        assert overlay["custom"]["list"] == ["a", "c"]


# -----------------------------------------------------------------------
# Overlay builder
# -----------------------------------------------------------------------

class TestBuildOverlay:
    def test_continuous_and_integer(self):
        """Mock trial that returns fixed values."""
        trial = MagicMock()
        trial.suggest_float.return_value = 0.1
        trial.suggest_int.return_value = 5

        axes = [
            AxisDef(key="models.xgb.params.lr", type="continuous", low=0.01, high=0.3),
            AxisDef(key="models.xgb.params.depth", type="integer", low=3, high=10),
        ]
        overlay = _build_overlay(trial, axes)

        assert overlay["models"]["xgb"]["params"]["lr"] == 0.1
        assert overlay["models"]["xgb"]["params"]["depth"] == 5

    def test_categorical(self):
        trial = MagicMock()
        trial.suggest_categorical.return_value = "stack"

        axes = [
            AxisDef(key="ensemble.method", type="categorical", values=["mean", "stack"]),
        ]
        overlay = _build_overlay(trial, axes)
        assert overlay["ensemble"]["method"] == "stack"


# -----------------------------------------------------------------------
# Directory numbering
# -----------------------------------------------------------------------

class TestNextExplorationDir:
    def test_first_exploration(self, tmp_path):
        result = _next_exploration_dir(tmp_path)
        assert result.name == "expl-001"

    def test_increments(self, tmp_path):
        (tmp_path / "experiments" / "expl-001").mkdir(parents=True)
        (tmp_path / "experiments" / "expl-002").mkdir(parents=True)
        result = _next_exploration_dir(tmp_path)
        assert result.name == "expl-003"

    def test_ignores_non_expl_dirs(self, tmp_path):
        (tmp_path / "experiments" / "exp-005").mkdir(parents=True)
        result = _next_exploration_dir(tmp_path)
        assert result.name == "expl-001"


# -----------------------------------------------------------------------
# Report formatting
# -----------------------------------------------------------------------

class TestFormatReport:
    def test_produces_markdown(self):
        """Smoke test with a mock study."""
        pytest.importorskip("optuna")
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")

        # Add a fake completed trial
        study.add_trial(
            optuna.trial.create_trial(
                params={"lr": 0.1},
                distributions={"lr": optuna.distributions.FloatDistribution(0.01, 0.3)},
                values=[0.19],
                user_attrs={"brier": 0.19, "accuracy": 0.67, "overlay": {"lr": 0.1}},
            )
        )

        space = ExplorationSpace(axes=[
            AxisDef(key="lr", type="continuous", low=0.01, high=0.3),
        ], budget=5, primary_metric="brier")

        report = format_exploration_report(
            exploration_id="expl-001",
            space=space,
            study=study,
            baseline_metrics={"brier": 0.20, "accuracy": 0.65},
            param_importance={},
            trial_results=[],
        )

        assert "expl-001" in report
        assert "Best trial" in report
        assert "Baseline Comparison" in report
        assert "brier" in report
        assert "accuracy" in report
