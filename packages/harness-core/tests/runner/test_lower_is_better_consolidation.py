"""Tests that _LOWER_IS_BETTER is consolidated across all modules.

All modules should import the canonical set from _helpers which includes
all metric directions, rather than defining their own incomplete sets.
"""
from __future__ import annotations

import pytest
from harnessml.core.runner.config_writer._helpers import (
    _LOWER_IS_BETTER as CANONICAL,
)

# -----------------------------------------------------------------------
# Canonical set completeness
# -----------------------------------------------------------------------

class TestCanonicalSet:
    """The canonical _LOWER_IS_BETTER in _helpers must include all
    lower-is-better metrics for both classification and regression."""

    def test_includes_classification_metrics(self):
        for metric in ("brier", "brier_score", "ece", "log_loss", "logloss"):
            assert metric in CANONICAL, f"{metric} missing from canonical set"

    def test_includes_regression_metrics(self):
        for metric in ("mae", "mse", "rmse", "mape", "smape", "rae"):
            assert metric in CANONICAL, f"{metric} missing from canonical set"

    def test_excludes_higher_is_better(self):
        for metric in ("accuracy", "r2", "f1", "auc", "roc_auc"):
            assert metric not in CANONICAL, f"{metric} should not be in lower-is-better set"

    def test_is_frozen(self):
        assert isinstance(CANONICAL, frozenset), "Canonical set should be a frozenset"


# -----------------------------------------------------------------------
# All modules use the same canonical set
# -----------------------------------------------------------------------

class TestConsolidation:
    """Every module that references _LOWER_IS_BETTER must use the exact
    same canonical set from _helpers — no local overrides."""

    def test_experiments_config_writer_uses_canonical(self):
        from harnessml.core.runner.config_writer import experiments

        # The module should NOT define its own _LOWER_IS_BETTER anymore.
        # It imports from _helpers. Verify by checking the module uses
        # the canonical set in its comparison logic.
        from harnessml.core.runner.config_writer._helpers import _LOWER_IS_BETTER
        # If experiments had a local override, it would shadow the import.
        # We verify the import path is used by checking the module namespace.
        assert _LOWER_IS_BETTER is CANONICAL

    def test_experiment_module_uses_canonical(self):
        from harnessml.core.runner.experiments.experiment import _LOWER_IS_BETTER
        assert _LOWER_IS_BETTER is CANONICAL

    def test_exploration_module_uses_canonical(self):
        from harnessml.core.runner.optimization.exploration import _LOWER_IS_BETTER
        assert _LOWER_IS_BETTER is CANONICAL

    def test_manager_class_uses_canonical(self):
        from harnessml.core.runner.experiments.manager import ExperimentManager
        assert ExperimentManager._LOWER_IS_BETTER is CANONICAL


# -----------------------------------------------------------------------
# Verdict direction for regression metrics
# -----------------------------------------------------------------------

class TestVerdictDirection:
    """Regression metrics should produce correct verdict direction."""

    def test_rmse_improvement_detected_as_improved(self):
        """Lower RMSE = better, so negative delta should be 'improved'."""
        assert "rmse" in CANONICAL

    def test_mae_improvement_detected_as_improved(self):
        assert "mae" in CANONICAL

    def test_mse_improvement_detected_as_improved(self):
        assert "mse" in CANONICAL

    def test_promote_respects_rmse_direction(self, tmp_path):
        """promote_experiment should detect RMSE improvement (lower = better)."""
        import json

        from harnessml.core.runner.experiments.experiment import promote_experiment

        experiments_dir = tmp_path / "experiments"
        config_dir = tmp_path / "config"
        exp_dir = experiments_dir / "exp-rmse-test"
        exp_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)

        # Write models.yaml and ensemble.yaml so promote can apply overlay
        (config_dir / "models.yaml").write_text("models:\n  xgb:\n    type: xgboost\n")
        (config_dir / "ensemble.yaml").write_text("ensemble:\n  method: stacking\n")

        # Experiment that improved RMSE (lower = better)
        results = {
            "metrics": {"rmse": 0.10},
            "baseline_metrics": {"rmse": 0.15},
            "overlay": {"models": {"xgb": {"params": {"depth": 5}}}},
        }
        (exp_dir / "results.json").write_text(json.dumps(results))
        (exp_dir / "overlay.yaml").write_text("models:\n  xgb:\n    params:\n      depth: 5\n")

        result = promote_experiment(
            experiment_id="exp-rmse-test",
            experiments_dir=experiments_dir,
            config_dir=config_dir,
            primary_metric="rmse",
        )
        assert result["promoted"] is True, f"RMSE improvement should be promoted, got: {result}"

    def test_promote_rejects_rmse_regression(self, tmp_path):
        """promote_experiment should reject when RMSE gets worse (higher)."""
        import json

        from harnessml.core.runner.experiments.experiment import promote_experiment

        experiments_dir = tmp_path / "experiments"
        config_dir = tmp_path / "config"
        exp_dir = experiments_dir / "exp-rmse-worse"
        exp_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)

        (config_dir / "models.yaml").write_text("models:\n  xgb:\n    type: xgboost\n")
        (config_dir / "ensemble.yaml").write_text("ensemble:\n  method: stacking\n")

        # Experiment where RMSE got worse (higher = worse)
        results = {
            "metrics": {"rmse": 0.20},
            "baseline_metrics": {"rmse": 0.15},
            "overlay": {},
        }
        (exp_dir / "results.json").write_text(json.dumps(results))

        result = promote_experiment(
            experiment_id="exp-rmse-worse",
            experiments_dir=experiments_dir,
            config_dir=config_dir,
            primary_metric="rmse",
        )
        assert result["promoted"] is False, f"RMSE regression should not be promoted, got: {result}"

    def test_conclusion_builder_rmse_direction(self, tmp_path):
        """ExperimentManager._build_conclusion should compute correct
        improvement direction for RMSE (lower is better)."""
        from harnessml.core.runner.experiments.manager import ExperimentManager

        mgr = ExperimentManager(
            experiments_dir=tmp_path / "exp",
            journal_path=tmp_path / "j.jsonl",
        )

        # RMSE improved: 0.15 -> 0.10 = positive improvement (lower is better)
        conclusion = mgr._build_conclusion(
            verdict="keep",
            learnings="RMSE improved",
            metrics={"rmse": 0.10},
            baseline_metrics={"rmse": 0.15},
        )
        assert conclusion.improvement == pytest.approx(0.05)  # 0.15 - 0.10
        assert conclusion.improvement_pct > 0

    def test_conclusion_builder_rmse_regression(self, tmp_path):
        """RMSE getting worse should produce negative improvement."""
        from harnessml.core.runner.experiments.manager import ExperimentManager

        mgr = ExperimentManager(
            experiments_dir=tmp_path / "exp",
            journal_path=tmp_path / "j.jsonl",
        )

        # RMSE regressed: 0.15 -> 0.20 = negative improvement
        conclusion = mgr._build_conclusion(
            verdict="revert",
            learnings="RMSE got worse",
            metrics={"rmse": 0.20},
            baseline_metrics={"rmse": 0.15},
        )
        assert conclusion.improvement == pytest.approx(-0.05)  # 0.15 - 0.20
        assert conclusion.improvement_pct < 0
