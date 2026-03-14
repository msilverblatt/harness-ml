"""Tests for warm-starting exploration from previous study results."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestLoadPreviousTrials:
    """Reconstruct FrozenTrial objects from study.json."""

    def _write_study_json(self, tmp_path: Path, trials: list[dict]) -> Path:
        study_data = {
            "study_name": "expl-001",
            "direction": "MAXIMIZE",
            "n_trials": len(trials),
            "best_trial": 0,
            "best_value": 0.85,
            "trials": trials,
        }
        path = tmp_path / "study.json"
        path.write_text(json.dumps(study_data, indent=2))
        return path

    def test_loads_completed_trials(self, tmp_path):
        from harnessml.core.runner.optimization.exploration import _load_previous_trials

        trials = [
            {
                "number": 0,
                "state": "COMPLETE",
                "value": 0.85,
                "params": {"lr": 0.01, "depth": 5},
                "user_attrs": {"brier": 0.15},
            },
            {
                "number": 1,
                "state": "COMPLETE",
                "value": 0.82,
                "params": {"lr": 0.1, "depth": 3},
                "user_attrs": {},
            },
        ]
        path = self._write_study_json(tmp_path, trials)
        result = _load_previous_trials(path)

        assert len(result) == 2
        assert result[0].values == [0.85]
        assert result[0].params["lr"] == 0.01
        assert result[0].params["depth"] == 5

    def test_skips_pruned_trials(self, tmp_path):
        from harnessml.core.runner.optimization.exploration import _load_previous_trials

        trials = [
            {
                "number": 0,
                "state": "COMPLETE",
                "value": 0.85,
                "params": {"lr": 0.01},
                "user_attrs": {},
            },
            {
                "number": 1,
                "state": "PRUNED",
                "value": None,
                "params": {"lr": 0.5},
                "user_attrs": {},
            },
            {
                "number": 2,
                "state": "FAIL",
                "value": None,
                "params": {"lr": 0.9},
                "user_attrs": {},
            },
        ]
        path = self._write_study_json(tmp_path, trials)
        result = _load_previous_trials(path)

        assert len(result) == 1
        assert result[0].params["lr"] == 0.01

    def test_handles_categorical_params(self, tmp_path):
        from harnessml.core.runner.optimization.exploration import _load_previous_trials

        trials = [
            {
                "number": 0,
                "state": "COMPLETE",
                "value": 0.80,
                "params": {"model_type": "xgb", "use_feature_x": True},
                "user_attrs": {},
            },
        ]
        path = self._write_study_json(tmp_path, trials)
        result = _load_previous_trials(path)

        assert len(result) == 1
        assert result[0].params["model_type"] == "xgb"
        assert result[0].params["use_feature_x"] is True

    def test_empty_trials(self, tmp_path):
        from harnessml.core.runner.optimization.exploration import _load_previous_trials

        path = self._write_study_json(tmp_path, [])
        result = _load_previous_trials(path)
        assert result == []

    def test_warm_start_adds_trials_to_study(self, tmp_path):
        """Verify that loaded trials can be added to a new Optuna study."""
        import optuna
        from harnessml.core.runner.optimization.exploration import _load_previous_trials

        trials = [
            {
                "number": 0,
                "state": "COMPLETE",
                "value": 0.85,
                "params": {"lr": 0.01},
                "user_attrs": {},
            },
            {
                "number": 1,
                "state": "COMPLETE",
                "value": 0.82,
                "params": {"lr": 0.1},
                "user_attrs": {},
            },
        ]
        path = self._write_study_json(tmp_path, trials)
        loaded = _load_previous_trials(path)

        study = optuna.create_study(direction="maximize")
        for trial in loaded:
            study.add_trial(trial)

        assert len(study.trials) == 2
        assert study.best_value == 0.85
