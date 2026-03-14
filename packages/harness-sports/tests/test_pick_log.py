"""Tests for build_pick_log (requires sports hooks for entity/prior column detection)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.analysis.reporting import build_pick_log
from harnessml.sports.hooks import register as register_sports_hooks


class TestBuildPickLog:
    """Test pick log construction."""

    @pytest.fixture(autouse=True)
    def _register_sports_hooks(self):
        """Register sports hooks for entity/prior column detection."""
        register_sports_hooks()
        yield

    def test_basic_pick_log(self):
        """Build pick log from simple predictions."""
        preds = pd.DataFrame({
            "prob_ensemble": [0.7, 0.3, 0.6],
            "prob_model_a": [0.65, 0.35, 0.55],
            "prob_model_b": [0.75, 0.25, 0.65],
            "result": [1, 0, 1],
        })
        log = build_pick_log(preds, fold_id=2023, fold_column="season")
        assert len(log) == 3
        assert "correct" in log.columns
        assert "confidence" in log.columns
        assert "model_agreement_pct" in log.columns
        assert log["correct"].sum() == 3  # all correct

    def test_pick_log_with_team_info(self):
        """Pick log includes entity/prior columns when available."""
        preds = pd.DataFrame({
            "prob_ensemble": [0.7, 0.3],
            "result": [1, 0],
            "TeamA": ["Duke", "UNC"],
            "TeamB": ["Kentucky", "Kansas"],
            "seed_a": [1, 4],
            "seed_b": [16, 13],
            "Round": [1, 1],
        })
        log = build_pick_log(preds, fold_id=2023, fold_column="season")
        assert "entity_a" in log.columns
        assert "entity_b" in log.columns
        assert "prior_a" in log.columns
        assert "prior_b" in log.columns
        assert "round" in log.columns
        assert log["entity_a"].iloc[0] == "Duke"

    def test_pick_log_confidence(self):
        """Confidence = abs(prob_a - 0.5)."""
        preds = pd.DataFrame({
            "prob_ensemble": [0.8, 0.3, 0.5],
            "result": [1, 0, 1],
        })
        log = build_pick_log(preds, fold_id=2023, fold_column="season")
        expected = [0.3, 0.2, 0.0]
        np.testing.assert_array_almost_equal(
            log["confidence"].values, expected
        )

    def test_pick_log_predicted_winner(self):
        """predicted_winner is A when prob > 0.5, B otherwise."""
        preds = pd.DataFrame({
            "prob_ensemble": [0.8, 0.3, 0.5],
            "result": [1, 0, 0],
        })
        log = build_pick_log(preds, fold_id=2023, fold_column="season")
        assert log["predicted_winner"].tolist() == ["A", "B", "B"]

    def test_pick_log_actual_winner(self):
        """actual_winner is A when result==1, B when result==0."""
        preds = pd.DataFrame({
            "prob_ensemble": [0.7, 0.3],
            "result": [1, 0],
        })
        log = build_pick_log(preds, fold_id=2023, fold_column="season")
        assert log["actual_winner"].tolist() == ["A", "B"]

    def test_pick_log_fold_column(self):
        """Fold column is set correctly."""
        preds = pd.DataFrame({
            "prob_ensemble": [0.7],
            "result": [1],
        })
        log = build_pick_log(preds, fold_id=2025, fold_column="season")
        assert log["season"].iloc[0] == 2025

    def test_pick_log_without_optional_columns(self):
        """Pick log works without entity/prior/round columns."""
        preds = pd.DataFrame({
            "prob_ensemble": [0.7, 0.3],
            "result": [1, 0],
        })
        log = build_pick_log(preds, fold_id=2023, fold_column="season")
        assert "entity_a" not in log.columns
        assert "prior_a" not in log.columns
        assert "round" not in log.columns
