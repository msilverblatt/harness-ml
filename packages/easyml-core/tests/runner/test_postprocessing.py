"""Tests for ensemble post-processing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.core.runner.calibration import SplineCalibrator
from easyml.core.runner.meta_learner import StackedEnsemble
from easyml.core.runner.postprocessing import (
    apply_logit_adjustments,
    apply_ensemble_postprocessing,
    apply_prior_compression,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_preds_df(n: int = 50, n_models: int = 3, seed: int = 42) -> pd.DataFrame:
    """Create a predictions DataFrame with prob_* columns and diff_prior."""
    rng = np.random.RandomState(seed)
    model_names = [f"model_{i}" for i in range(n_models)]
    data = {}
    for name in model_names:
        data[f"prob_{name}"] = rng.beta(2, 2, size=n)
    # Add logreg_seed (should be excluded by pipeline)
    data["prob_logreg_seed"] = rng.beta(2, 2, size=n)
    data["diff_prior"] = rng.choice([-8, -4, -2, -1, 1, 2, 4, 8], size=n).astype(float)
    return pd.DataFrame(data)


def _fit_meta_learner(preds_df: pd.DataFrame, model_names: list[str]) -> StackedEnsemble:
    """Fit a meta-learner on the preds DataFrame for testing."""
    rng = np.random.RandomState(123)
    n = len(preds_df)
    y_true = rng.randint(0, 2, size=n).astype(float)

    model_preds = {name: preds_df[f"prob_{name}"].values for name in model_names}
    prior_diffs = preds_df["diff_prior"].values

    meta = StackedEnsemble(model_names)
    meta.fit(model_preds, prior_diffs, y_true)
    return meta


def _fit_calibrator(n: int = 100, seed: int = 42) -> SplineCalibrator:
    """Fit a SplineCalibrator for testing."""
    rng = np.random.RandomState(seed)
    true_p = rng.beta(2, 2, size=n)
    y_true = (rng.rand(n) < true_p).astype(float)
    y_prob = np.clip(true_p + rng.normal(0, 0.1, n), 0.05, 0.95)

    cal = SplineCalibrator(prob_max=0.985, n_bins=10)
    cal.fit(y_true, y_prob)
    return cal


# -----------------------------------------------------------------------
# Full pipeline
# -----------------------------------------------------------------------

class TestApplyEnsemblePostprocessing:
    def test_full_pipeline(self):
        preds_df = _make_preds_df(50, 3)
        model_names = ["model_0", "model_1", "model_2"]
        meta = _fit_meta_learner(preds_df, model_names)
        calibrator = _fit_calibrator()

        config = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
            "prior_compression_threshold": 4,
        }

        result = apply_ensemble_postprocessing(
            preds_df, meta, calibrator, config,
        )

        assert "prob_ensemble" in result.columns
        assert len(result) == 50
        assert np.all(result["prob_ensemble"] >= 0)
        assert np.all(result["prob_ensemble"] <= 1)

    def test_prob_logreg_seed_excluded(self):
        """prob_logreg_seed should not be passed to the meta-learner."""
        preds_df = _make_preds_df(50, 2)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        config = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }

        # This should work without error (logreg_seed is excluded)
        result = apply_ensemble_postprocessing(preds_df, meta, None, config)
        assert "prob_ensemble" in result.columns

    def test_exclude_models_filtering(self):
        preds_df = _make_preds_df(50, 3)
        # Only use model_0, model_1 (exclude model_2)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        config = {
            "exclude_models": ["model_2"],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }

        result = apply_ensemble_postprocessing(preds_df, meta, None, config)
        assert "prob_ensemble" in result.columns

    def test_with_temperature_scaling(self):
        preds_df = _make_preds_df(50, 2)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        config_t1 = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }
        result_t1 = apply_ensemble_postprocessing(preds_df, meta, None, config_t1)

        config_t2 = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 2.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }
        result_t2 = apply_ensemble_postprocessing(preds_df, meta, None, config_t2)

        # T=2.0 should push probabilities closer to 0.5
        dist_from_half_t1 = np.abs(result_t1["prob_ensemble"] - 0.5).mean()
        dist_from_half_t2 = np.abs(result_t2["prob_ensemble"] - 0.5).mean()
        assert dist_from_half_t2 < dist_from_half_t1

    def test_with_clip_floor(self):
        preds_df = _make_preds_df(50, 2)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        config = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.1,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }

        result = apply_ensemble_postprocessing(preds_df, meta, None, config)
        assert result["prob_ensemble"].min() >= 0.1
        assert result["prob_ensemble"].max() <= 0.9

    def test_with_prior_compression(self):
        preds_df = _make_preds_df(50, 2)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        config_no_comp = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }
        result_no = apply_ensemble_postprocessing(preds_df, meta, None, config_no_comp)

        config_comp = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.3,
            "prior_compression_threshold": 4,
        }
        result_comp = apply_ensemble_postprocessing(preds_df, meta, None, config_comp)

        # Close seed matchups should be more compressed toward 0.5
        close_mask = np.abs(preds_df["diff_prior"]) <= 4
        if close_mask.sum() > 0:
            dist_no = np.abs(result_no.loc[close_mask, "prob_ensemble"] - 0.5).mean()
            dist_comp = np.abs(result_comp.loc[close_mask, "prob_ensemble"] - 0.5).mean()
            assert dist_comp < dist_no

    def test_with_pre_calibrators(self):
        preds_df = _make_preds_df(50, 2)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        # Fit a pre-calibrator for model_0
        rng = np.random.RandomState(42)
        pre_cal = SplineCalibrator(prob_max=0.985, n_bins=10)
        y_true = (rng.rand(100) < rng.beta(2, 2, size=100)).astype(float)
        y_prob = np.clip(rng.beta(2, 2, size=100), 0.05, 0.95)
        pre_cal.fit(y_true, y_prob)

        config = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }

        result = apply_ensemble_postprocessing(
            preds_df, meta, None, config,
            pre_calibrators={"model_0": pre_cal},
        )
        assert "prob_ensemble" in result.columns

    def test_default_config_noop(self):
        """Default/empty config should produce valid output (no-op pipeline)."""
        preds_df = _make_preds_df(50, 2)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        result = apply_ensemble_postprocessing(preds_df, meta, None, {})
        assert "prob_ensemble" in result.columns
        assert np.all(result["prob_ensemble"] >= 0)
        assert np.all(result["prob_ensemble"] <= 1)

    def test_does_not_mutate_input(self):
        """Input DataFrame should not be mutated."""
        preds_df = _make_preds_df(50, 2)
        original_cols = set(preds_df.columns)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        _ = apply_ensemble_postprocessing(preds_df, meta, None, {})
        assert set(preds_df.columns) == original_cols
        assert "prob_ensemble" not in preds_df.columns

    def test_with_post_calibrator(self):
        preds_df = _make_preds_df(50, 2)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)
        calibrator = _fit_calibrator()

        config = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }

        result_with = apply_ensemble_postprocessing(preds_df, meta, calibrator, config)
        result_without = apply_ensemble_postprocessing(preds_df, meta, None, config)

        # Post-calibration should change the output
        assert not np.allclose(
            result_with["prob_ensemble"].values,
            result_without["prob_ensemble"].values,
        )

    def test_with_logit_adjustments_paired(self):
        """Logit adjustments in paired mode should shift probabilities."""
        preds_df = _make_preds_df(50, 2)
        preds_df["a_health"] = 0.5  # entity A is unhealthy
        preds_df["b_health"] = 1.0  # entity B is healthy
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        config_no_adj = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }
        result_no = apply_ensemble_postprocessing(preds_df, meta, None, config_no_adj)

        config_adj = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [{
                "columns": ["a_health", "b_health"],
                "strength": 0.5,
                "default": 1.0,
                "mode": "paired",
            }],
            "prior_compression": 0.0,
        }
        result_adj = apply_ensemble_postprocessing(preds_df, meta, None, config_adj)

        # A is unhealthy -> probs should decrease (less confident in A)
        assert result_adj["prob_ensemble"].mean() < result_no["prob_ensemble"].mean()

    def test_with_logit_adjustments_diff(self):
        """Logit adjustments in diff mode should shift probabilities."""
        preds_df = _make_preds_df(50, 2)
        preds_df["diff_health"] = -0.5  # negative = A is worse
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        config_no_adj = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "prior_compression": 0.0,
        }
        result_no = apply_ensemble_postprocessing(preds_df, meta, None, config_no_adj)

        config_adj = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [{
                "columns": ["diff_health"],
                "strength": 0.5,
                "default": 0.0,
                "mode": "diff",
            }],
            "prior_compression": 0.0,
        }
        result_adj = apply_ensemble_postprocessing(preds_df, meta, None, config_adj)

        # Negative diff -> probs should decrease
        assert result_adj["prob_ensemble"].mean() < result_no["prob_ensemble"].mean()


# -----------------------------------------------------------------------
# Individual steps
# -----------------------------------------------------------------------

class TestApplySeedCompression:
    def test_compression_toward_half(self):
        probs = np.array([0.8, 0.2, 0.9, 0.1])
        preds = pd.DataFrame({
            "diff_prior": [2, 2, 10, 10],  # first two are close, last two are far
        })
        result = apply_prior_compression(probs, preds, compression=0.5, threshold=4)

        # Close matchups (seed_diff=2) should be compressed toward 0.5
        assert abs(result[0] - 0.65) < 1e-10  # 0.8 * 0.5 + 0.5 * 0.5 = 0.65
        assert abs(result[1] - 0.35) < 1e-10  # 0.2 * 0.5 + 0.5 * 0.5 = 0.35
        # Far matchups (seed_diff=10) should be unchanged
        assert abs(result[2] - 0.9) < 1e-10
        assert abs(result[3] - 0.1) < 1e-10

    def test_no_compression(self):
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({"diff_prior": [2, 2]})
        result = apply_prior_compression(probs, preds, compression=0.0, threshold=4)
        np.testing.assert_allclose(result, probs)

    def test_full_compression(self):
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({"diff_prior": [2, 2]})
        result = apply_prior_compression(probs, preds, compression=1.0, threshold=4)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_missing_seed_column(self):
        """Without diff_prior column, should return unchanged."""
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({"other_col": [1, 2]})
        result = apply_prior_compression(probs, preds, compression=0.5, threshold=4)
        np.testing.assert_allclose(result, probs)

    def test_negative_prior_diffs(self):
        """Negative seed diffs should also be handled (abs value)."""
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({"diff_prior": [-2, -2]})
        result = apply_prior_compression(probs, preds, compression=0.5, threshold=4)
        assert abs(result[0] - 0.65) < 1e-10
        assert abs(result[1] - 0.35) < 1e-10


class TestApplyLogitAdjustments:
    def test_paired_mode_both_healthy_no_change(self):
        """Both entities fully healthy -> no change."""
        probs = np.array([0.8, 0.2, 0.5])
        preds = pd.DataFrame({
            "a_health": [1.0, 1.0, 1.0],
            "b_health": [1.0, 1.0, 1.0],
        })
        adjustments = [{
            "columns": ["a_health", "b_health"],
            "strength": 0.5,
            "default": 1.0,
            "mode": "paired",
        }]
        result = apply_logit_adjustments(probs, preds, adjustments)
        np.testing.assert_allclose(result, probs, atol=1e-7)

    def test_paired_mode_a_unhealthy_pulls_down(self):
        """Entity A unhealthy should reduce probability (less confident in A)."""
        probs = np.array([0.7])
        preds = pd.DataFrame({
            "a_health": [0.5],
            "b_health": [1.0],
        })
        adjustments = [{
            "columns": ["a_health", "b_health"],
            "strength": 0.1,
            "default": 1.0,
            "mode": "paired",
        }]
        result = apply_logit_adjustments(probs, preds, adjustments)
        assert result[0] < 0.7

    def test_paired_mode_b_unhealthy_pulls_up(self):
        """Entity B unhealthy should increase probability (more confident in A)."""
        probs = np.array([0.3])
        preds = pd.DataFrame({
            "a_health": [1.0],
            "b_health": [0.5],
        })
        adjustments = [{
            "columns": ["a_health", "b_health"],
            "strength": 0.1,
            "default": 1.0,
            "mode": "paired",
        }]
        result = apply_logit_adjustments(probs, preds, adjustments)
        assert result[0] > 0.3

    def test_paired_mode_nan_uses_default(self):
        """NaN values should be replaced with default (1.0 = no penalty)."""
        probs = np.array([0.7])
        preds = pd.DataFrame({
            "a_health": [np.nan],
            "b_health": [np.nan],
        })
        adjustments = [{
            "columns": ["a_health", "b_health"],
            "strength": 0.5,
            "default": 1.0,
            "mode": "paired",
        }]
        result = apply_logit_adjustments(probs, preds, adjustments)
        np.testing.assert_allclose(result, probs, atol=1e-7)

    def test_paired_mode_zero_strength_no_change(self):
        """strength=0 should produce no change regardless of values."""
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({
            "a_health": [0.0, 0.0],
            "b_health": [0.0, 0.0],
        })
        adjustments = [{
            "columns": ["a_health", "b_health"],
            "strength": 0.0,
            "default": 1.0,
            "mode": "paired",
        }]
        result = apply_logit_adjustments(probs, preds, adjustments)
        np.testing.assert_allclose(result, probs, atol=1e-7)

    def test_diff_mode_positive_increases_prob(self):
        """Positive diff with positive strength should increase probability."""
        probs = np.array([0.5])
        preds = pd.DataFrame({"diff_metric": [1.0]})
        adjustments = [{
            "columns": ["diff_metric"],
            "strength": 0.5,
            "default": 0.0,
            "mode": "diff",
        }]
        result = apply_logit_adjustments(probs, preds, adjustments)
        assert result[0] > 0.5

    def test_diff_mode_negative_decreases_prob(self):
        """Negative diff with positive strength should decrease probability."""
        probs = np.array([0.5])
        preds = pd.DataFrame({"diff_metric": [-1.0]})
        adjustments = [{
            "columns": ["diff_metric"],
            "strength": 0.5,
            "default": 0.0,
            "mode": "diff",
        }]
        result = apply_logit_adjustments(probs, preds, adjustments)
        assert result[0] < 0.5

    def test_diff_mode_nan_uses_default(self):
        """NaN in diff mode should use default (0.0 = no change)."""
        probs = np.array([0.6])
        preds = pd.DataFrame({"diff_metric": [np.nan]})
        adjustments = [{
            "columns": ["diff_metric"],
            "strength": 1.0,
            "default": 0.0,
            "mode": "diff",
        }]
        result = apply_logit_adjustments(probs, preds, adjustments)
        np.testing.assert_allclose(result, probs, atol=1e-7)

    def test_missing_columns_skipped(self):
        """Adjustments referencing missing columns should be skipped."""
        probs = np.array([0.7, 0.3])
        preds = pd.DataFrame({"unrelated": [1, 2]})
        adjustments = [{
            "columns": ["a_health", "b_health"],
            "strength": 1.0,
            "default": 1.0,
            "mode": "paired",
        }]
        result = apply_logit_adjustments(probs, preds, adjustments)
        np.testing.assert_allclose(result, probs, atol=1e-7)

    def test_multiple_adjustments_stack(self):
        """Multiple adjustments should be applied sequentially."""
        probs = np.array([0.5])
        preds = pd.DataFrame({
            "a_health": [0.5],
            "b_health": [1.0],
            "diff_boost": [0.5],
        })
        adj_paired_only = [{
            "columns": ["a_health", "b_health"],
            "strength": 0.1,
            "default": 1.0,
            "mode": "paired",
        }]
        adj_both = [
            {
                "columns": ["a_health", "b_health"],
                "strength": 0.1,
                "default": 1.0,
                "mode": "paired",
            },
            {
                "columns": ["diff_boost"],
                "strength": 0.1,
                "default": 0.0,
                "mode": "diff",
            },
        ]
        result_one = apply_logit_adjustments(probs, preds, adj_paired_only)
        result_both = apply_logit_adjustments(probs, preds, adj_both)
        # The diff_boost is positive, so result_both should be higher than result_one
        assert result_both[0] > result_one[0]

    def test_empty_adjustments_no_change(self):
        """Empty adjustments list should produce no change."""
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({"anything": [1, 2]})
        result = apply_logit_adjustments(probs, preds, [])
        np.testing.assert_allclose(result, probs, atol=1e-7)

    def test_symmetry_of_paired_adjustment(self):
        """Swapping A and B health should produce complementary probabilities."""
        probs = np.array([0.6])
        preds_ab = pd.DataFrame({"a_h": [0.5], "b_h": [1.0]})
        preds_ba = pd.DataFrame({"a_h": [1.0], "b_h": [0.5]})
        adj = [{
            "columns": ["a_h", "b_h"],
            "strength": 0.2,
            "default": 1.0,
            "mode": "paired",
        }]
        result_ab = apply_logit_adjustments(probs, preds_ab, adj)
        result_ba = apply_logit_adjustments(1.0 - probs, preds_ba, adj)
        # result_ab[0] + result_ba[0] should equal 1.0 (symmetry in logit space)
        np.testing.assert_allclose(result_ab[0] + result_ba[0], 1.0, atol=1e-7)
