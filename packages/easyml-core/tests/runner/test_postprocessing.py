"""Tests for ensemble post-processing pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.core.runner.calibration import SplineCalibrator
from easyml.core.runner.meta_learner import StackedEnsemble
from easyml.core.runner.postprocessing import (
    apply_availability_adjustment,
    apply_ensemble_postprocessing,
    apply_seed_compression,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_preds_df(n: int = 50, n_models: int = 3, seed: int = 42) -> pd.DataFrame:
    """Create a predictions DataFrame with prob_* columns and diff_seed_num."""
    rng = np.random.RandomState(seed)
    model_names = [f"model_{i}" for i in range(n_models)]
    data = {}
    for name in model_names:
        data[f"prob_{name}"] = rng.beta(2, 2, size=n)
    # Add logreg_seed (should be excluded by pipeline)
    data["prob_logreg_seed"] = rng.beta(2, 2, size=n)
    data["diff_seed_num"] = rng.choice([-8, -4, -2, -1, 1, 2, 4, 8], size=n).astype(float)
    return pd.DataFrame(data)


def _fit_meta_learner(preds_df: pd.DataFrame, model_names: list[str]) -> StackedEnsemble:
    """Fit a meta-learner on the preds DataFrame for testing."""
    rng = np.random.RandomState(123)
    n = len(preds_df)
    y_true = rng.randint(0, 2, size=n).astype(float)

    model_preds = {name: preds_df[f"prob_{name}"].values for name in model_names}
    seed_diffs = preds_df["diff_seed_num"].values

    meta = StackedEnsemble(model_names)
    meta.fit(model_preds, seed_diffs, y_true)
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
            "availability_adjustment": 0.0,
            "seed_compression": 0.0,
            "seed_compression_threshold": 4,
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
            "availability_adjustment": 0.0,
            "seed_compression": 0.0,
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
            "availability_adjustment": 0.0,
            "seed_compression": 0.0,
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
            "availability_adjustment": 0.0,
            "seed_compression": 0.0,
        }
        result_t1 = apply_ensemble_postprocessing(preds_df, meta, None, config_t1)

        config_t2 = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 2.0,
            "clip_floor": 0.0,
            "availability_adjustment": 0.0,
            "seed_compression": 0.0,
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
            "availability_adjustment": 0.0,
            "seed_compression": 0.0,
        }

        result = apply_ensemble_postprocessing(preds_df, meta, None, config)
        assert result["prob_ensemble"].min() >= 0.1
        assert result["prob_ensemble"].max() <= 0.9

    def test_with_seed_compression(self):
        preds_df = _make_preds_df(50, 2)
        model_names = ["model_0", "model_1"]
        meta = _fit_meta_learner(preds_df, model_names)

        config_no_comp = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "availability_adjustment": 0.0,
            "seed_compression": 0.0,
        }
        result_no = apply_ensemble_postprocessing(preds_df, meta, None, config_no_comp)

        config_comp = {
            "exclude_models": [],
            "meta_features": [],
            "temperature": 1.0,
            "clip_floor": 0.0,
            "availability_adjustment": 0.0,
            "seed_compression": 0.3,
            "seed_compression_threshold": 4,
        }
        result_comp = apply_ensemble_postprocessing(preds_df, meta, None, config_comp)

        # Close seed matchups should be more compressed toward 0.5
        close_mask = np.abs(preds_df["diff_seed_num"]) <= 4
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
            "availability_adjustment": 0.0,
            "seed_compression": 0.0,
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
            "availability_adjustment": 0.0,
            "seed_compression": 0.0,
        }

        result_with = apply_ensemble_postprocessing(preds_df, meta, calibrator, config)
        result_without = apply_ensemble_postprocessing(preds_df, meta, None, config)

        # Post-calibration should change the output
        assert not np.allclose(
            result_with["prob_ensemble"].values,
            result_without["prob_ensemble"].values,
        )


# -----------------------------------------------------------------------
# Individual steps
# -----------------------------------------------------------------------

class TestApplySeedCompression:
    def test_compression_toward_half(self):
        probs = np.array([0.8, 0.2, 0.9, 0.1])
        preds = pd.DataFrame({
            "diff_seed_num": [2, 2, 10, 10],  # first two are close, last two are far
        })
        result = apply_seed_compression(probs, preds, compression=0.5, threshold=4)

        # Close matchups (seed_diff=2) should be compressed toward 0.5
        assert abs(result[0] - 0.65) < 1e-10  # 0.8 * 0.5 + 0.5 * 0.5 = 0.65
        assert abs(result[1] - 0.35) < 1e-10  # 0.2 * 0.5 + 0.5 * 0.5 = 0.35
        # Far matchups (seed_diff=10) should be unchanged
        assert abs(result[2] - 0.9) < 1e-10
        assert abs(result[3] - 0.1) < 1e-10

    def test_no_compression(self):
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({"diff_seed_num": [2, 2]})
        result = apply_seed_compression(probs, preds, compression=0.0, threshold=4)
        np.testing.assert_allclose(result, probs)

    def test_full_compression(self):
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({"diff_seed_num": [2, 2]})
        result = apply_seed_compression(probs, preds, compression=1.0, threshold=4)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_missing_seed_column(self):
        """Without diff_seed_num column, should return unchanged."""
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({"other_col": [1, 2]})
        result = apply_seed_compression(probs, preds, compression=0.5, threshold=4)
        np.testing.assert_allclose(result, probs)

    def test_negative_seed_diffs(self):
        """Negative seed diffs should also be handled (abs value)."""
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({"diff_seed_num": [-2, -2]})
        result = apply_seed_compression(probs, preds, compression=0.5, threshold=4)
        assert abs(result[0] - 0.65) < 1e-10
        assert abs(result[1] - 0.35) < 1e-10


class TestApplyAvailabilityAdjustment:
    def test_no_adjustment_all_models_present(self):
        """All prob_* columns have valid values -> no change."""
        probs = np.array([0.8, 0.2, 0.5])
        preds = pd.DataFrame({
            "prob_model_a": [0.7, 0.3, 0.6],
            "prob_model_b": [0.9, 0.1, 0.4],
            "prob_model_c": [0.6, 0.4, 0.5],
        })
        config = {"exclude_models": []}
        result = apply_availability_adjustment(probs, preds, config, strength=0.5)
        np.testing.assert_allclose(result, probs)

    def test_pulls_toward_half_when_models_missing(self):
        """Setting some prob_* to NaN should pull output toward 0.5."""
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({
            "prob_model_a": [0.7, np.nan],
            "prob_model_b": [0.9, np.nan],
            "prob_model_c": [0.6, 0.5],
        })
        config = {"exclude_models": []}
        result = apply_availability_adjustment(probs, preds, config, strength=0.6)
        # Row 0: all 3 present -> coverage=1.0, pull_factor=0.0 -> no change
        np.testing.assert_allclose(result[0], 0.8)
        # Row 1: 1 of 3 present -> coverage=1/3, pull_factor=0.6*(2/3)=0.4
        # adjusted = 0.2 * 0.6 + 0.5 * 0.4 = 0.32
        np.testing.assert_allclose(result[1], 0.32)

    def test_zero_strength_no_change(self):
        """strength=0.0 should return unchanged probabilities."""
        probs = np.array([0.9, 0.1, 0.7])
        preds = pd.DataFrame({
            "prob_model_a": [0.7, np.nan, 0.6],
            "prob_model_b": [np.nan, np.nan, 0.4],
        })
        config = {"exclude_models": []}
        result = apply_availability_adjustment(probs, preds, config, strength=0.0)
        np.testing.assert_allclose(result, probs)

    def test_full_strength_all_missing(self):
        """strength=1.0 and all models NaN should fully pull to 0.5."""
        probs = np.array([0.9, 0.1])
        preds = pd.DataFrame({
            "prob_model_a": [np.nan, np.nan],
            "prob_model_b": [np.nan, np.nan],
        })
        config = {"exclude_models": []}
        result = apply_availability_adjustment(probs, preds, config, strength=1.0)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_partial_missing_some_games(self):
        """Only some rows have NaN - verify per-row behavior."""
        probs = np.array([0.8, 0.6, 0.3, 0.9])
        preds = pd.DataFrame({
            "prob_model_a": [0.7, np.nan, 0.5, 0.8],
            "prob_model_b": [0.9, 0.4, np.nan, 0.7],
            "prob_model_c": [0.6, 0.3, np.nan, np.nan],
            "prob_model_d": [0.5, np.nan, 0.6, 0.9],
        })
        config = {"exclude_models": []}
        strength = 1.0
        result = apply_availability_adjustment(probs, preds, config, strength)

        # Row 0: 4/4 present -> coverage=1.0, pull=0.0 -> 0.8
        np.testing.assert_allclose(result[0], 0.8)
        # Row 1: 2/4 present -> coverage=0.5, pull=0.5 -> 0.6*0.5 + 0.5*0.5 = 0.55
        np.testing.assert_allclose(result[1], 0.55)
        # Row 2: 2/4 present -> coverage=0.5, pull=0.5 -> 0.3*0.5 + 0.5*0.5 = 0.4
        np.testing.assert_allclose(result[2], 0.4)
        # Row 3: 3/4 present -> coverage=0.75, pull=0.25 -> 0.9*0.75 + 0.5*0.25 = 0.8
        np.testing.assert_allclose(result[3], 0.8)

    def test_excluded_models_not_counted(self):
        """Excluded models should not be counted in coverage calculation."""
        probs = np.array([0.8])
        preds = pd.DataFrame({
            "prob_model_a": [0.7],
            "prob_model_b": [np.nan],  # excluded, should not matter
        })
        config = {"exclude_models": ["model_b"]}
        result = apply_availability_adjustment(probs, preds, config, strength=1.0)
        # Only model_a is active, and it is present -> coverage=1.0 -> no change
        np.testing.assert_allclose(result[0], 0.8)

    def test_ensemble_column_excluded(self):
        """prob_ensemble should not count as a model column."""
        probs = np.array([0.8])
        preds = pd.DataFrame({
            "prob_model_a": [np.nan],
            "prob_model_b": [0.5],
            "prob_ensemble": [0.6],
        })
        config = {"exclude_models": []}
        result = apply_availability_adjustment(probs, preds, config, strength=1.0)
        # model_a (NaN) + model_b (present) -> coverage=0.5, pull=0.5 -> 0.65
        np.testing.assert_allclose(result[0], 0.65)

    def test_no_active_models_returns_unchanged(self):
        """Edge case: no active model columns -> return probs unchanged."""
        probs = np.array([0.8, 0.2])
        preds = pd.DataFrame({
            "prob_ensemble": [0.5, 0.5],
            "some_other_col": [1, 2],
        })
        config = {"exclude_models": []}
        result = apply_availability_adjustment(probs, preds, config, strength=1.0)
        np.testing.assert_allclose(result, probs)
