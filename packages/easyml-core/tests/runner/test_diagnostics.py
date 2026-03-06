"""Tests for diagnostics and metrics computation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.core.runner.diagnostics import (
    compute_brier_score,
    compute_calibration_curve,
    compute_ece,
    compute_pooled_metrics,
    evaluate_fold_predictions,
)


# -----------------------------------------------------------------------
# Tests: compute_brier_score
# -----------------------------------------------------------------------

class TestBrierScore:
    """Test Brier score computation."""

    def test_perfect_predictions(self):
        """Perfect predictions give Brier = 0."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        assert compute_brier_score(y_true, y_prob) == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Inverted predictions give Brier = 1."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        assert compute_brier_score(y_true, y_prob) == pytest.approx(1.0)

    def test_coin_flip_predictions(self):
        """Coin flip predictions give Brier = 0.25."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        assert compute_brier_score(y_true, y_prob) == pytest.approx(0.25)

    def test_matches_sklearn(self):
        """Brier score matches sklearn computation."""
        from sklearn.metrics import brier_score_loss

        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(float)
        y_prob = rng.uniform(0, 1, size=100)

        our_brier = compute_brier_score(y_true, y_prob)
        sklearn_brier = brier_score_loss(y_true, y_prob)
        assert our_brier == pytest.approx(sklearn_brier, abs=1e-10)

    def test_single_sample(self):
        """Works with a single sample."""
        assert compute_brier_score(np.array([1.0]), np.array([0.7])) == pytest.approx(0.09)


# -----------------------------------------------------------------------
# Tests: compute_ece
# -----------------------------------------------------------------------

class TestECE:
    """Test Expected Calibration Error computation."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions give ECE = 0."""
        # All predictions are 0.5, half are 1 and half are 0
        y_true = np.array([1.0, 0.0, 1.0, 0.0] * 25)
        y_prob = np.array([0.5] * 100)
        ece = compute_ece(y_true, y_prob, n_bins=10)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_ece_nonnegative(self):
        """ECE is always non-negative."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200).astype(float)
        y_prob = rng.uniform(0, 1, size=200)
        ece = compute_ece(y_true, y_prob)
        assert ece >= 0

    def test_ece_bounded(self):
        """ECE is bounded by 1."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200).astype(float)
        y_prob = rng.uniform(0, 1, size=200)
        ece = compute_ece(y_true, y_prob)
        assert ece <= 1.0

    def test_known_calibration(self):
        """Test with known miscalibration."""
        # All predicted 0.9 but true rate is 0.5 -> large ECE
        y_true = np.array([1.0, 0.0] * 50)
        y_prob = np.array([0.9] * 100)
        ece = compute_ece(y_true, y_prob, n_bins=10)
        # ECE should be approximately 0.4 (|0.9 - 0.5|)
        assert ece == pytest.approx(0.4, abs=0.05)

    def test_empty_input(self):
        """Empty arrays return ECE = 0."""
        assert compute_ece(np.array([]), np.array([]), n_bins=10) == 0.0


# -----------------------------------------------------------------------
# Tests: compute_calibration_curve
# -----------------------------------------------------------------------

class TestCalibrationCurve:
    """Test calibration curve computation."""

    def test_returns_three_arrays(self):
        """Returns (mean_predicted, mean_actual, bin_counts)."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(float)
        y_prob = rng.uniform(0, 1, size=100)

        mean_pred, mean_actual, counts = compute_calibration_curve(y_true, y_prob)
        assert len(mean_pred) == len(mean_actual) == len(counts)
        assert len(mean_pred) > 0

    def test_counts_sum_to_total(self):
        """Bin counts sum to total number of samples."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(float)
        y_prob = rng.uniform(0, 1, size=100)

        _, _, counts = compute_calibration_curve(y_true, y_prob)
        assert counts.sum() == 100

    def test_mean_predicted_in_range(self):
        """Mean predicted values are in [0, 1]."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(float)
        y_prob = rng.uniform(0, 1, size=100)

        mean_pred, _, _ = compute_calibration_curve(y_true, y_prob)
        assert np.all(mean_pred >= 0)
        assert np.all(mean_pred <= 1)

    def test_empty_bins_excluded(self):
        """Bins with no samples are excluded."""
        # All predictions clustered in a narrow range
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.45, 0.48, 0.52, 0.55])

        mean_pred, mean_actual, counts = compute_calibration_curve(
            y_true, y_prob, n_bins=10
        )
        # Most bins should be empty
        assert len(mean_pred) < 10


# -----------------------------------------------------------------------
# Tests: evaluate_fold_predictions
# -----------------------------------------------------------------------

class TestEvaluateFoldPredictions:
    """Test per-fold evaluation."""

    def test_returns_expected_structure(self):
        """Output has expected keys."""
        preds = pd.DataFrame({
            "result": [1, 0, 1, 0, 1] * 10,
            "prob_model_a": [0.7, 0.3, 0.8, 0.2, 0.6] * 10,
            "prob_model_b": [0.6, 0.4, 0.7, 0.3, 0.5] * 10,
        })
        results = evaluate_fold_predictions(preds, {}, fold_id=2024)
        assert len(results) == 2

        for r in results:
            assert "model" in r
            assert "fold" in r
            assert "accuracy" in r
            assert "brier_score" in r
            assert "ece" in r
            assert "log_loss" in r
            assert r["fold"] == 2024

    def test_model_names_extracted(self):
        """Model names are extracted from prob_ column names."""
        preds = pd.DataFrame({
            "result": [1, 0, 1, 0],
            "prob_xgb_core": [0.7, 0.3, 0.8, 0.2],
            "prob_logreg": [0.6, 0.4, 0.7, 0.3],
        })
        results = evaluate_fold_predictions(preds, {}, fold_id=2024)
        model_names = {r["model"] for r in results}
        assert model_names == {"xgb_core", "logreg"}

    def test_uses_actuals_dict_fallback(self):
        """Uses actuals dict when result column is missing."""
        preds = pd.DataFrame({
            "prob_model_a": [0.7, 0.3, 0.8, 0.2],
        })
        actuals = {"m1": 1, "m2": 0, "m3": 1, "m4": 0}
        results = evaluate_fold_predictions(preds, actuals, fold_id=2024)
        assert len(results) == 1

    def test_raises_without_ground_truth(self):
        """Raises if no result column and empty actuals."""
        preds = pd.DataFrame({
            "prob_model_a": [0.7, 0.3],
        })
        with pytest.raises(ValueError, match="No ground truth"):
            evaluate_fold_predictions(preds, {}, fold_id=2024)


# -----------------------------------------------------------------------
# Tests: compute_pooled_metrics
# -----------------------------------------------------------------------

class TestPooledMetrics:
    """Test pooled metrics across multiple folds."""

    def test_basic_pooling(self):
        """Pooled metrics combine data from multiple folds."""
        rng = np.random.default_rng(42)

        fold_dfs = []
        for season in [2023, 2024]:
            n = 50
            df = pd.DataFrame({
                "result": rng.integers(0, 2, size=n).astype(float),
                "prob_model_a": rng.uniform(0.2, 0.8, size=n),
                "prob_model_b": rng.uniform(0.3, 0.7, size=n),
            })
            fold_dfs.append(df)

        metrics = compute_pooled_metrics(fold_dfs)
        assert "model_a" in metrics
        assert "model_b" in metrics

        for model_name, m in metrics.items():
            assert "accuracy" in m
            assert "brier_score" in m
            assert "ece" in m
            assert "log_loss" in m
            assert "n_samples" in m
            assert m["n_samples"] == 100  # 50 per fold

    def test_empty_list(self):
        """Empty list returns empty dict."""
        assert compute_pooled_metrics([]) == {}

    def test_pooled_vs_averaged(self):
        """Pooled metrics can differ from averaged per-fold metrics."""
        # This is just a sanity check that pooling works differently
        df1 = pd.DataFrame({
            "result": [1.0, 1.0, 1.0],
            "prob_model": [0.9, 0.9, 0.9],
        })
        df2 = pd.DataFrame({
            "result": [0.0, 0.0, 0.0],
            "prob_model": [0.1, 0.1, 0.1],
        })

        metrics = compute_pooled_metrics([df1, df2])
        assert "model" in metrics
        # Good predictions -> low Brier
        assert metrics["model"]["brier_score"] < 0.05

    def test_missing_result_raises(self):
        """Raises if result column is missing."""
        df = pd.DataFrame({"prob_model": [0.5, 0.5]})
        with pytest.raises(ValueError, match="result"):
            compute_pooled_metrics([df])

    def test_single_fold(self):
        """Works with a single fold."""
        df = pd.DataFrame({
            "result": [1.0, 0.0, 1.0, 0.0],
            "prob_model": [0.7, 0.3, 0.8, 0.2],
        })
        metrics = compute_pooled_metrics([df])
        assert "model" in metrics
        assert metrics["model"]["n_samples"] == 4
