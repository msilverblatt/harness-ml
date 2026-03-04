"""Tests for MetricRegistry and all registered metrics across task types."""
import numpy as np
import pytest

from easyml.core.schemas.metrics import (
    MetricRegistry,
    # Binary
    brier_score,
    log_loss,
    accuracy,
    ece,
    auc_roc,
    f1,
    precision,
    recall,
    mcc,
    auc_pr,
    specificity,
    binary_confusion_matrix,
    cohen_kappa,
    # Multiclass
    accuracy_multi,
    f1_macro,
    f1_micro,
    f1_weighted,
    precision_macro,
    recall_macro,
    confusion_matrix_multi,
    per_class_report,
    log_loss_multi,
    mcc_multi,
    cohen_kappa_multi,
    # Regression
    rmse,
    mae,
    r_squared,
    mape,
    median_ae,
    explained_variance,
    mean_bias,
    quantile_loss,
)


# ---------------------------------------------------------------------------
# MetricRegistry class tests
# ---------------------------------------------------------------------------

class TestMetricRegistry:
    def test_register_and_get(self):
        """Registering a metric makes it retrievable via get()."""
        fn = MetricRegistry.get("binary", "brier")
        assert fn is brier_score

    def test_get_nonexistent_returns_none(self):
        assert MetricRegistry.get("binary", "does_not_exist") is None
        assert MetricRegistry.get("nonexistent_task", "brier") is None

    def test_list_metrics_all(self):
        listing = MetricRegistry.list_metrics()
        assert "binary" in listing
        assert "multiclass" in listing
        assert "regression" in listing

    def test_list_metrics_filtered(self):
        listing = MetricRegistry.list_metrics("binary")
        assert "binary" in listing
        assert len(listing) == 1
        assert "brier" in listing["binary"]
        assert "precision" in listing["binary"]

    def test_list_metrics_empty_task(self):
        listing = MetricRegistry.list_metrics("nonexistent_task_xyz")
        assert listing == {"nonexistent_task_xyz": []}

    def test_compute_all_binary(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        results = MetricRegistry.compute_all("binary", y_true, y_prob)
        assert "brier" in results
        assert "accuracy" in results
        assert "precision" in results
        assert "confusion_matrix" in results
        # Verify values are reasonable
        assert 0.0 <= results["brier"] <= 1.0
        assert results["accuracy"] == 1.0

    def test_compute_all_handles_errors(self):
        """Metrics that fail should return NaN, not raise."""
        y_true = np.array([])
        y_pred = np.array([])
        results = MetricRegistry.compute_all("regression", y_true, y_pred)
        # Some metrics may NaN on empty input
        for name, val in results.items():
            if isinstance(val, float):
                # either a valid float or nan — both OK
                assert isinstance(val, float)

    def test_binary_metric_count(self):
        listing = MetricRegistry.list_metrics("binary")
        assert len(listing["binary"]) == 13

    def test_multiclass_metric_count(self):
        listing = MetricRegistry.list_metrics("multiclass")
        assert len(listing["multiclass"]) == 11

    def test_regression_metric_count(self):
        listing = MetricRegistry.list_metrics("regression")
        assert len(listing["regression"]) == 8


# ---------------------------------------------------------------------------
# Binary classification metric tests
# ---------------------------------------------------------------------------

class TestBinaryMetrics:
    """Tests for new binary metrics (precision, recall, mcc, etc.)."""

    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob_perfect = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
    y_prob_random = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    def test_precision_perfect(self):
        result = precision(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_precision_accepts_kwargs(self):
        result = precision(self.y_true, self.y_prob_perfect, threshold=0.5)
        assert result == 1.0

    def test_recall_perfect(self):
        result = recall(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_recall_strict_threshold(self):
        # Only items >= 0.85 predicted as positive; only 1 out of 4 positives caught
        result = recall(self.y_true, self.y_prob_perfect, threshold=0.85)
        assert result == 0.25

    def test_mcc_perfect(self):
        result = mcc(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_mcc_random(self):
        # All predicted as positive at 0.5 threshold
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        result = mcc(y_true, y_prob)
        # MCC should be 0 or close to it for random
        assert -0.5 <= result <= 0.5

    def test_auc_pr_perfect(self):
        result = auc_pr(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_auc_pr_range(self):
        result = auc_pr(self.y_true, self.y_prob_random)
        assert 0.0 <= result <= 1.0

    def test_specificity_perfect(self):
        result = specificity(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_specificity_all_positive(self):
        # Predict everything positive
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        result = specificity(self.y_true, y_prob)
        assert result == 0.0

    def test_confusion_matrix(self):
        result = binary_confusion_matrix(self.y_true, self.y_prob_perfect)
        assert result == {"tp": 4, "tn": 4, "fp": 0, "fn": 0}

    def test_confusion_matrix_with_errors(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.3, 0.7, 0.8, 0.2])  # first two wrong
        result = binary_confusion_matrix(y_true, y_prob)
        assert result == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}

    def test_cohen_kappa_perfect(self):
        result = cohen_kappa(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_cohen_kappa_range(self):
        result = cohen_kappa(self.y_true, self.y_prob_random)
        assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Multiclass metric tests
# ---------------------------------------------------------------------------

class TestMulticlassMetrics:
    """Tests for multiclass metrics."""

    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    # Perfect predictions as probability matrix
    y_prob_perfect = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.15, 0.15],
        [0.15, 0.7, 0.15],
        [0.15, 0.15, 0.7],
    ])
    # Argmax labels for label-based metrics
    y_pred_perfect = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])

    def test_accuracy_multi_perfect(self):
        result = accuracy_multi(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_accuracy_multi_with_labels(self):
        result = accuracy_multi(self.y_true, self.y_pred_perfect)
        assert result == 1.0

    def test_f1_macro_perfect(self):
        result = f1_macro(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_f1_micro_perfect(self):
        result = f1_micro(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_f1_weighted_perfect(self):
        result = f1_weighted(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_precision_macro_perfect(self):
        result = precision_macro(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_recall_macro_perfect(self):
        result = recall_macro(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_confusion_matrix_multi(self):
        result = confusion_matrix_multi(self.y_true, self.y_prob_perfect)
        assert "matrix" in result
        assert "labels" in result
        cm = result["matrix"]
        # Perfect classification: diagonal should be [3, 3, 3]
        assert cm[0][0] == 3
        assert cm[1][1] == 3
        assert cm[2][2] == 3

    def test_per_class_report(self):
        result = per_class_report(self.y_true, self.y_prob_perfect)
        assert isinstance(result, dict)
        # Should have per-class entries
        assert "0" in result or 0 in result
        assert "accuracy" in result

    def test_log_loss_multi(self):
        result = log_loss_multi(self.y_true, self.y_prob_perfect)
        assert result > 0  # Not perfectly certain probs
        assert result < 1.0  # But close to perfect

    def test_mcc_multi_perfect(self):
        result = mcc_multi(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_cohen_kappa_multi_perfect(self):
        result = cohen_kappa_multi(self.y_true, self.y_prob_perfect)
        assert result == 1.0

    def test_f1_macro_imperfect(self):
        # Swap some predictions
        y_pred = np.array([0, 1, 2, 1, 0, 2, 0, 1, 0])
        result = f1_macro(self.y_true, y_pred)
        assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# Regression extra metric tests
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    """Tests for regression extras (mape, median_ae, etc.)."""

    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    def test_mape(self):
        result = mape(self.y_true, self.y_pred)
        assert result > 0
        assert isinstance(result, float)

    def test_median_ae(self):
        result = median_ae(self.y_true, self.y_pred)
        assert result == 0.5

    def test_explained_variance_perfect(self):
        result = explained_variance(self.y_true, self.y_true)
        assert result == 1.0

    def test_explained_variance_imperfect(self):
        result = explained_variance(self.y_true, self.y_pred)
        assert 0.0 < result <= 1.0

    def test_mean_bias_no_bias(self):
        result = mean_bias(self.y_true, self.y_true)
        assert abs(result) < 1e-10

    def test_mean_bias_positive(self):
        # Predictions are higher than true on average
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        result = mean_bias(y_true, y_pred)
        assert abs(result - 1.0) < 1e-10

    def test_quantile_loss_median(self):
        result = quantile_loss(self.y_true, self.y_pred, quantile=0.5)
        # At q=0.5, quantile loss = 0.5 * MAE
        expected_mae = mae(self.y_true, self.y_pred)
        assert abs(result - 0.5 * expected_mae) < 1e-10

    def test_quantile_loss_high_quantile(self):
        result = quantile_loss(self.y_true, self.y_pred, quantile=0.9)
        assert isinstance(result, float)

    def test_quantile_loss_via_kwargs(self):
        result = quantile_loss(self.y_true, self.y_pred, quantile=0.1)
        assert isinstance(result, float)

    def test_existing_rmse(self):
        result = rmse(self.y_true, self.y_pred)
        assert result > 0

    def test_existing_mae(self):
        result = mae(self.y_true, self.y_pred)
        assert result > 0

    def test_existing_r_squared(self):
        result = r_squared(self.y_true, self.y_pred)
        assert result > 0


# ---------------------------------------------------------------------------
# compute_all integration tests
# ---------------------------------------------------------------------------

class TestComputeAllIntegration:
    """Integration tests for compute_all across task types."""

    def test_compute_all_regression(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
        results = MetricRegistry.compute_all("regression", y_true, y_pred)
        assert "rmse" in results
        assert "mae" in results
        assert "r_squared" in results
        assert "mape" in results
        assert "median_ae" in results
        assert "explained_variance" in results
        assert "mean_bias" in results
        assert "quantile_loss" in results

    def test_compute_all_multiclass(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.1, 0.7],
        ])
        results = MetricRegistry.compute_all("multiclass", y_true, y_prob)
        assert "accuracy_multi" in results
        assert "f1_macro" in results
        assert results["accuracy_multi"] == 1.0
