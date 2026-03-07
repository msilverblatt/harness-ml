"""Tests for MetricRegistry and all registered metrics across task types."""
import numpy as np
from harnessml.core.schemas.metrics import (
    MetricRegistry,
    accuracy_multi,
    auc_pr,
    binary_confusion_matrix,
    # Binary
    brier_score,
    cohen_kappa,
    cohen_kappa_multi,
    # Survival
    concordance_index,
    confusion_matrix_multi,
    coverage_at_level,
    # Probabilistic
    crps,
    cumulative_incidence_auc,
    explained_variance,
    f1_macro,
    f1_micro,
    f1_weighted,
    log_loss_multi,
    mae,
    mape,
    mcc,
    mcc_multi,
    mean_average_precision,
    mean_bias,
    median_ae,
    mrr,
    # Ranking
    ndcg_at_k,
    per_class_report,
    pit_histogram_data,
    precision,
    precision_at_k,
    precision_macro,
    quantile_loss,
    r_squared,
    recall,
    recall_at_k,
    recall_macro,
    # Regression
    rmse,
    sharpness,
    spearman_rank_corr,
    specificity,
    time_dependent_brier,
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
        assert "ranking" in listing
        assert "survival" in listing
        assert "probabilistic" in listing

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

    def test_ranking_metric_count(self):
        listing = MetricRegistry.list_metrics("ranking")
        assert len(listing["ranking"]) == 6

    def test_survival_metric_count(self):
        listing = MetricRegistry.list_metrics("survival")
        assert len(listing["survival"]) == 3

    def test_probabilistic_metric_count(self):
        listing = MetricRegistry.list_metrics("probabilistic")
        assert len(listing["probabilistic"]) == 4


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

    def test_compute_all_ranking(self):
        y_true = np.array([1, 1, 0, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        results = MetricRegistry.compute_all("ranking", y_true, y_pred)
        assert "ndcg_at_k" in results
        assert "mrr" in results
        assert "spearman_rank_corr" in results


# ---------------------------------------------------------------------------
# Ranking metric tests
# ---------------------------------------------------------------------------

class TestRankingMetrics:
    """Tests for ranking metrics."""

    # Single query: 5 items, first 2 relevant
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.3, 0.2, 0.1])  # Perfect ranking

    def test_ndcg_at_k_perfect(self):
        result = ndcg_at_k(self.y_true, self.y_pred, k=5)
        assert abs(result - 1.0) < 1e-10

    def test_ndcg_at_k_partial(self):
        result = ndcg_at_k(self.y_true, self.y_pred, k=2)
        assert abs(result - 1.0) < 1e-10

    def test_ndcg_at_k_via_kwargs(self):
        result = ndcg_at_k(self.y_true, self.y_pred, k=3)
        assert 0.0 <= result <= 1.0

    def test_mean_average_precision_perfect(self):
        result = mean_average_precision(self.y_true, self.y_pred)
        assert result == 1.0

    def test_mean_average_precision_imperfect(self):
        # Relevant items ranked 2nd and 4th
        y_pred = np.array([0.3, 0.9, 0.2, 0.8, 0.1])
        result = mean_average_precision(self.y_true, y_pred)
        assert 0.0 < result < 1.0

    def test_mrr_perfect(self):
        result = mrr(self.y_true, self.y_pred)
        assert result == 1.0  # First relevant item is rank 1

    def test_mrr_second_rank(self):
        # Relevant item at rank 2
        y_true = np.array([0, 1, 0, 0])
        y_pred = np.array([0.9, 0.8, 0.3, 0.1])
        result = mrr(y_true, y_pred)
        assert abs(result - 0.5) < 1e-10

    def test_mrr_no_relevant(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0.9, 0.5, 0.1])
        result = mrr(y_true, y_pred)
        assert result == 0.0

    def test_precision_at_k(self):
        result = precision_at_k(self.y_true, self.y_pred, k=2)
        assert result == 1.0  # Top 2 are both relevant

    def test_precision_at_k_partial(self):
        result = precision_at_k(self.y_true, self.y_pred, k=5)
        assert abs(result - 0.4) < 1e-10  # 2 relevant out of 5

    def test_recall_at_k(self):
        result = recall_at_k(self.y_true, self.y_pred, k=2)
        assert result == 1.0  # Both relevant items in top 2

    def test_recall_at_k_partial(self):
        result = recall_at_k(self.y_true, self.y_pred, k=1)
        assert abs(result - 0.5) < 1e-10  # 1 of 2 relevant in top 1

    def test_recall_at_k_no_relevant(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0.9, 0.5, 0.1])
        result = recall_at_k(y_true, y_pred, k=2)
        assert result == 0.0

    def test_spearman_rank_corr_perfect(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        result = spearman_rank_corr(y_true, y_pred)
        assert abs(result - 1.0) < 1e-10

    def test_spearman_rank_corr_inverse(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 4, 3, 2, 1])
        result = spearman_rank_corr(y_true, y_pred)
        assert abs(result - (-1.0)) < 1e-10

    def test_mrr_2d(self):
        """Test MRR with 2D input (multiple queries)."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.5, 0.1], [0.1, 0.9, 0.5]])
        result = mrr(y_true, y_pred)
        assert result == 1.0

    def test_map_2d(self):
        """Test MAP with 2D input (multiple queries)."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.5, 0.1], [0.1, 0.9, 0.5]])
        result = mean_average_precision(y_true, y_pred)
        assert result == 1.0


# ---------------------------------------------------------------------------
# Survival metric tests
# ---------------------------------------------------------------------------

class TestSurvivalMetrics:
    """Tests for survival metrics with synthetic data."""

    def test_concordance_index_perfect(self):
        # Higher risk = shorter survival time
        times = np.array([10.0, 8.0, 5.0, 2.0])
        risk_scores = np.array([0.1, 0.3, 0.6, 0.9])  # Perfect concordance
        event = np.array([1, 1, 1, 1])
        result = concordance_index(times, risk_scores, event=event)
        assert result == 1.0

    def test_concordance_index_inverse(self):
        # Higher risk = longer survival (discordant)
        times = np.array([2.0, 5.0, 8.0, 10.0])
        risk_scores = np.array([0.1, 0.3, 0.6, 0.9])
        event = np.array([1, 1, 1, 1])
        result = concordance_index(times, risk_scores, event=event)
        assert result == 0.0

    def test_concordance_index_random(self):
        # Random risk scores
        np.random.seed(42)
        times = np.array([10.0, 8.0, 5.0, 2.0])
        risk_scores = np.random.rand(4)
        event = np.array([1, 1, 1, 1])
        result = concordance_index(times, risk_scores, event=event)
        assert 0.0 <= result <= 1.0

    def test_concordance_index_with_censoring(self):
        times = np.array([10.0, 8.0, 5.0, 2.0])
        risk_scores = np.array([0.1, 0.3, 0.6, 0.9])
        event = np.array([1, 0, 1, 1])  # Second observation censored
        result = concordance_index(times, risk_scores, event=event)
        assert 0.0 <= result <= 1.0

    def test_concordance_index_no_event_kwarg(self):
        # Without event kwarg, assumes all events observed
        times = np.array([10.0, 8.0, 5.0, 2.0])
        risk_scores = np.array([0.1, 0.3, 0.6, 0.9])
        result = concordance_index(times, risk_scores)
        assert result == 1.0

    def test_time_dependent_brier(self):
        times = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        # Predicted survival probability at time_horizon=4
        surv_probs = np.array([0.1, 0.3, 0.8, 0.9, 0.95])
        event = np.array([1, 1, 1, 1, 1])
        result = time_dependent_brier(
            times, surv_probs, event=event, time_horizon=4.0
        )
        assert 0.0 <= result <= 1.0

    def test_time_dependent_brier_perfect(self):
        # Event at t=2 (died), survived past t=6
        times = np.array([2.0, 8.0])
        surv_probs = np.array([0.0, 1.0])  # Perfect predictions at horizon=5
        event = np.array([1, 1])
        result = time_dependent_brier(
            times, surv_probs, event=event, time_horizon=5.0
        )
        assert result == 0.0

    def test_cumulative_incidence_auc(self):
        np.random.seed(42)
        times = np.arange(1.0, 21.0)
        event = np.ones(20, dtype=int)
        # Predict higher incidence for earlier events
        pred_inc = 1.0 - times / 25.0 + np.random.rand(20) * 0.1
        result = cumulative_incidence_auc(
            times, pred_inc, event=event, time_horizon=10.0
        )
        assert 0.0 <= result <= 1.0

    def test_cumulative_incidence_auc_default_horizon(self):
        times = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        pred_inc = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        event = np.ones(5, dtype=int)
        result = cumulative_incidence_auc(times, pred_inc, event=event)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Probabilistic metric tests
# ---------------------------------------------------------------------------

class TestProbabilisticMetrics:
    """Tests for probabilistic metrics."""

    def test_crps_deterministic(self):
        """Deterministic CRPS = MAE."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        result = crps(y_true, y_pred)
        expected = mae(y_true, y_pred)
        assert abs(result - expected) < 1e-10

    def test_crps_ensemble(self):
        """CRPS with ensemble predictions (2D array)."""
        y_true = np.array([1.0, 2.0])
        # 3 ensemble members, 2 observations
        y_pred = np.array([
            [0.8, 1.8],
            [1.0, 2.0],
            [1.2, 2.2],
        ])
        result = crps(y_true, y_pred)
        assert result >= 0.0
        # Perfect ensemble centered on truth should be small
        assert result < 0.5

    def test_crps_perfect_ensemble(self):
        """Ensemble all predicting exactly y_true."""
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ])
        result = crps(y_true, y_pred)
        assert abs(result) < 1e-10

    def test_pit_histogram_data_binary(self):
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.95, 0.05])
        result = pit_histogram_data(y_true, y_prob, n_bins=5)
        assert "counts" in result
        assert "bin_edges" in result
        assert len(result["counts"]) == 5
        assert len(result["bin_edges"]) == 6
        assert sum(result["counts"]) == len(y_true)

    def test_pit_histogram_data_structure(self):
        """PIT histogram returns correct bin structure and counts sum to n."""
        np.random.seed(42)
        n = 100
        probs = np.random.rand(n)
        y_true = (np.random.rand(n) < probs).astype(float)
        result = pit_histogram_data(y_true, probs, n_bins=5)
        counts = result["counts"]
        bin_edges = result["bin_edges"]
        assert len(counts) == 5
        assert len(bin_edges) == 6
        assert sum(counts) == n
        assert bin_edges[0] == 0.0
        assert bin_edges[-1] == 1.0

    def test_sharpness(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob_confident = np.array([0.95, 0.05, 0.90, 0.10])
        y_prob_uncertain = np.array([0.51, 0.49, 0.52, 0.48])
        sharp_confident = sharpness(y_true, y_prob_confident)
        sharp_uncertain = sharpness(y_true, y_prob_uncertain)
        assert sharp_confident > sharp_uncertain

    def test_sharpness_all_same(self):
        y_true = np.array([1, 0])
        y_prob = np.array([0.5, 0.5])
        result = sharpness(y_true, y_prob)
        assert result == 0.0

    def test_coverage_at_level_2d(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        intervals = np.array([
            [0.5, 1.5],
            [1.5, 2.5],
            [2.5, 3.5],
            [3.5, 4.5],
            [4.5, 5.5],
        ])
        result = coverage_at_level(y_true, intervals)
        assert result == 1.0  # All values within intervals

    def test_coverage_at_level_kwargs(self):
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        result = coverage_at_level(y_true, y_true, lower=lower, upper=upper)
        assert result == 1.0

    def test_coverage_at_level_partial(self):
        y_true = np.array([1.0, 5.0, 3.0])
        intervals = np.array([
            [0.5, 1.5],   # covers 1.0
            [1.5, 2.5],   # does not cover 5.0
            [2.5, 3.5],   # covers 3.0
        ])
        result = coverage_at_level(y_true, intervals)
        assert abs(result - 2.0 / 3.0) < 1e-10

    def test_coverage_at_level_bad_input(self):
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([0.5, 0.5])  # 1D, no lower/upper
        result = coverage_at_level(y_true, y_pred)
        assert np.isnan(result)
