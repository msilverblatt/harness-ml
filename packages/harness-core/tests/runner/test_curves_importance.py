"""Tests for ROC/PR curves and permutation importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.analysis.diagnostics import (
    permutation_importance_data,
    pr_curve_data,
    roc_curve_data,
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _synth_binary_data(n: int = 200, seed: int = 42):
    """Generate synthetic binary classification data and predictions."""
    rng = np.random.RandomState(seed)
    true_p = rng.beta(2, 2, size=n)
    y_true = (rng.rand(n) < true_p).astype(float)
    y_prob = np.clip(true_p + rng.normal(0, 0.1, n), 0.01, 0.99)
    return y_true, y_prob


def _make_trained_model():
    """Train a simple model for permutation importance."""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "informative": rng.randn(200),
        "noise": rng.randn(200),
    })
    y = (X["informative"] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X, y


# -----------------------------------------------------------------------
# ROC Curve
# -----------------------------------------------------------------------

class TestROCCurveData:
    def test_returns_expected_keys(self):
        y_true, y_prob = _synth_binary_data()
        result = roc_curve_data(y_true, y_prob)
        assert "fpr" in result
        assert "tpr" in result
        assert "thresholds" in result
        assert "auc" in result

    def test_auc_between_0_and_1(self):
        y_true, y_prob = _synth_binary_data()
        result = roc_curve_data(y_true, y_prob)
        assert 0.0 <= result["auc"] <= 1.0

    def test_perfect_predictions_auc_1(self):
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = roc_curve_data(y_true, y_prob)
        assert result["auc"] == pytest.approx(1.0)

    def test_random_predictions_auc_near_05(self):
        rng = np.random.RandomState(42)
        n = 10000
        y_true = rng.randint(0, 2, size=n).astype(float)
        y_prob = rng.uniform(0, 1, size=n)
        result = roc_curve_data(y_true, y_prob)
        # Random predictions should give AUC near 0.5
        assert abs(result["auc"] - 0.5) < 0.05

    def test_fpr_tpr_are_lists(self):
        y_true, y_prob = _synth_binary_data()
        result = roc_curve_data(y_true, y_prob)
        assert isinstance(result["fpr"], list)
        assert isinstance(result["tpr"], list)

    def test_fpr_tpr_bounded(self):
        y_true, y_prob = _synth_binary_data()
        result = roc_curve_data(y_true, y_prob)
        assert all(0.0 <= v <= 1.0 for v in result["fpr"])
        assert all(0.0 <= v <= 1.0 for v in result["tpr"])


# -----------------------------------------------------------------------
# PR Curve
# -----------------------------------------------------------------------

class TestPRCurveData:
    def test_returns_expected_keys(self):
        y_true, y_prob = _synth_binary_data()
        result = pr_curve_data(y_true, y_prob)
        assert "precision" in result
        assert "recall" in result
        assert "thresholds" in result
        assert "average_precision" in result

    def test_average_precision_between_0_and_1(self):
        y_true, y_prob = _synth_binary_data()
        result = pr_curve_data(y_true, y_prob)
        assert 0.0 <= result["average_precision"] <= 1.0

    def test_precision_recall_bounded(self):
        y_true, y_prob = _synth_binary_data()
        result = pr_curve_data(y_true, y_prob)
        assert all(0.0 <= v <= 1.0 for v in result["precision"])
        assert all(0.0 <= v <= 1.0 for v in result["recall"])

    def test_precision_recall_are_lists(self):
        y_true, y_prob = _synth_binary_data()
        result = pr_curve_data(y_true, y_prob)
        assert isinstance(result["precision"], list)
        assert isinstance(result["recall"], list)

    def test_good_model_high_avg_precision(self):
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = pr_curve_data(y_true, y_prob)
        assert result["average_precision"] == pytest.approx(1.0)


# -----------------------------------------------------------------------
# Permutation Importance
# -----------------------------------------------------------------------

class TestPermutationImportanceData:
    def test_returns_expected_keys(self):
        model, X, y = _make_trained_model()
        result = permutation_importance_data(model, X, y, n_repeats=5)
        assert "importances_mean" in result
        assert "importances_std" in result
        assert "feature_names" in result

    def test_feature_names_from_dataframe(self):
        model, X, y = _make_trained_model()
        result = permutation_importance_data(model, X, y, n_repeats=5)
        assert result["feature_names"] == ["informative", "noise"]

    def test_feature_names_from_array(self):
        model, X, y = _make_trained_model()
        result = permutation_importance_data(model, X.values, y, n_repeats=5)
        assert result["feature_names"] == ["feature_0", "feature_1"]

    def test_importances_correct_length(self):
        model, X, y = _make_trained_model()
        result = permutation_importance_data(model, X, y, n_repeats=5)
        assert len(result["importances_mean"]) == 2
        assert len(result["importances_std"]) == 2

    def test_informative_feature_higher_importance(self):
        """The informative feature should have higher importance than noise."""
        model, X, y = _make_trained_model()
        result = permutation_importance_data(model, X, y, n_repeats=10)
        idx_informative = result["feature_names"].index("informative")
        idx_noise = result["feature_names"].index("noise")
        assert result["importances_mean"][idx_informative] > result["importances_mean"][idx_noise]

    def test_importances_are_lists(self):
        model, X, y = _make_trained_model()
        result = permutation_importance_data(model, X, y, n_repeats=5)
        assert isinstance(result["importances_mean"], list)
        assert isinstance(result["importances_std"], list)

    def test_std_nonnegative(self):
        model, X, y = _make_trained_model()
        result = permutation_importance_data(model, X, y, n_repeats=5)
        assert all(s >= 0 for s in result["importances_std"])
