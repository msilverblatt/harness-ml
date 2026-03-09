"""Tests for SHAP value computation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")

from harnessml.core.runner.analysis.diagnostics import compute_shap_values

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_simple_tree_model():
    """Train a simple decision tree for testing."""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "feature_a": rng.randn(100),
        "feature_b": rng.randn(100),
        "feature_c": rng.randn(100),
    })
    y = (X["feature_a"] + X["feature_b"] > 0).astype(int)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    return model, X, y


def _make_simple_linear_model():
    """Train a simple logistic regression for testing."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(42)
    X = pd.DataFrame({
        "feature_a": rng.randn(100),
        "feature_b": rng.randn(100),
    })
    y = (X["feature_a"] + X["feature_b"] > 0).astype(int)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model, X, y


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestComputeShapValues:
    def test_tree_method(self):
        model, X, _ = _make_simple_tree_model()
        result = compute_shap_values(model, X, method="tree")

        assert "values" in result
        assert "feature_names" in result
        assert "base_value" in result
        assert result["feature_names"] == ["feature_a", "feature_b", "feature_c"]
        assert np.isfinite(result["base_value"])

    def test_auto_method_with_tree(self):
        """Auto method should use TreeExplainer for tree-based models."""
        model, X, _ = _make_simple_tree_model()
        result = compute_shap_values(model, X, method="auto")

        assert "values" in result
        assert "error" not in result
        assert result["feature_names"] == ["feature_a", "feature_b", "feature_c"]

    def test_linear_method(self):
        model, X, _ = _make_simple_linear_model()
        result = compute_shap_values(model, X, method="linear")

        assert "values" in result
        assert "feature_names" in result
        assert result["feature_names"] == ["feature_a", "feature_b"]
        assert np.isfinite(result["base_value"])

    def test_feature_names_from_dataframe(self):
        model, X, _ = _make_simple_tree_model()
        result = compute_shap_values(model, X, method="tree")
        assert result["feature_names"] == list(X.columns)

    def test_feature_names_from_array(self):
        model, X, _ = _make_simple_tree_model()
        result = compute_shap_values(model, X.values, method="tree")
        assert result["feature_names"] == ["feature_0", "feature_1", "feature_2"]

    def test_unknown_method_returns_error(self):
        model, X, _ = _make_simple_tree_model()
        result = compute_shap_values(model, X, method="nonexistent")
        assert "error" in result

    def test_base_value_is_float(self):
        model, X, _ = _make_simple_tree_model()
        result = compute_shap_values(model, X, method="tree")
        assert isinstance(result["base_value"], float)
