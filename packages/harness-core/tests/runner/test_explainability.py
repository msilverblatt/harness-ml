"""Tests for explainability module."""
import pytest
import numpy as np


class TestComputeShapSummary:
    def test_compute_shap_summary(self):
        shap = pytest.importorskip("shap")
        from sklearn.ensemble import GradientBoostingClassifier
        from harnessml.core.runner.explainability import compute_shap_summary

        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 5))
        y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
        feature_names = ["strong_signal", "weak_signal", "noise_1", "noise_2", "noise_3"]

        model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)

        result = compute_shap_summary(model, X, feature_names, top_n=3)
        assert len(result) == 3
        # strong_signal should be in top features
        top_names = [name for name, _ in result]
        assert "strong_signal" in top_names

    def test_compute_shap_top_n_limits(self):
        shap = pytest.importorskip("shap")
        from sklearn.ensemble import GradientBoostingClassifier
        from harnessml.core.runner.explainability import compute_shap_summary

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 10))
        y = (X[:, 0] > 0).astype(int)
        feature_names = [f"feat_{i}" for i in range(10)]

        model = GradientBoostingClassifier(n_estimators=20, max_depth=2, random_state=42)
        model.fit(X, y)

        result = compute_shap_summary(model, X, feature_names, top_n=5)
        assert len(result) == 5


class TestFormatShapReport:
    def test_format_shap_report(self):
        from harnessml.core.runner.explainability import format_shap_report

        results = [("feat_a", 0.1234), ("feat_b", 0.0567), ("feat_c", 0.0123)]
        output = format_shap_report(results, model_name="xgb_core")
        assert "xgb_core" in output
        assert "feat_a" in output
        assert "0.1234" in output
        assert "SHAP" in output

    def test_format_shap_report_no_model_name(self):
        from harnessml.core.runner.explainability import format_shap_report

        results = [("feat_a", 0.1)]
        output = format_shap_report(results)
        assert "SHAP" in output
        assert "feat_a" in output
