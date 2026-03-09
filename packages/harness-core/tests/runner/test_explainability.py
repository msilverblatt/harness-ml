"""Tests for explainability module."""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


class TestComputeShapSummary:
    def test_compute_shap_summary(self):
        shap = pytest.importorskip("shap")
        from harnessml.core.runner.explainability import compute_shap_summary
        from sklearn.ensemble import GradientBoostingClassifier

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
        from harnessml.core.runner.explainability import compute_shap_summary
        from sklearn.ensemble import GradientBoostingClassifier

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


@pytest.fixture
def fitted_rf_model():
    shap = pytest.importorskip("shap")  # noqa: F841
    np.random.seed(42)
    X = pd.DataFrame({f"f{i}": np.random.randn(200) for i in range(5)})
    y = (X["f0"] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


class TestComputeShapValues:
    def test_shap_values_shape(self, fitted_rf_model):
        from harnessml.core.runner.explainability import compute_shap_values

        model, X, y = fitted_rf_model
        shap_values = compute_shap_values(model, X)
        assert shap_values.shape == X.shape

    def test_shap_values_max_samples(self, fitted_rf_model):
        from harnessml.core.runner.explainability import compute_shap_values

        model, X, y = fitted_rf_model
        shap_values = compute_shap_values(model, X, max_samples=50)
        assert shap_values.shape == (50, X.shape[1])


class TestComputePdp:
    def test_pdp_values(self, fitted_rf_model):
        from harnessml.core.runner.explainability import compute_pdp

        model, X, y = fitted_rf_model
        pdp = compute_pdp(model, X, feature_idx=0)
        assert "values" in pdp
        assert "avg_predictions" in pdp
        assert len(pdp["values"]) > 0
        assert len(pdp["avg_predictions"]) > 0

    def test_pdp_by_column_name(self, fitted_rf_model):
        from harnessml.core.runner.explainability import compute_pdp

        model, X, y = fitted_rf_model
        pdp = compute_pdp(model, X, feature_idx="f0")
        assert "values" in pdp
        assert len(pdp["values"]) > 0


class TestComputeFeatureInteractions:
    def test_feature_interactions(self, fitted_rf_model):
        from harnessml.core.runner.explainability import compute_feature_interactions

        model, X, y = fitted_rf_model
        interactions = compute_feature_interactions(
            model, X, feature_names=list(X.columns)
        )
        assert len(interactions) > 0
        assert "feature_1" in interactions[0]
        assert "feature_2" in interactions[0]
        assert "interaction_strength" in interactions[0]

    def test_feature_interactions_auto_names(self, fitted_rf_model):
        from harnessml.core.runner.explainability import compute_feature_interactions

        model, X, y = fitted_rf_model
        interactions = compute_feature_interactions(model, X)
        assert len(interactions) > 0
        # Should use DataFrame column names automatically
        assert interactions[0]["feature_1"].startswith("f")

    def test_feature_interactions_top_k(self, fitted_rf_model):
        from harnessml.core.runner.explainability import compute_feature_interactions

        model, X, y = fitted_rf_model
        interactions = compute_feature_interactions(model, X, top_k=3)
        assert len(interactions) <= 3
