"""Tests for optional matplotlib visualization module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip all tests if matplotlib is not installed
plt_mod = pytest.importorskip("matplotlib")

from harnessml.core.runner.analysis.viz import (
    render_calibration,
    render_confusion_matrix,
    render_feature_importance,
    render_pr_curve,
    render_roc_curve,
    render_shap_summary,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def roc_data():
    return {
        "fpr": [0.0, 0.1, 0.3, 0.5, 1.0],
        "tpr": [0.0, 0.4, 0.7, 0.9, 1.0],
        "thresholds": [1.0, 0.8, 0.5, 0.3, 0.0],
        "auc": 0.85,
    }


@pytest.fixture
def pr_data():
    return {
        "precision": [1.0, 0.9, 0.8, 0.7, 0.5],
        "recall": [0.0, 0.3, 0.5, 0.7, 1.0],
        "thresholds": [0.9, 0.7, 0.5, 0.3],
        "average_precision": 0.82,
    }


@pytest.fixture
def reliability_data():
    return [
        {"bin_center": 0.15, "mean_predicted": 0.15, "fraction_positive": 0.12, "count": 20},
        {"bin_center": 0.35, "mean_predicted": 0.35, "fraction_positive": 0.33, "count": 30},
        {"bin_center": 0.55, "mean_predicted": 0.55, "fraction_positive": 0.58, "count": 25},
        {"bin_center": 0.75, "mean_predicted": 0.75, "fraction_positive": 0.72, "count": 15},
        {"bin_center": 0.95, "mean_predicted": 0.92, "fraction_positive": 0.90, "count": 10},
    ]


@pytest.fixture
def importance_data():
    return {
        "feature_names": ["feat_a", "feat_b", "feat_c", "feat_d"],
        "importances_mean": [0.3, 0.1, 0.5, 0.05],
        "importances_std": [0.02, 0.01, 0.03, 0.005],
    }


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestRenderROCCurve:
    def test_creates_file(self, tmp_dir, roc_data):
        path = tmp_dir / "roc.png"
        result = render_roc_curve(roc_data, path)
        assert result == str(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_returns_path_string(self, tmp_dir, roc_data):
        path = tmp_dir / "roc.png"
        result = render_roc_curve(roc_data, path)
        assert isinstance(result, str)


class TestRenderPRCurve:
    def test_creates_file(self, tmp_dir, pr_data):
        path = tmp_dir / "pr.png"
        result = render_pr_curve(pr_data, path)
        assert result == str(path)
        assert path.exists()
        assert path.stat().st_size > 0


class TestRenderCalibration:
    def test_creates_file(self, tmp_dir, reliability_data):
        path = tmp_dir / "cal.png"
        result = render_calibration(reliability_data, path)
        assert result == str(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_handles_single_bin(self, tmp_dir):
        data = [{"bin_center": 0.5, "mean_predicted": 0.5, "fraction_positive": 0.5, "count": 100}]
        path = tmp_dir / "cal_single.png"
        result = render_calibration(data, path)
        assert result == str(path)
        assert path.exists()


class TestRenderConfusionMatrix:
    def test_creates_file(self, tmp_dir):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        path = tmp_dir / "cm.png"
        result = render_confusion_matrix(y_true, y_pred, path)
        assert result == str(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_custom_labels(self, tmp_dir):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        path = tmp_dir / "cm_labels.png"
        result = render_confusion_matrix(y_true, y_pred, path, labels=["Neg", "Pos"])
        assert result == str(path)
        assert path.exists()


class TestRenderFeatureImportance:
    def test_creates_file(self, tmp_dir, importance_data):
        path = tmp_dir / "imp.png"
        result = render_feature_importance(importance_data, path)
        assert result == str(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_top_n(self, tmp_dir, importance_data):
        path = tmp_dir / "imp_top2.png"
        result = render_feature_importance(importance_data, path, top_n=2)
        assert result == str(path)
        assert path.exists()


class TestRenderShapSummary:
    def test_creates_file_with_fallback(self, tmp_dir):
        """Test fallback bar chart when shap is not available."""
        import pandas as pd

        shap_data = {
            "values": np.random.randn(50, 3),
            "feature_names": ["f1", "f2", "f3"],
            "base_value": 0.5,
        }
        X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
        path = tmp_dir / "shap.png"
        result = render_shap_summary(shap_data, X, path)
        assert result == str(path)
        assert path.exists()

    def test_handles_error_in_shap_data(self, tmp_dir):
        import pandas as pd
        shap_data = {"error": "shap not installed"}
        X = pd.DataFrame({"a": [1, 2, 3]})
        path = tmp_dir / "shap_err.png"
        result = render_shap_summary(shap_data, X, path)
        assert result == "shap not installed"
        assert not path.exists()

    def test_handles_3d_shap_values(self, tmp_dir):
        """Multi-output SHAP values (3D array) should be handled."""
        import pandas as pd

        shap_data = {
            "values": np.random.randn(50, 3, 2),  # 2 outputs
            "feature_names": ["f1", "f2", "f3"],
            "base_value": 0.5,
        }
        X = pd.DataFrame(np.random.randn(50, 3), columns=["f1", "f2", "f3"])
        path = tmp_dir / "shap_3d.png"
        result = render_shap_summary(shap_data, X, path)
        assert result == str(path)
        assert path.exists()
