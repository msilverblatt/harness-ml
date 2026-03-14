"""Tests for statistical outlier detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.data.profiler import detect_outliers


class TestDetectOutliersIQR:
    """IQR-based outlier detection."""

    def test_detects_obvious_outlier(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
        df = pd.DataFrame({"x": data})
        results = detect_outliers(df, method="iqr")
        assert len(results) == 1
        assert results[0]["column"] == "x"
        assert results[0]["n_outliers"] >= 1

    def test_no_outliers_in_uniform_data(self):
        df = pd.DataFrame({"x": list(range(100))})
        results = detect_outliers(df, method="iqr")
        assert results[0]["n_outliers"] == 0

    def test_specific_column(self):
        df = pd.DataFrame({"a": [1, 2, 3, 100], "b": [1, 2, 3, 4]})
        results = detect_outliers(df, columns=["b"], method="iqr")
        assert len(results) == 1
        assert results[0]["column"] == "b"

    def test_skips_non_numeric(self):
        df = pd.DataFrame({"name": ["a", "b", "c"], "x": [1.0, 2.0, 100.0]})
        results = detect_outliers(df, method="iqr")
        columns_checked = {r["column"] for r in results}
        assert "name" not in columns_checked


class TestDetectOutliersZScore:
    """Z-score-based outlier detection."""

    def test_detects_outlier(self):
        rng = np.random.default_rng(42)
        normal_data = rng.normal(0, 1, 100).tolist()
        normal_data.append(10.0)  # extreme outlier
        df = pd.DataFrame({"x": normal_data})
        results = detect_outliers(df, method="zscore", threshold=3.0)
        assert results[0]["n_outliers"] >= 1

    def test_custom_threshold(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]})
        strict = detect_outliers(df, method="zscore", threshold=1.5)
        loose = detect_outliers(df, method="zscore", threshold=3.0)
        assert strict[0]["n_outliers"] >= loose[0]["n_outliers"]

    def test_invalid_method(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="Unknown method"):
            detect_outliers(df, method="invalid")
