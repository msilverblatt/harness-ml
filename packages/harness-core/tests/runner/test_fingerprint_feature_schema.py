"""Tests for feature-schema-aware fingerprinting.

Verifies that fingerprints change when features change (columns, dtypes,
row counts), ensuring the prediction cache invalidates properly.
"""
from __future__ import annotations

import pandas as pd
import pytest
from harnessml.core.runner.training.fingerprint import (
    compute_feature_schema,
    compute_fingerprint,
)

BASE_CONFIG = {"type": "xgboost", "params": {"max_depth": 6, "learning_rate": 0.1}}


class TestFingerprintFeatureSchemaAwareness:
    """Fingerprints must differ when feature schema differs."""

    def test_different_columns_different_fingerprint(self):
        schema_a = {"columns": ["a", "b"], "dtypes": {"a": "double", "b": "double"}, "row_count": 100}
        schema_b = {"columns": ["a", "b", "c"], "dtypes": {"a": "double", "b": "double", "c": "double"}, "row_count": 100}

        fp_a = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema_a)
        fp_b = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema_b)
        assert fp_a != fp_b

    def test_different_dtypes_different_fingerprint(self):
        schema_a = {"columns": ["a"], "dtypes": {"a": "double"}, "row_count": 100}
        schema_b = {"columns": ["a"], "dtypes": {"a": "int64"}, "row_count": 100}

        fp_a = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema_a)
        fp_b = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema_b)
        assert fp_a != fp_b

    def test_different_row_count_different_fingerprint(self):
        schema_a = {"columns": ["a"], "dtypes": {"a": "double"}, "row_count": 100}
        schema_b = {"columns": ["a"], "dtypes": {"a": "double"}, "row_count": 200}

        fp_a = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema_a)
        fp_b = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema_b)
        assert fp_a != fp_b

    def test_none_schema_vs_present_different_fingerprint(self):
        schema = {"columns": ["a"], "dtypes": {"a": "double"}, "row_count": 50}

        fp_none = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=None)
        fp_with = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema)
        assert fp_none != fp_with

    def test_same_schema_same_fingerprint(self):
        schema = {"columns": ["x", "y"], "dtypes": {"x": "double", "y": "int64"}, "row_count": 500}

        fp1 = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema)
        fp2 = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema)
        assert fp1 == fp2

    def test_schema_change_with_same_config_and_upstream(self):
        upstream = {"provider_model": "abc123"}
        schema_a = {"columns": ["a"], "dtypes": {"a": "double"}, "row_count": 100}
        schema_b = {"columns": ["a", "b"], "dtypes": {"a": "double", "b": "double"}, "row_count": 100}

        fp_a = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema_a, upstream_fingerprints=upstream)
        fp_b = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=schema_b, upstream_fingerprints=upstream)
        assert fp_a != fp_b


class TestComputeFeatureSchema:
    """Tests for compute_feature_schema reading parquet metadata."""

    def test_returns_correct_columns(self, tmp_path):
        df = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3, 4], "target": [0, 1]})
        path = tmp_path / "features.parquet"
        df.to_parquet(path, index=False)

        schema = compute_feature_schema(path)
        assert schema is not None
        assert schema["columns"] == ["feat_a", "feat_b", "target"]

    def test_returns_correct_row_count(self, tmp_path):
        df = pd.DataFrame({"a": range(42)})
        path = tmp_path / "features.parquet"
        df.to_parquet(path, index=False)

        schema = compute_feature_schema(path)
        assert schema["row_count"] == 42

    def test_returns_dtypes(self, tmp_path):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [1, 2]})
        path = tmp_path / "features.parquet"
        df.to_parquet(path, index=False)

        schema = compute_feature_schema(path)
        assert "x" in schema["dtypes"]
        assert "y" in schema["dtypes"]

    def test_nonexistent_file_returns_none(self, tmp_path):
        schema = compute_feature_schema(tmp_path / "nonexistent.parquet")
        assert schema is None

    def test_schema_changes_when_column_added(self, tmp_path):
        df1 = pd.DataFrame({"a": [1.0], "b": [2.0]})
        df2 = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})

        path1 = tmp_path / "v1.parquet"
        path2 = tmp_path / "v2.parquet"
        df1.to_parquet(path1, index=False)
        df2.to_parquet(path2, index=False)

        s1 = compute_feature_schema(path1)
        s2 = compute_feature_schema(path2)

        fp1 = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=s1)
        fp2 = compute_fingerprint(model_config=BASE_CONFIG, feature_schema=s2)
        assert fp1 != fp2
