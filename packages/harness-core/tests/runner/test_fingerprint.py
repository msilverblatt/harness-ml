"""Tests for fingerprint caching."""
from __future__ import annotations

from harnessml.core.runner.training.fingerprint import (
    compute_feature_schema,
    compute_fingerprint,
    compute_meta_fingerprint,
    is_cached,
    load_meta_cache,
    save_fingerprint,
    save_meta_cache,
)

# -----------------------------------------------------------------------
# Tests: compute_fingerprint
# -----------------------------------------------------------------------

class TestComputeFingerprint:
    """Test compute_fingerprint determinism and sensitivity."""

    def test_deterministic(self):
        config = {"type": "xgboost", "params": {"n_estimators": 100}}
        fp1 = compute_fingerprint(config, data_mtime=1234567890.0)
        fp2 = compute_fingerprint(config, data_mtime=1234567890.0)
        assert fp1 == fp2

    def test_returns_16_hex(self):
        config = {"type": "xgboost"}
        fp = compute_fingerprint(config)
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_changes_when_config_changes(self):
        config_a = {"type": "xgboost", "params": {"n_estimators": 100}}
        config_b = {"type": "xgboost", "params": {"n_estimators": 200}}
        fp_a = compute_fingerprint(config_a, data_mtime=1234567890.0)
        fp_b = compute_fingerprint(config_b, data_mtime=1234567890.0)
        assert fp_a != fp_b

    def test_changes_when_mtime_changes(self):
        config = {"type": "xgboost"}
        fp_a = compute_fingerprint(config, data_mtime=1000.0)
        fp_b = compute_fingerprint(config, data_mtime=2000.0)
        assert fp_a != fp_b

    def test_none_mtime(self):
        config = {"type": "xgboost"}
        fp = compute_fingerprint(config, data_mtime=None)
        assert len(fp) == 16


# -----------------------------------------------------------------------
# Tests: is_cached / save_fingerprint round-trip
# -----------------------------------------------------------------------

class TestFingerprintRoundTrip:
    """Test is_cached and save_fingerprint together."""

    def test_not_cached_initially(self, tmp_path):
        assert not is_cached(tmp_path, "my_model", "abc123")

    def test_save_then_cached(self, tmp_path):
        save_fingerprint(tmp_path, "my_model", "abc123")
        assert is_cached(tmp_path, "my_model", "abc123")

    def test_mismatch_not_cached(self, tmp_path):
        save_fingerprint(tmp_path, "my_model", "abc123")
        assert not is_cached(tmp_path, "my_model", "different")

    def test_file_contents(self, tmp_path):
        save_fingerprint(tmp_path, "my_model", "abc123")
        fp_path = tmp_path / "my_model.fingerprint"
        assert fp_path.exists()
        assert fp_path.read_text().strip() == "abc123"

    def test_creates_directory(self, tmp_path):
        models_dir = tmp_path / "nested" / "dir"
        save_fingerprint(models_dir, "my_model", "abc123")
        assert is_cached(models_dir, "my_model", "abc123")


# -----------------------------------------------------------------------
# Tests: compute_meta_fingerprint
# -----------------------------------------------------------------------

class TestComputeMetaFingerprint:
    """Test compute_meta_fingerprint."""

    def test_deterministic(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        (pred_dir / "2022.parquet").write_text("fake data")
        (pred_dir / "2023.parquet").write_text("more fake data")

        config = {"method": "stacked", "meta_learner": {"C": 1.0}}
        fp1 = compute_meta_fingerprint(pred_dir, config)
        fp2 = compute_meta_fingerprint(pred_dir, config)
        assert fp1 == fp2

    def test_changes_with_config(self, tmp_path):
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        (pred_dir / "2022.parquet").write_text("fake data")

        config_a = {"method": "stacked", "meta_learner": {"C": 1.0}}
        config_b = {"method": "stacked", "meta_learner": {"C": 2.0}}
        fp_a = compute_meta_fingerprint(pred_dir, config_a)
        fp_b = compute_meta_fingerprint(pred_dir, config_b)
        assert fp_a != fp_b

    def test_returns_16_hex(self, tmp_path):
        config = {"method": "stacked"}
        fp = compute_meta_fingerprint(tmp_path, config)
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_empty_dir(self, tmp_path):
        config = {"method": "stacked"}
        fp = compute_meta_fingerprint(tmp_path / "nonexistent", config)
        assert len(fp) == 16


# -----------------------------------------------------------------------
# Tests: save_meta_cache / load_meta_cache round-trip
# -----------------------------------------------------------------------

class TestMetaCacheRoundTrip:
    """Test save_meta_cache and load_meta_cache."""

    def test_round_trip(self, tmp_path):
        cache_dir = tmp_path / "cache"
        meta = {"model_names": ["a", "b"], "coefs": [1.0, 2.0]}
        calibrator = {"type": "spline"}
        pre_cals = {"a": {"type": "spline"}}
        fingerprint = "abc123def456"

        save_meta_cache(cache_dir, meta, calibrator, pre_cals, fingerprint)
        result = load_meta_cache(cache_dir, fingerprint)

        assert result is not None
        loaded_meta, loaded_cal, loaded_pre_cals = result
        assert loaded_meta == meta
        assert loaded_cal == calibrator
        assert loaded_pre_cals == pre_cals

    def test_miss_on_wrong_fingerprint(self, tmp_path):
        cache_dir = tmp_path / "cache"
        save_meta_cache(cache_dir, "meta", "cal", {}, "abc123")
        result = load_meta_cache(cache_dir, "different")
        assert result is None

    def test_miss_on_no_cache(self, tmp_path):
        result = load_meta_cache(tmp_path / "nonexistent", "abc123")
        assert result is None


# -----------------------------------------------------------------------
# Tests: compute_feature_schema
# -----------------------------------------------------------------------

class TestComputeFeatureSchema:
    """Test compute_feature_schema reads parquet metadata."""

    def test_compute_feature_schema_reads_parquet(self, tmp_path):
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
        pq_path = tmp_path / "features.parquet"
        df.to_parquet(pq_path, index=False)

        schema = compute_feature_schema(pq_path)
        assert schema is not None
        assert schema["columns"] == ["a", "b", "c"]
        assert "a" in schema["dtypes"]
        assert "b" in schema["dtypes"]
        assert "c" in schema["dtypes"]
        assert schema["row_count"] == 3

    def test_compute_feature_schema_returns_none_missing_file(self, tmp_path):
        result = compute_feature_schema(tmp_path / "nonexistent.parquet")
        assert result is None

    def test_compute_feature_schema_includes_null_counts(self, tmp_path):
        import numpy as np
        import pandas as pd

        df = pd.DataFrame({
            "x": [1.0, np.nan, 3.0, np.nan],
            "y": [10, 20, 30, 40],
        })
        pq_path = tmp_path / "with_nulls.parquet"
        df.to_parquet(pq_path, index=False)

        schema = compute_feature_schema(pq_path)
        assert schema is not None
        assert "null_counts" in schema
        assert schema["null_counts"]["x"] == 2
        assert schema["null_counts"]["y"] == 0


# -----------------------------------------------------------------------
# Tests: compute_fingerprint with feature_schema
# -----------------------------------------------------------------------

class TestFingerprintWithSchema:
    """Test compute_fingerprint feature_schema integration."""

    def test_fingerprint_with_schema_changes_hash(self):
        config = {"type": "xgboost", "params": {"n_estimators": 100}}
        schema_a = {
            "columns": ["a", "b"],
            "dtypes": {"a": "int64", "b": "double"},
            "null_counts": {"a": 0, "b": 0},
            "row_count": 100,
        }
        schema_b = {
            "columns": ["a", "b", "c"],
            "dtypes": {"a": "int64", "b": "double", "c": "string"},
            "null_counts": {"a": 0, "b": 0, "c": 5},
            "row_count": 100,
        }
        fp_a = compute_fingerprint(config, data_mtime=1000.0, feature_schema=schema_a)
        fp_b = compute_fingerprint(config, data_mtime=1000.0, feature_schema=schema_b)
        assert fp_a != fp_b

    def test_fingerprint_none_schema_backward_compatible(self):
        config = {"type": "xgboost", "params": {"n_estimators": 100}}
        fp_without = compute_fingerprint(config, data_mtime=1234567890.0)
        fp_with_none = compute_fingerprint(config, data_mtime=1234567890.0, feature_schema=None)
        assert fp_without == fp_with_none
