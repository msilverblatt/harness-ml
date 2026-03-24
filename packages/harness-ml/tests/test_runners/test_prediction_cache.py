import numpy as np
import pytest

from harness.ml.runners.prediction_cache import PredictionCache


class TestFingerprint:
    def test_deterministic(self):
        cache = PredictionCache()
        config = {"model_type": "xgb", "n_estimators": 100}
        fp1 = cache.compute_fingerprint(config, "schema_v1")
        fp2 = cache.compute_fingerprint(config, "schema_v1")
        assert fp1 == fp2

    def test_different_config_different_fingerprint(self):
        cache = PredictionCache()
        fp1 = cache.compute_fingerprint({"n_estimators": 100}, "schema_v1")
        fp2 = cache.compute_fingerprint({"n_estimators": 200}, "schema_v1")
        assert fp1 != fp2

    def test_upstream_fingerprint_change_cascades(self):
        cache = PredictionCache()
        config = {"model_type": "lgbm"}
        fp1 = cache.compute_fingerprint(config, "schema_v1", upstream_fingerprints={"provider": "abc123"})
        fp2 = cache.compute_fingerprint(config, "schema_v1", upstream_fingerprints={"provider": "def456"})
        assert fp1 != fp2

    def test_no_upstream_vs_empty_upstream_same(self):
        cache = PredictionCache()
        config = {"model_type": "lgbm"}
        fp1 = cache.compute_fingerprint(config, "schema_v1", upstream_fingerprints=None)
        fp2 = cache.compute_fingerprint(config, "schema_v1", upstream_fingerprints={})
        assert fp1 == fp2


class TestGetPut:
    def test_get_put_roundtrip(self, tmp_path):
        cache = PredictionCache(cache_dir=tmp_path)
        preds = np.array([0.1, 0.2, 0.3])
        cache.put("model_a", "fold_0", "fp123", preds)
        result = cache.get("model_a", "fold_0", "fp123")
        assert result is not None
        np.testing.assert_array_equal(result, preds)

    def test_get_cache_miss_returns_none(self, tmp_path):
        cache = PredictionCache(cache_dir=tmp_path)
        assert cache.get("model_a", "fold_0", "nonexistent") is None

    def test_has_true_after_put(self, tmp_path):
        cache = PredictionCache(cache_dir=tmp_path)
        preds = np.array([1.0, 2.0])
        assert not cache.has("model_a", "fold_0", "fp_abc")
        cache.put("model_a", "fold_0", "fp_abc", preds)
        assert cache.has("model_a", "fold_0", "fp_abc")

    def test_has_false_before_put(self, tmp_path):
        cache = PredictionCache(cache_dir=tmp_path)
        assert not cache.has("model_x", "fold_1", "fp_xyz")


class TestNoCacheDir:
    def test_get_returns_none_when_no_dir(self):
        cache = PredictionCache(cache_dir=None)
        assert cache.get("model_a", "fold_0", "fp123") is None

    def test_put_is_noop_when_no_dir(self):
        cache = PredictionCache(cache_dir=None)
        # Should not raise
        cache.put("model_a", "fold_0", "fp123", np.array([1.0, 2.0]))

    def test_has_returns_false_when_no_dir(self):
        cache = PredictionCache(cache_dir=None)
        assert not cache.has("model_a", "fold_0", "fp123")
