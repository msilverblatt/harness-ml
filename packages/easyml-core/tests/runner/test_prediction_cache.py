"""Tests for prediction cache."""
from __future__ import annotations

import pandas as pd
import pytest

from easyml.core.runner.prediction_cache import PredictionCache


@pytest.fixture()
def cache(tmp_path):
    return PredictionCache(tmp_path / "pred_cache")


@pytest.fixture()
def sample_preds():
    return pd.DataFrame({"prediction": [0.6, 0.4, 0.7], "team_a": [1, 2, 3]})


# -----------------------------------------------------------------------
# store / lookup round-trip
# -----------------------------------------------------------------------

class TestStoreAndLookup:
    def test_round_trip(self, cache, sample_preds):
        cache.store("xgb_core", 2024, "abc123", sample_preds)
        result = cache.lookup("xgb_core", 2024, "abc123")
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_preds)

    def test_different_fingerprint_returns_none(self, cache, sample_preds):
        cache.store("xgb_core", 2024, "abc123", sample_preds)
        assert cache.lookup("xgb_core", 2024, "different") is None

    def test_different_fold_value_returns_none(self, cache, sample_preds):
        cache.store("xgb_core", 2024, "abc123", sample_preds)
        assert cache.lookup("xgb_core", 2023, "abc123") is None

    def test_different_model_returns_none(self, cache, sample_preds):
        cache.store("xgb_core", 2024, "abc123", sample_preds)
        assert cache.lookup("logreg", 2024, "abc123") is None


# -----------------------------------------------------------------------
# miss and corruption
# -----------------------------------------------------------------------

class TestMissAndCorruption:
    def test_lookup_on_empty_cache(self, cache):
        assert cache.lookup("nonexistent", 2024, "fp") is None

    def test_corrupt_file_returns_none(self, cache):
        # Write garbage to the expected path
        path = cache._path("xgb", 2024, "bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"this is not parquet")
        assert cache.lookup("xgb", 2024, "bad") is None


# -----------------------------------------------------------------------
# overwrite
# -----------------------------------------------------------------------

class TestOverwrite:
    def test_store_overwrites_same_fingerprint(self, cache):
        df1 = pd.DataFrame({"prediction": [0.5]})
        df2 = pd.DataFrame({"prediction": [0.9]})
        cache.store("m", 2024, "fp1", df1)
        cache.store("m", 2024, "fp1", df2)
        result = cache.lookup("m", 2024, "fp1")
        assert result is not None
        assert result["prediction"].iloc[0] == pytest.approx(0.9)


# -----------------------------------------------------------------------
# directory structure
# -----------------------------------------------------------------------

class TestDirectoryLayout:
    def test_path_structure(self, cache, sample_preds):
        path = cache.store("xgb_core", 2024, "abc123", sample_preds)
        # cache_dir / model_name / fold_value / fingerprint.parquet
        assert path.parent.name == "2024"
        assert path.parent.parent.name == "xgb_core"
        assert path.name == "abc123.parquet"


# -----------------------------------------------------------------------
# prune
# -----------------------------------------------------------------------

class TestPrune:
    def test_prune_keeps_newest(self, cache):
        import time

        for i in range(5):
            df = pd.DataFrame({"prediction": [float(i)]})
            cache.store("m", 2024, f"fp{i}", df)
            time.sleep(0.01)  # ensure distinct mtimes

        removed = cache.prune(keep_last_n=2)
        assert removed == 3
        # Oldest 3 removed, newest 2 remain
        assert cache.lookup("m", 2024, "fp3") is not None
        assert cache.lookup("m", 2024, "fp4") is not None
        assert cache.lookup("m", 2024, "fp0") is None

    def test_prune_on_empty_cache(self, cache):
        assert cache.prune() == 0

    def test_prune_per_model_per_fold_value(self, cache):
        import time

        for i in range(3):
            cache.store("m1", 2024, f"fp{i}", pd.DataFrame({"x": [i]}))
            time.sleep(0.01)
        for i in range(3):
            cache.store("m2", 2024, f"fp{i}", pd.DataFrame({"x": [i]}))
            time.sleep(0.01)

        removed = cache.prune(keep_last_n=1)
        # 2 removed from m1, 2 from m2
        assert removed == 4


# -----------------------------------------------------------------------
# clear
# -----------------------------------------------------------------------

class TestClear:
    def test_clear_removes_everything(self, cache, sample_preds):
        cache.store("a", 2024, "fp1", sample_preds)
        cache.store("b", 2024, "fp2", sample_preds)
        removed = cache.clear()
        assert removed == 2
        assert cache.lookup("a", 2024, "fp1") is None
