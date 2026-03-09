"""Tests for feature cache with manifest and cascade invalidation."""
from __future__ import annotations

import pandas as pd
from harnessml.core.runner.features.cache import CacheEntry, FeatureCache


class TestCacheEntry:
    def test_create_entry(self):
        entry = CacheEntry(
            cache_key="abc123", path="entity/adj_em.parquet",
            feature_type="entity", source="kenpom",
        )
        assert entry.cache_key == "abc123"
        assert entry.derived_from == []
        assert entry.derivatives == []

    def test_entry_with_derivatives(self):
        entry = CacheEntry(
            cache_key="abc123", path="entity/adj_em.parquet",
            feature_type="entity", derivatives=["diff_adj_em", "ratio_adj_em"],
        )
        assert entry.derivatives == ["diff_adj_em", "ratio_adj_em"]


class TestFeatureCache:
    def test_init_creates_dirs(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        assert (tmp_path / "cache").exists()
        assert (tmp_path / "cache" / "entity").exists()
        assert (tmp_path / "cache" / "pairwise").exists()
        assert (tmp_path / "cache" / "instance").exists()
        assert (tmp_path / "cache" / "regime").exists()

    def test_put_and_get(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        series = pd.Series([1.0, 2.0, 3.0], name="adj_em")
        cache.put("adj_em", "key123", series, feature_type="entity")
        result = cache.get("adj_em", "key123")
        assert result is not None
        pd.testing.assert_series_equal(result, series, check_names=False)

    def test_get_miss_wrong_key(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        series = pd.Series([1.0, 2.0], name="adj_em")
        cache.put("adj_em", "key123", series, feature_type="entity")
        result = cache.get("adj_em", "different_key")
        assert result is None

    def test_get_miss_not_cached(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        assert cache.get("nonexistent", "key") is None

    def test_invalidate_feature(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        series = pd.Series([1.0, 2.0], name="adj_em")
        cache.put("adj_em", "key123", series, feature_type="entity")
        cache.invalidate("adj_em")
        assert cache.get("adj_em", "key123") is None

    def test_invalidate_cascades(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        team_s = pd.Series([1.0, 2.0], name="adj_em")
        pair_s = pd.Series([0.5, -0.3], name="diff_adj_em")
        cache.put("adj_em", "k1", team_s, feature_type="entity",
                  derivatives=["diff_adj_em"])
        cache.put("diff_adj_em", "k2", pair_s, feature_type="pairwise",
                  derived_from=["adj_em"])
        cache.invalidate("adj_em")
        assert cache.get("adj_em", "k1") is None
        assert cache.get("diff_adj_em", "k2") is None

    def test_manifest_persists(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache1 = FeatureCache(cache_dir)
        cache1.put("adj_em", "k1", pd.Series([1.0]), feature_type="entity")
        cache2 = FeatureCache(cache_dir)
        assert cache2.get("adj_em", "k1") is not None

    def test_compute_cache_key(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        key1 = cache.compute_key(name="adj_em", feature_type="entity",
                                 source="kenpom", column="AdjEM")
        key2 = cache.compute_key(name="adj_em", feature_type="entity",
                                 source="kenpom", column="AdjEM")
        key3 = cache.compute_key(name="adj_em", feature_type="entity",
                                 source="different", column="AdjEM")
        assert key1 == key2
        assert key1 != key3

    def test_list_cached(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        cache.put("adj_em", "k1", pd.Series([1.0]), feature_type="entity")
        cache.put("diff_adj_em", "k2", pd.Series([0.5]), feature_type="pairwise")
        cached = cache.list_cached()
        assert "adj_em" in cached
        assert "diff_adj_em" in cached

    def test_list_cached_by_type(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        cache.put("adj_em", "k1", pd.Series([1.0]), feature_type="entity")
        cache.put("diff_adj_em", "k2", pd.Series([0.5]), feature_type="pairwise")
        team_only = cache.list_cached(feature_type="entity")
        assert "adj_em" in team_only
        assert "diff_adj_em" not in team_only
