"""Tests for FeatureBuilder — compute, cache, and invalidation."""
import pandas as pd
from easyml.features.registry import FeatureRegistry
from easyml.features.builder import FeatureBuilder


def test_build_computes(tmp_path):
    registry = FeatureRegistry()
    call_count = {"n": 0}

    @registry.register(name="test", category="t", level="team", output_columns=["x"])
    def compute(df, config):
        call_count["n"] += 1
        df = df.copy()
        df["x"] = 1.0
        return df[["entity_id", "period_id", "x"]]

    sample_df = pd.DataFrame(
        {"entity_id": [1, 2], "period_id": [2025, 2025], "raw_col": [10, 20]}
    )
    builder = FeatureBuilder(
        registry=registry,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.json",
    )

    result = builder.build_all(raw_data=sample_df, config={})
    assert call_count["n"] == 1
    assert "x" in result.columns


def test_build_uses_cache(tmp_path):
    registry = FeatureRegistry()
    call_count = {"n": 0}

    @registry.register(name="test", category="t", level="team", output_columns=["x"])
    def compute(df, config):
        call_count["n"] += 1
        df = df.copy()
        df["x"] = 1.0
        return df[["entity_id", "period_id", "x"]]

    sample_df = pd.DataFrame(
        {"entity_id": [1, 2], "period_id": [2025, 2025], "raw_col": [10, 20]}
    )
    builder = FeatureBuilder(
        registry=registry,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.json",
    )

    builder.build_all(raw_data=sample_df, config={})
    assert call_count["n"] == 1

    builder.build_all(raw_data=sample_df, config={})
    assert call_count["n"] == 1  # NOT called again (source hash unchanged, cached)


def test_build_invalidates_on_code_change(tmp_path):
    registry1 = FeatureRegistry()
    call_count = {"n": 0}

    @registry1.register(name="test", category="t", level="team", output_columns=["x"])
    def compute1(df, config):
        call_count["n"] += 1
        df = df.copy()
        df["x"] = 1.0
        return df[["entity_id", "period_id", "x"]]

    sample_df = pd.DataFrame({"entity_id": [1, 2], "period_id": [2025, 2025]})
    builder1 = FeatureBuilder(
        registry=registry1,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.json",
    )
    builder1.build_all(raw_data=sample_df, config={})
    assert call_count["n"] == 1

    # New registry with different function — hash should differ
    registry2 = FeatureRegistry()

    @registry2.register(name="test", category="t", level="team", output_columns=["x"])
    def compute2(df, config):
        call_count["n"] += 1
        df = df.copy()
        df["x"] = 2.0  # different code
        return df[["entity_id", "period_id", "x"]]

    builder2 = FeatureBuilder(
        registry=registry2,
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.json",
    )
    builder2.build_all(raw_data=sample_df, config={})
    assert call_count["n"] == 2  # recomputed because source hash changed
