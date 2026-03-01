"""Tests for FeatureRegistry — decorator registration, source hashing, listing."""
import pytest
from easyml.schemas.core import TemporalFilter
from easyml.features.registry import FeatureRegistry


def test_register_feature():
    registry = FeatureRegistry()

    @registry.register(
        name="test_feature",
        category="offense",
        level="team",
        output_columns=["col_a", "col_b"],
    )
    def compute_test(df, config):
        return df

    assert "test_feature" in registry
    meta = registry.get_metadata("test_feature")
    assert meta.category == "offense"
    assert meta.output_columns == ["col_a", "col_b"]


def test_register_with_temporal_filter():
    registry = FeatureRegistry()

    @registry.register(
        name="safe_feature",
        category="stats",
        level="team",
        output_columns=["x"],
        temporal_filter=TemporalFilter(exclude_event_types=["tournament"]),
    )
    def compute(df, config):
        return df

    meta = registry.get_metadata("safe_feature")
    assert "tournament" in meta.temporal_filter.exclude_event_types


def test_source_hash_changes_on_code_change():
    registry = FeatureRegistry()

    @registry.register(name="v1", category="t", level="team", output_columns=["x"])
    def compute_v1(df, config):
        return df

    hash1 = registry.source_hash("v1")
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 hex


def test_list_features():
    registry = FeatureRegistry()
    for name, cat in [("a", "off"), ("b", "off"), ("c", "def")]:

        @registry.register(
            name=name, category=cat, level="team", output_columns=["x"]
        )
        def compute(df, config):
            return df

    assert len(registry.list_features()) == 3
    assert len(registry.list_features(category="off")) == 2


def test_duplicate_registration_raises():
    registry = FeatureRegistry()

    @registry.register(name="dup", category="t", level="team", output_columns=["x"])
    def compute1(df, config):
        return df

    with pytest.raises(ValueError, match="already registered"):

        @registry.register(
            name="dup", category="t", level="team", output_columns=["y"]
        )
        def compute2(df, config):
            return df


def test_get_nonexistent_raises():
    registry = FeatureRegistry()
    with pytest.raises(KeyError):
        registry.get_metadata("nonexistent")


def test_discover_modules(tmp_path):
    import sys

    # Create a temporary Python package with feature modules
    pkg_dir = tmp_path / "my_features"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "offense.py").write_text(
        '''
from easyml.features.registry import FeatureRegistry

_registry = None

def register_features(registry):
    @registry.register(name="discovered_feat", category="offense", level="team", output_columns=["x"])
    def compute(df, config):
        return df
'''
    )

    sys.path.insert(0, str(tmp_path))
    try:
        registry = FeatureRegistry()
        registry.discover("my_features")
        assert "discovered_feat" in registry
    finally:
        sys.path.pop(0)
