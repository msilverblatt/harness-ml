"""Tests for feature and source auto-loaders."""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from easyml.core.runner.sources import SourceRegistry
from easyml.core.feature_eng.registry import FeatureRegistry
from easyml.core.runner.loaders import load_features, load_sources
from easyml.core.runner.schema import FeatureDecl, SourceDecl


# -----------------------------------------------------------------------
# Helpers — create mock modules injected into sys.modules
# -----------------------------------------------------------------------

def _make_mock_module(name: str, functions: dict) -> types.ModuleType:
    """Create a mock module with given functions and inject into sys.modules."""
    mod = types.ModuleType(name)
    for fn_name, fn in functions.items():
        setattr(mod, fn_name, fn)
    sys.modules[name] = mod
    return mod


def _cleanup_module(name: str) -> None:
    """Remove a mock module from sys.modules."""
    sys.modules.pop(name, None)


# -----------------------------------------------------------------------
# Feature loader tests
# -----------------------------------------------------------------------

class TestLoadFeatures:
    """load_features() — dynamic import and registration."""

    def test_loads_feature_from_mock_module(self):
        """Single feature loaded and registered."""
        def compute_eff(df):
            return df

        _make_mock_module("test_feat_mod", {"compute_eff": compute_eff})
        try:
            registry = FeatureRegistry()
            decls = {
                "efficiency": FeatureDecl(
                    module="test_feat_mod",
                    function="compute_eff",
                    category="efficiency",
                    level="entity",
                    columns=["adj_oe", "adj_de"],
                    nan_strategy="median",
                )
            }
            load_features(decls, registry)
            assert "efficiency" in registry
            meta = registry.get_metadata("efficiency")
            assert meta.category == "efficiency"
            assert meta.output_columns == ["adj_oe", "adj_de"]
            assert meta.nan_strategy == "median"
        finally:
            _cleanup_module("test_feat_mod")

    def test_loads_multiple_features(self):
        """Multiple features loaded from different modules."""
        def feat_a(df):
            return df

        def feat_b(df):
            return df

        _make_mock_module("mod_a", {"feat_a": feat_a})
        _make_mock_module("mod_b", {"feat_b": feat_b})
        try:
            registry = FeatureRegistry()
            decls = {
                "feature_a": FeatureDecl(
                    module="mod_a",
                    function="feat_a",
                    category="cat_a",
                    level="entity",
                    columns=["col_a"],
                ),
                "feature_b": FeatureDecl(
                    module="mod_b",
                    function="feat_b",
                    category="cat_b",
                    level="pairwise",
                    columns=["col_b1", "col_b2"],
                ),
            }
            load_features(decls, registry)
            assert len(registry) == 2
            assert "feature_a" in registry
            assert "feature_b" in registry
            assert registry.get_metadata("feature_b").level == "pairwise"
        finally:
            _cleanup_module("mod_a")
            _cleanup_module("mod_b")

    def test_bad_module_raises_import_error(self):
        """Missing module raises ImportError with clear message."""
        registry = FeatureRegistry()
        decls = {
            "broken": FeatureDecl(
                module="nonexistent_module_xyz",
                function="fn",
                category="cat",
                level="entity",
                columns=["x"],
            )
        }
        with pytest.raises(ImportError, match="nonexistent_module_xyz"):
            load_features(decls, registry)

    def test_bad_function_raises_attribute_error(self):
        """Module exists but function doesn't — AttributeError with clear message."""
        _make_mock_module("mod_no_fn", {})
        try:
            registry = FeatureRegistry()
            decls = {
                "broken": FeatureDecl(
                    module="mod_no_fn",
                    function="missing_function",
                    category="cat",
                    level="entity",
                    columns=["x"],
                )
            }
            with pytest.raises(AttributeError, match="missing_function"):
                load_features(decls, registry)
        finally:
            _cleanup_module("mod_no_fn")


# -----------------------------------------------------------------------
# Source loader tests
# -----------------------------------------------------------------------

class TestLoadSources:
    """load_sources() — dynamic import and registration."""

    def test_loads_source_from_mock_module(self):
        """Single source loaded and registered."""
        def scrape_data(output_dir, config):
            pass

        _make_mock_module("test_src_mod", {"scrape_data": scrape_data})
        try:
            registry = SourceRegistry()
            decls = {
                "kenpom": SourceDecl(
                    module="test_src_mod",
                    function="scrape_data",
                    category="external",
                    temporal_safety="pre_event",
                    outputs=["data/raw/kenpom/"],
                    leakage_notes="Pre-tournament only",
                )
            }
            load_sources(decls, registry)
            assert "kenpom" in registry
            meta = registry.get_metadata("kenpom")
            assert meta.category == "external"
            assert meta.temporal_safety == "pre_event"
        finally:
            _cleanup_module("test_src_mod")

    def test_bad_source_module_raises(self):
        """Missing source module raises ImportError with clear message."""
        registry = SourceRegistry()
        decls = {
            "broken": SourceDecl(
                module="nonexistent_source_mod",
                function="fn",
                category="external",
                temporal_safety="unknown",
                outputs=[],
            )
        }
        with pytest.raises(ImportError, match="nonexistent_source_mod"):
            load_sources(decls, registry)

    def test_bad_source_function_raises(self):
        """Source module exists but function missing — AttributeError."""
        _make_mock_module("src_mod_no_fn", {})
        try:
            registry = SourceRegistry()
            decls = {
                "broken": SourceDecl(
                    module="src_mod_no_fn",
                    function="missing_fn",
                    category="internal",
                    temporal_safety="unknown",
                    outputs=[],
                )
            }
            with pytest.raises(AttributeError, match="missing_fn"):
                load_sources(decls, registry)
        finally:
            _cleanup_module("src_mod_no_fn")
