"""Tests for ViewDef enhancements (depends_on, cache_ttl_seconds)."""
from harnessml.core.runner.schema import ViewDef


class TestViewDefDependsOn:
    def test_default_empty(self):
        v = ViewDef(source="raw")
        assert v.depends_on == []

    def test_explicit_deps(self):
        v = ViewDef(source="raw", depends_on=["view_a", "view_b"])
        assert v.depends_on == ["view_a", "view_b"]


class TestViewDefCacheTTL:
    def test_default_none(self):
        v = ViewDef(source="raw")
        assert v.cache_ttl_seconds is None

    def test_explicit_ttl(self):
        v = ViewDef(source="raw", cache_ttl_seconds=3600)
        assert v.cache_ttl_seconds == 3600

    def test_existing_fields_unchanged(self):
        v = ViewDef(source="raw", cache=False, description="test view")
        assert v.source == "raw"
        assert v.cache is False
        assert v.description == "test view"
        assert v.steps == []
