"""Tests for FeatureResolver — column resolution from registry metadata."""
import pytest
from easyml.features.registry import FeatureRegistry
from easyml.features.resolver import FeatureResolver


def make_test_registry():
    registry = FeatureRegistry()

    @registry.register(
        name="scoring",
        category="offense",
        level="team",
        output_columns=["scoring_margin", "ppg"],
    )
    def compute_scoring(df, config):
        return df

    @registry.register(
        name="defense",
        category="defense",
        level="team",
        output_columns=["opp_ppg", "blocks"],
    )
    def compute_defense(df, config):
        return df

    @registry.register(
        name="seed",
        category="meta",
        level="team",
        output_columns=["seed_num"],
    )
    def compute_seed(df, config):
        return df

    return registry


def test_resolve_explicit_columns():
    registry = make_test_registry()
    resolver = FeatureResolver(registry=registry)
    columns = resolver.resolve(
        ["scoring_margin", "seed_num"],
        available_columns=["scoring_margin", "ppg", "seed_num", "opp_ppg"],
    )
    assert columns == ["scoring_margin", "seed_num"]


def test_resolve_missing_column_raises():
    registry = make_test_registry()
    resolver = FeatureResolver(registry=registry)
    with pytest.raises(ValueError, match="not found"):
        resolver.resolve(["nonexistent"], available_columns=["scoring_margin"])


def test_resolve_by_category():
    registry = make_test_registry()
    resolver = FeatureResolver(registry=registry)
    columns = resolver.resolve_category(
        "offense",
        available_columns=["scoring_margin", "ppg", "seed_num", "opp_ppg"],
    )
    assert set(columns) == {"scoring_margin", "ppg"}


def test_resolve_all():
    registry = make_test_registry()
    resolver = FeatureResolver(registry=registry)
    columns = resolver.resolve_all(
        available_columns=["scoring_margin", "ppg", "seed_num", "opp_ppg", "blocks"],
    )
    assert set(columns) == {"scoring_margin", "ppg", "seed_num", "opp_ppg", "blocks"}
