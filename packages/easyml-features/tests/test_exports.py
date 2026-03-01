"""Tests for easyml.features package exports."""


def test_all_exports_importable():
    from easyml.features import (
        FeatureBuilder,
        FeatureRegistry,
        FeatureResolver,
        PairwiseFeatureBuilder,
    )

    assert FeatureRegistry is not None
    assert FeatureResolver is not None
    assert FeatureBuilder is not None
    assert PairwiseFeatureBuilder is not None


def test_all_list_matches_exports():
    import easyml.features as mod

    expected = {"FeatureRegistry", "FeatureResolver", "FeatureBuilder", "PairwiseFeatureBuilder"}
    assert set(mod.__all__) == expected
