import numpy as np
import pandas as pd
import pytest
from harness.ml.features.schema import FeatureDefinition, FeatureType, FeatureSet
from harness.ml.features.resolver import FeatureResolver


class TestFeatureResolver:
    def _base_df(self):
        return pd.DataFrame({
            "entity_a_elo": [90.0, 80.0, 70.0],
            "entity_b_elo": [80.0, 85.0, 60.0],
            "surface": ["clay", "hard", "grass"],
            "court_surface": ["clay", "hard", "grass"],
        })

    def test_resolve_instance(self):
        df = self._base_df()
        fs = FeatureSet(features={
            "surface": FeatureDefinition(
                name="surface",
                feature_type=FeatureType.INSTANCE,
                source_column="court_surface",
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(df, fs)
        assert "surface" in result.columns
        assert list(result["surface"]) == ["clay", "hard", "grass"]

    def test_resolve_entity_generates_derivatives(self):
        df = self._base_df()
        fs = FeatureSet(features={
            "elo": FeatureDefinition(
                name="elo",
                feature_type=FeatureType.ENTITY,
                pairwise_methods=["diff", "ratio"],
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(df, fs)
        assert "diff_elo" in result.columns
        assert "ratio_elo" in result.columns
        np.testing.assert_array_almost_equal(
            result["diff_elo"].values, [10.0, -5.0, 10.0]
        )
        np.testing.assert_array_almost_equal(
            result["ratio_elo"].values, [90.0 / 80.0, 80.0 / 85.0, 70.0 / 60.0]
        )

    def test_resolve_pairwise_formula(self):
        df = self._base_df()
        fs = FeatureSet(features={
            "elo_diff": FeatureDefinition(
                name="elo_diff",
                feature_type=FeatureType.PAIRWISE,
                formula="entity_a_elo - entity_b_elo",
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(df, fs)
        assert "elo_diff" in result.columns
        np.testing.assert_almost_equal(result["elo_diff"].iloc[0], 10.0)

    def test_only_active_features_resolved(self):
        df = self._base_df()
        fs = FeatureSet(features={
            "elo": FeatureDefinition(
                name="elo",
                feature_type=FeatureType.ENTITY,
                active=False,
            ),
            "surface": FeatureDefinition(
                name="surface",
                feature_type=FeatureType.INSTANCE,
                source_column="court_surface",
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(df, fs)
        assert "diff_elo" not in result.columns
        assert "surface" in result.columns

    def test_multiple_types_together(self):
        df = self._base_df()
        fs = FeatureSet(features={
            "elo": FeatureDefinition(
                name="elo",
                feature_type=FeatureType.ENTITY,
                pairwise_methods=["diff"],
            ),
            "surface": FeatureDefinition(
                name="surface",
                feature_type=FeatureType.INSTANCE,
                source_column="court_surface",
            ),
            "elo_formula": FeatureDefinition(
                name="elo_formula",
                feature_type=FeatureType.PAIRWISE,
                formula="entity_a_elo - entity_b_elo",
            ),
        })
        resolver = FeatureResolver()
        result = resolver.resolve(df, fs)
        assert "diff_elo" in result.columns
        assert "surface" in result.columns
        assert "elo_formula" in result.columns

    def test_missing_column_raises(self):
        df = pd.DataFrame({"x": [1]})
        fs = FeatureSet(features={
            "missing": FeatureDefinition(
                name="missing",
                feature_type=FeatureType.INSTANCE,
                source_column="nonexistent",
            ),
        })
        resolver = FeatureResolver()
        with pytest.raises(ValueError, match="column.*not found"):
            resolver.resolve(df, fs)

    def test_resolved_feature_names_tracking(self):
        df = self._base_df()
        fs = FeatureSet(features={
            "elo": FeatureDefinition(
                name="elo",
                feature_type=FeatureType.ENTITY,
                pairwise_methods=["diff", "ratio"],
            ),
            "surface": FeatureDefinition(
                name="surface",
                feature_type=FeatureType.INSTANCE,
                source_column="court_surface",
            ),
        })
        resolver = FeatureResolver()
        resolver.resolve(df, fs)
        names = resolver.resolved_feature_names
        assert "diff_elo" in names
        assert "ratio_elo" in names
        assert "surface" in names
        assert len(names) == 3
