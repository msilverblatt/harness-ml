"""Tests for FeatureDecl/FeatureDef merge."""
from harnessml.core.runner.schema import FeatureDecl, FeatureDef, FeatureType


class TestFeatureDeclIsFeatureDef:
    """FeatureDecl is now an alias for FeatureDef."""

    def test_alias_identity(self):
        assert FeatureDecl is FeatureDef

    def test_legacy_decl_construction(self):
        """Old FeatureDecl construction still works."""
        fd = FeatureDecl(
            module="my_module",
            function="compute",
            category="stats",
            level="entity",
            columns=["a", "b"],
        )
        assert fd.module == "my_module"
        assert fd.function == "compute"
        assert fd.category == "stats"
        assert fd.level == "entity"
        assert fd.columns == ["a", "b"]
        assert fd.nan_strategy == "median"

    def test_new_feature_def_construction(self):
        """New FeatureDef construction still works."""
        fd = FeatureDef(
            name="adj_em",
            type=FeatureType.ENTITY,
            source="kenpom",
            description="Adjusted efficiency margin",
        )
        assert fd.name == "adj_em"
        assert fd.type == FeatureType.ENTITY
        assert fd.source == "kenpom"
        assert fd.module is None
        assert fd.columns is None

    def test_combined_fields(self):
        """Both old and new fields can coexist."""
        fd = FeatureDef(
            name="hybrid",
            type=FeatureType.INSTANCE,
            module="features.hybrid",
            function="compute_hybrid",
            columns=["x", "y"],
            level="interaction",
        )
        assert fd.name == "hybrid"
        assert fd.type == FeatureType.INSTANCE
        assert fd.module == "features.hybrid"
        assert fd.function == "compute_hybrid"
        assert fd.columns == ["x", "y"]
        assert fd.level == "interaction"

    def test_defaults(self):
        """Defaults for optional fields."""
        fd = FeatureDef()
        assert fd.name == ""
        assert fd.type == FeatureType.ENTITY
        assert fd.module is None
        assert fd.function is None
        assert fd.columns is None
        assert fd.level is None
        assert fd.enabled is True
