import pytest
from harness.ml.features.schema import FeatureType, FeatureDefinition, FeatureSet


class TestFeatureType:
    def test_enum_values(self):
        assert FeatureType.ENTITY == "entity"
        assert FeatureType.PAIRWISE == "pairwise"
        assert FeatureType.INSTANCE == "instance"
        assert FeatureType.MODEL_OUTPUT == "model_output"

    def test_is_str_enum(self):
        assert isinstance(FeatureType.ENTITY, str)


class TestFeatureDefinition:
    def test_entity_feature(self):
        fd = FeatureDefinition(name="elo", feature_type=FeatureType.ENTITY)
        assert fd.name == "elo"
        assert fd.feature_type == FeatureType.ENTITY
        assert fd.source_column is None
        assert fd.formula is None
        assert fd.model is None
        assert fd.auto_pairwise is True
        assert fd.pairwise_methods == ["diff", "ratio"]
        assert fd.active is True
        assert fd.metadata == {}

    def test_pairwise_with_formula(self):
        fd = FeatureDefinition(
            name="elo_diff",
            feature_type=FeatureType.PAIRWISE,
            formula="entity_a_elo - entity_b_elo",
            auto_pairwise=False,
        )
        assert fd.feature_type == FeatureType.PAIRWISE
        assert fd.formula == "entity_a_elo - entity_b_elo"
        assert fd.auto_pairwise is False

    def test_instance_feature(self):
        fd = FeatureDefinition(
            name="surface",
            feature_type=FeatureType.INSTANCE,
            source_column="court_surface",
        )
        assert fd.feature_type == FeatureType.INSTANCE
        assert fd.source_column == "court_surface"

    def test_model_output_feature(self):
        fd = FeatureDefinition(
            name="predicted_score",
            feature_type=FeatureType.MODEL_OUTPUT,
            model="xgb_v1",
        )
        assert fd.feature_type == FeatureType.MODEL_OUTPUT
        assert fd.model == "xgb_v1"

    def test_active_default_true(self):
        fd = FeatureDefinition(name="x", feature_type=FeatureType.ENTITY)
        assert fd.active is True

    def test_active_can_be_false(self):
        fd = FeatureDefinition(name="x", feature_type=FeatureType.ENTITY, active=False)
        assert fd.active is False


class TestFeatureSet:
    def _make_feature_set(self):
        return FeatureSet(features={
            "elo": FeatureDefinition(name="elo", feature_type=FeatureType.ENTITY),
            "surface": FeatureDefinition(name="surface", feature_type=FeatureType.INSTANCE),
            "disabled": FeatureDefinition(name="disabled", feature_type=FeatureType.ENTITY, active=False),
            "pred": FeatureDefinition(name="pred", feature_type=FeatureType.MODEL_OUTPUT, model="m1"),
        })

    def test_create(self):
        fs = self._make_feature_set()
        assert len(fs.features) == 4

    def test_active_features(self):
        fs = self._make_feature_set()
        active = fs.active_features()
        assert "elo" in active
        assert "surface" in active
        assert "pred" in active
        assert "disabled" not in active
        assert len(active) == 3

    def test_features_by_type(self):
        fs = self._make_feature_set()
        entities = fs.features_by_type(FeatureType.ENTITY)
        assert "elo" in entities
        assert "disabled" in entities
        assert len(entities) == 2

        instances = fs.features_by_type(FeatureType.INSTANCE)
        assert "surface" in instances
        assert len(instances) == 1

    def test_from_yaml_dict(self):
        data = {
            "elo": {"type": "entity", "source_column": "elo_rating"},
            "surface": {"type": "instance", "source_column": "court_surface"},
            "custom_diff": {"type": "pairwise", "formula": "entity_a_elo - entity_b_elo"},
        }
        fs = FeatureSet.from_yaml_dict(data)
        assert len(fs.features) == 3
        assert fs.features["elo"].name == "elo"
        assert fs.features["elo"].feature_type == FeatureType.ENTITY
        assert fs.features["elo"].source_column == "elo_rating"
        assert fs.features["custom_diff"].formula == "entity_a_elo - entity_b_elo"
