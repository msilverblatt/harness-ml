import pytest
from harness.ml.config.models import SingleModelConfig, ModelsConfig


class TestSingleModelConfig:
    def test_required_fields(self):
        cfg = SingleModelConfig(name="m1", model_type="logistic")
        assert cfg.name == "m1"
        assert cfg.model_type == "logistic"

    def test_defaults(self):
        cfg = SingleModelConfig(name="m1", model_type="logistic")
        assert cfg.params == {}
        assert cfg.features == []
        assert cfg.active is True
        assert cfg.include_in_ensemble is True
        assert cfg.n_seeds == 1
        assert cfg.depends_on == []
        assert cfg.provides is None
        assert cfg.provides_level == "instance"
        assert cfg.training_filter is None
        assert cfg.zero_fill_features == []
        assert cfg.class_weight is None
        assert cfg.augment_symmetry is False

    def test_custom_params(self):
        cfg = SingleModelConfig(
            name="xgb",
            model_type="xgboost",
            params={"n_estimators": 100, "max_depth": 6},
            features=["feat_a", "feat_b"],
            n_seeds=5,
        )
        assert cfg.params["n_estimators"] == 100
        assert cfg.features == ["feat_a", "feat_b"]
        assert cfg.n_seeds == 5

    def test_class_weight_string(self):
        cfg = SingleModelConfig(name="m", model_type="logistic", class_weight="balanced")
        assert cfg.class_weight == "balanced"

    def test_class_weight_dict(self):
        cfg = SingleModelConfig(name="m", model_type="logistic", class_weight={0: 1, 1: 3})
        assert cfg.class_weight == {0: 1, 1: 3}

    def test_all_fields_accessible(self):
        cfg = SingleModelConfig(name="m", model_type="logistic")
        _ = cfg.name
        _ = cfg.model_type
        _ = cfg.params
        _ = cfg.features
        _ = cfg.active
        _ = cfg.include_in_ensemble
        _ = cfg.n_seeds
        _ = cfg.depends_on
        _ = cfg.provides
        _ = cfg.provides_level
        _ = cfg.training_filter
        _ = cfg.zero_fill_features
        _ = cfg.class_weight
        _ = cfg.augment_symmetry


class TestModelsConfig:
    def test_empty(self):
        cfg = ModelsConfig()
        assert cfg.models == {}

    def test_from_yaml_dict(self):
        data = {
            "logistic_base": {
                "model_type": "logistic",
                "features": ["x1", "x2"],
            },
            "xgb": {
                "model_type": "xgboost",
                "params": {"n_estimators": 50},
                "active": False,
            },
        }
        cfg = ModelsConfig.from_yaml_dict(data)
        assert "logistic_base" in cfg.models
        assert "xgb" in cfg.models
        assert cfg.models["logistic_base"].name == "logistic_base"
        assert cfg.models["logistic_base"].model_type == "logistic"
        assert cfg.models["logistic_base"].features == ["x1", "x2"]
        assert cfg.models["xgb"].name == "xgb"
        assert cfg.models["xgb"].params == {"n_estimators": 50}
        assert cfg.models["xgb"].active is False

    def test_from_yaml_dict_defaults_filled(self):
        data = {"m": {"model_type": "logistic"}}
        cfg = ModelsConfig.from_yaml_dict(data)
        assert cfg.models["m"].n_seeds == 1
        assert cfg.models["m"].active is True

    def test_from_yaml_dict_empty(self):
        cfg = ModelsConfig.from_yaml_dict({})
        assert cfg.models == {}
