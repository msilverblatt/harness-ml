import pytest
from harness.ml.config.ensemble import EnsembleConfig


class TestEnsembleConfig:
    def test_defaults(self):
        cfg = EnsembleConfig()
        assert cfg.method == "stacked"
        assert cfg.meta_learner_type == "logistic"
        assert cfg.meta_learner_params == {}
        assert cfg.exclude_models == []
        assert cfg.calibration == "none"
        assert cfg.pre_calibration == {}
        assert cfg.temperature == 1.0
        assert cfg.clip_floor is None
        assert cfg.meta_features == []
        assert cfg.prior_feature is None
        assert cfg.conformal_alpha is None

    def test_custom_values(self):
        cfg = EnsembleConfig(
            method="weighted_average",
            meta_learner_type="ridge",
            meta_learner_params={"alpha": 0.1},
            exclude_models=["bad_model"],
            calibration="isotonic",
            pre_calibration={"xgb": "platt"},
            temperature=0.8,
            clip_floor=0.05,
            meta_features=["prior"],
            prior_feature="base_rate",
        )
        assert cfg.method == "weighted_average"
        assert cfg.meta_learner_type == "ridge"
        assert cfg.meta_learner_params == {"alpha": 0.1}
        assert cfg.exclude_models == ["bad_model"]
        assert cfg.calibration == "isotonic"
        assert cfg.pre_calibration == {"xgb": "platt"}
        assert cfg.temperature == 0.8
        assert cfg.clip_floor == 0.05
        assert cfg.meta_features == ["prior"]
        assert cfg.prior_feature == "base_rate"

    def test_conformal_alpha_validation(self):
        assert EnsembleConfig(conformal_alpha=0.1).conformal_alpha == 0.1
        with pytest.raises(ValueError):
            EnsembleConfig(conformal_alpha=1.0)

    def test_all_fields_accessible(self):
        cfg = EnsembleConfig()
        _ = cfg.method
        _ = cfg.meta_learner_type
        _ = cfg.meta_learner_params
        _ = cfg.exclude_models
        _ = cfg.calibration
        _ = cfg.pre_calibration
        _ = cfg.temperature
        _ = cfg.clip_floor
        _ = cfg.meta_features
        _ = cfg.prior_feature
