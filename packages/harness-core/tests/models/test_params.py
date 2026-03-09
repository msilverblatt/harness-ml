"""Tests for per-model Pydantic param schemas."""
from __future__ import annotations

import pytest
from harnessml.core.models.params import (
    CatBoostParams,
    ElasticNetParams,
    LightGBMParams,
    LogisticParams,
    MLPParams,
    RandomForestParams,
    TabNetParams,
    XGBoostParams,
    get_params_schema,
)
from pydantic import ValidationError


# -------------------------------------------------------------------
# XGBoost
# -------------------------------------------------------------------
class TestXGBoostParams:
    def test_xgboost_params_defaults(self):
        p = XGBoostParams()
        assert p.n_estimators == 100
        assert p.max_depth == 6
        assert p.learning_rate == 0.1
        assert p.subsample == 1.0
        assert p.colsample_bytree == 1.0
        assert p.reg_alpha == 0.0
        assert p.reg_lambda == 1.0
        assert p.gamma == 0.0
        assert p.min_child_weight == 1.0
        assert p.scale_pos_weight == 1.0
        assert p.early_stopping_rounds is None

    def test_xgboost_params_rejects_negative_depth(self):
        with pytest.raises(ValidationError):
            XGBoostParams(max_depth=-1)

    def test_xgboost_params_rejects_zero_depth(self):
        with pytest.raises(ValidationError):
            XGBoostParams(max_depth=0)

    def test_xgboost_params_accepts_custom(self):
        p = XGBoostParams(n_estimators=500, max_depth=10, learning_rate=0.05)
        assert p.n_estimators == 500
        assert p.max_depth == 10
        assert p.learning_rate == 0.05

    def test_xgboost_extra_params_forwarded(self):
        p = XGBoostParams(tree_method="gpu_hist")
        assert p.tree_method == "gpu_hist"


# -------------------------------------------------------------------
# LightGBM
# -------------------------------------------------------------------
class TestLightGBMParams:
    def test_lightgbm_params_defaults(self):
        p = LightGBMParams()
        assert p.n_estimators == 100
        assert p.max_depth == -1
        assert p.learning_rate == 0.1
        assert p.num_leaves == 31
        assert p.verbosity == -1

    def test_lightgbm_rejects_zero_learning_rate(self):
        with pytest.raises(ValidationError):
            LightGBMParams(learning_rate=0)


# -------------------------------------------------------------------
# CatBoost
# -------------------------------------------------------------------
class TestCatBoostParams:
    def test_catboost_params_defaults(self):
        p = CatBoostParams()
        assert p.iterations == 1000
        assert p.depth == 6
        assert p.learning_rate == 0.03
        assert p.allow_writing_files is False
        assert p.verbose == 0


# -------------------------------------------------------------------
# Random Forest
# -------------------------------------------------------------------
class TestRandomForestParams:
    def test_random_forest_defaults(self):
        p = RandomForestParams()
        assert p.n_estimators == 100
        assert p.max_depth is None
        assert p.max_features == "sqrt"


# -------------------------------------------------------------------
# Logistic
# -------------------------------------------------------------------
class TestLogisticParams:
    def test_logistic_defaults(self):
        p = LogisticParams()
        assert p.C == 1.0
        assert p.penalty == "l2"
        assert p.solver == "lbfgs"
        assert p.max_iter == 100


# -------------------------------------------------------------------
# ElasticNet
# -------------------------------------------------------------------
class TestElasticNetParams:
    def test_elastic_net_defaults(self):
        p = ElasticNetParams()
        assert p.l1_ratio == 0.5
        assert p.penalty == "elasticnet"
        assert p.solver == "saga"
        assert p.normalize is False


# -------------------------------------------------------------------
# MLP
# -------------------------------------------------------------------
class TestMLPParams:
    def test_mlp_params_defaults(self):
        p = MLPParams()
        assert p.hidden_layers == [128, 64]
        assert p.dropout == 0.0
        assert p.learning_rate == 0.001
        assert p.epochs == 10
        assert p.batch_size == 32
        assert p.n_seeds == 1
        assert p.normalize is False
        assert p.batch_norm is False
        assert p.weight_decay == 0.0
        assert p.early_stopping_rounds is None
        assert p.seed_stride == 1
        assert p.seed == 0

    def test_mlp_params_rejects_empty_layers(self):
        with pytest.raises(ValidationError, match="hidden_layers must not be empty"):
            MLPParams(hidden_layers=[])

    def test_mlp_params_rejects_negative_layer_size(self):
        with pytest.raises(ValidationError, match="hidden layer sizes must be >= 1"):
            MLPParams(hidden_layers=[128, 0])

    def test_mlp_accepts_custom_layers(self):
        p = MLPParams(hidden_layers=[256, 128, 64])
        assert p.hidden_layers == [256, 128, 64]


# -------------------------------------------------------------------
# TabNet
# -------------------------------------------------------------------
class TestTabNetParams:
    def test_tabnet_defaults(self):
        p = TabNetParams()
        assert p.n_d == 8
        assert p.n_a == 8
        assert p.n_steps == 3
        assert p.max_epochs == 200
        assert p.normalize is False


# -------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------
class TestParamsRegistry:
    def test_get_params_schema_by_type(self):
        assert get_params_schema("xgboost") is XGBoostParams
        assert get_params_schema("lightgbm") is LightGBMParams
        assert get_params_schema("catboost") is CatBoostParams
        assert get_params_schema("random_forest") is RandomForestParams
        assert get_params_schema("logistic_regression") is LogisticParams
        assert get_params_schema("elastic_net") is ElasticNetParams
        assert get_params_schema("mlp") is MLPParams
        assert get_params_schema("tabnet") is TabNetParams

    def test_get_params_schema_unknown_returns_none(self):
        assert get_params_schema("nonexistent_model") is None

    def test_params_to_dict(self):
        p = XGBoostParams(n_estimators=200, max_depth=8)
        d = p.model_dump()
        assert d["n_estimators"] == 200
        assert d["max_depth"] == 8
        assert isinstance(d, dict)

    def test_params_to_dict_includes_extra(self):
        p = XGBoostParams(tree_method="gpu_hist")
        d = p.model_dump()
        assert d["tree_method"] == "gpu_hist"
