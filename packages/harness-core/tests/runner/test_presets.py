"""Tests for model presets."""
from __future__ import annotations

import pytest
from harnessml.core.runner.presets import apply_preset, get_preset, list_presets
from harnessml.core.runner.schema import ModelDef

# -----------------------------------------------------------------------
# list_presets
# -----------------------------------------------------------------------

class TestListPresets:
    def test_returns_all_names(self):
        names = list_presets()
        assert "xgboost_classifier" in names
        assert "xgboost_regressor" in names
        assert "logistic_regression" in names
        assert "catboost_classifier" in names
        assert "lightgbm_classifier" in names
        assert "mlp_classifier" in names
        assert "mlp_regressor" in names

    def test_returns_sorted(self):
        names = list_presets()
        assert names == sorted(names)


# -----------------------------------------------------------------------
# get_preset
# -----------------------------------------------------------------------

class TestGetPreset:
    def test_returns_deep_copy(self):
        a = get_preset("xgboost_classifier")
        b = get_preset("xgboost_classifier")
        a["params"]["learning_rate"] = 999
        assert b["params"]["learning_rate"] != 999

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown preset"):
            get_preset("nonexistent_model")

    def test_error_message_lists_available(self):
        with pytest.raises(KeyError, match="xgboost_classifier"):
            get_preset("bad")


# -----------------------------------------------------------------------
# apply_preset — overrides
# -----------------------------------------------------------------------

class TestApplyPreset:
    def test_no_overrides_returns_base(self):
        result = apply_preset("logistic_regression")
        assert result["type"] == "logistic_regression"
        assert result["params"]["C"] == 1.0

    def test_override_top_level_key(self):
        result = apply_preset("xgboost_classifier", {"mode": "regressor"})
        assert result["mode"] == "regressor"
        # params should still be present from preset
        assert "max_depth" in result["params"]

    def test_override_params_merges(self):
        """Override individual params without losing the rest."""
        result = apply_preset(
            "xgboost_classifier",
            {"params": {"learning_rate": 0.1}},
        )
        # Overridden
        assert result["params"]["learning_rate"] == 0.1
        # Original params preserved
        assert result["params"]["max_depth"] == 4
        assert result["params"]["n_estimators"] == 1000

    def test_override_adds_new_param(self):
        result = apply_preset(
            "xgboost_classifier",
            {"params": {"gamma": 0.5}},
        )
        assert result["params"]["gamma"] == 0.5
        assert result["params"]["max_depth"] == 4

    def test_override_replaces_non_params_key(self):
        result = apply_preset(
            "mlp_classifier",
            {"n_seeds": 10},
        )
        assert result["n_seeds"] == 10

    def test_add_features_via_override(self):
        result = apply_preset(
            "xgboost_classifier",
            {"features": ["diff_prior", "diff_adj_em"]},
        )
        assert result["features"] == ["diff_prior", "diff_adj_em"]


# -----------------------------------------------------------------------
# All presets produce valid ModelDef
# -----------------------------------------------------------------------

class TestPresetsValidateAsModelDef:
    @pytest.mark.parametrize("name", list_presets())
    def test_preset_validates(self, name):
        """Every preset (with a dummy features list) must pass ModelDef validation."""
        config = apply_preset(name, {"features": ["diff_prior"]})
        model = ModelDef(**config)
        assert model.type in (
            "xgboost", "xgboost_regression", "logistic_regression",
            "catboost", "lightgbm", "mlp",
        )

    def test_xgboost_regressor_has_margin_prediction_type(self):
        config = get_preset("xgboost_regressor")
        assert config["prediction_type"] == "margin"
        assert config["mode"] == "regressor"

    def test_mlp_regressor_has_multiple_seeds(self):
        config = get_preset("mlp_regressor")
        assert config["n_seeds"] >= 3

    def test_mlp_classifier_has_seeds(self):
        config = get_preset("mlp_classifier")
        assert config["n_seeds"] >= 2
