"""Tests for sweep expansion."""
from __future__ import annotations

import pytest
from harnessml.core.runner.optimization.sweep import expand_sweep, get_nested_key, set_nested_key

# -----------------------------------------------------------------------
# set_nested_key
# -----------------------------------------------------------------------

class TestSetNestedKey:
    def test_single_level(self):
        d: dict = {}
        set_nested_key(d, "foo", 1)
        assert d == {"foo": 1}

    def test_multi_level(self):
        d: dict = {}
        set_nested_key(d, "a.b.c", 42)
        assert d == {"a": {"b": {"c": 42}}}

    def test_preserves_existing_siblings(self):
        d = {"a": {"x": 10}}
        set_nested_key(d, "a.y", 20)
        assert d == {"a": {"x": 10, "y": 20}}

    def test_overwrites_non_dict_intermediate(self):
        d = {"a": "scalar"}
        set_nested_key(d, "a.b", 5)
        assert d == {"a": {"b": 5}}

    def test_deeply_nested(self):
        d: dict = {}
        set_nested_key(d, "models.xgb_core.params.learning_rate", 0.05)
        assert d["models"]["xgb_core"]["params"]["learning_rate"] == 0.05


# -----------------------------------------------------------------------
# get_nested_key
# -----------------------------------------------------------------------

class TestGetNestedKey:
    def test_existing_path(self):
        d = {"a": {"b": {"c": 42}}}
        assert get_nested_key(d, "a.b.c") == 42

    def test_missing_path_returns_default(self):
        d = {"a": {"b": 1}}
        assert get_nested_key(d, "a.x.y") is None
        assert get_nested_key(d, "a.x.y", "fallback") == "fallback"

    def test_single_level(self):
        d = {"foo": "bar"}
        assert get_nested_key(d, "foo") == "bar"


# -----------------------------------------------------------------------
# expand_sweep — no sweep key
# -----------------------------------------------------------------------

class TestExpandSweepPassthrough:
    def test_no_sweep_key_returns_single_element_list(self):
        overlay = {"description": "baseline", "models": {"a": {"type": "xgboost"}}}
        result = expand_sweep(overlay)
        assert len(result) == 1
        assert result[0] == overlay

    def test_empty_overlay_returns_single(self):
        result = expand_sweep({})
        assert len(result) == 1


# -----------------------------------------------------------------------
# expand_sweep — single axis
# -----------------------------------------------------------------------

class TestExpandSweepSingleAxis:
    def test_single_axis_produces_correct_count(self):
        overlay = {
            "description": "lr test",
            "sweep": {
                "key": "models.xgb.params.learning_rate",
                "values": [0.01, 0.02, 0.05],
            },
        }
        variants = expand_sweep(overlay)
        assert len(variants) == 3

    def test_sweep_key_removed_from_variants(self):
        overlay = {
            "sweep": {"key": "a.b", "values": [1, 2]},
        }
        for v in expand_sweep(overlay):
            assert "sweep" not in v

    def test_values_injected_at_correct_path(self):
        overlay = {
            "sweep": {"key": "models.xgb.params.lr", "values": [0.01, 0.05]},
        }
        variants = expand_sweep(overlay)
        assert get_nested_key(variants[0], "models.xgb.params.lr") == 0.01
        assert get_nested_key(variants[1], "models.xgb.params.lr") == 0.05

    def test_descriptions_include_swept_values(self):
        overlay = {
            "description": "LR sweep",
            "sweep": {"key": "params.lr", "values": [0.01, 0.1]},
        }
        variants = expand_sweep(overlay)
        assert "lr=0.01" in variants[0]["description"]
        assert "lr=0.1" in variants[1]["description"]
        # base description is preserved as prefix
        assert variants[0]["description"].startswith("LR sweep")

    def test_default_description_when_none_provided(self):
        overlay = {
            "sweep": {"key": "x", "values": [1]},
        }
        variants = expand_sweep(overlay)
        assert "sweep" in variants[0]["description"].lower()


# -----------------------------------------------------------------------
# expand_sweep — multi axis (cartesian product)
# -----------------------------------------------------------------------

class TestExpandSweepMultiAxis:
    def test_cartesian_product_count(self):
        overlay = {
            "description": "grid",
            "sweep": [
                {"key": "models.xgb.params.lr", "values": [0.01, 0.05]},
                {"key": "models.xgb.params.max_depth", "values": [3, 4, 5]},
            ],
        }
        variants = expand_sweep(overlay)
        assert len(variants) == 2 * 3  # 6

    def test_all_combinations_present(self):
        overlay = {
            "sweep": [
                {"key": "a", "values": [1, 2]},
                {"key": "b", "values": ["x", "y"]},
            ],
        }
        variants = expand_sweep(overlay)
        combos = [(v["a"], v["b"]) for v in variants]
        assert (1, "x") in combos
        assert (1, "y") in combos
        assert (2, "x") in combos
        assert (2, "y") in combos

    def test_multi_axis_descriptions_include_all_values(self):
        overlay = {
            "description": "grid",
            "sweep": [
                {"key": "lr", "values": [0.01]},
                {"key": "depth", "values": [4]},
            ],
        }
        variants = expand_sweep(overlay)
        assert "lr=0.01" in variants[0]["description"]
        assert "depth=4" in variants[0]["description"]


# -----------------------------------------------------------------------
# expand_sweep — variants are independent copies
# -----------------------------------------------------------------------

class TestExpandSweepIsolation:
    def test_mutating_one_variant_does_not_affect_others(self):
        overlay = {
            "sweep": {"key": "a.b", "values": [1, 2]},
            "shared": {"nested": "original"},
        }
        variants = expand_sweep(overlay)
        variants[0]["shared"]["nested"] = "mutated"
        assert variants[1]["shared"]["nested"] == "original"


# -----------------------------------------------------------------------
# expand_sweep — error cases
# -----------------------------------------------------------------------

class TestExpandSweepErrors:
    def test_invalid_sweep_type(self):
        with pytest.raises(ValueError, match="dict or list"):
            expand_sweep({"sweep": "bad"})

    def test_missing_key(self):
        with pytest.raises(ValueError, match="key"):
            expand_sweep({"sweep": {"values": [1, 2]}})

    def test_missing_values(self):
        with pytest.raises(ValueError, match="values"):
            expand_sweep({"sweep": {"key": "a"}})

    def test_empty_values_list(self):
        with pytest.raises(ValueError, match="non-empty"):
            expand_sweep({"sweep": {"key": "a", "values": []}})
