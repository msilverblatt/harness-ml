import pytest
from harness.data.expressions.registry import FunctionRegistry


class TestFunctionRegistry:
    def test_list_functions(self):
        registry = FunctionRegistry()
        registry.load_defaults()
        funcs = registry.list_functions()
        assert "abs" in funcs
        assert "zscore" in funcs
        assert "safe_div" in funcs
        assert "rank_pct" in funcs

    def test_get_function(self):
        registry = FunctionRegistry()
        registry.load_defaults()
        func = registry.get("abs")
        assert func is not None
        assert func.name == "abs"
        assert func.description is not None

    def test_register_custom(self):
        registry = FunctionRegistry()
        import numpy as np
        registry.register("my_func", np.square, description="Square a value")
        assert "my_func" in registry.list_functions()

    def test_get_unknown_returns_none(self):
        registry = FunctionRegistry()
        assert registry.get("nonexistent") is None
