import pandas as pd
import pytest
import numpy as np
from harness.data.expressions.engine import ExpressionEngine


class TestExpressionEngine:
    def test_simple_arithmetic(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "score * 2")
        assert result.iloc[0] == 170.0

    def test_column_reference(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "score + 10")
        assert result.iloc[0] == 95.0

    def test_registered_function(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "abs(score - 90)")
        assert all(result >= 0)

    def test_nested_functions(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "abs(zscore(score))")
        assert all(result >= 0)

    def test_rejects_dangerous_builtins(self, sample_df):
        engine = ExpressionEngine()
        with pytest.raises((ValueError, TypeError, KeyError)):
            engine.evaluate(sample_df, "__import__('os').system('ls')")

    def test_rejects_attribute_access(self, sample_df):
        engine = ExpressionEngine()
        with pytest.raises((ValueError, TypeError, KeyError, AttributeError)):
            engine.evaluate(sample_df, "score.__class__.__bases__")
