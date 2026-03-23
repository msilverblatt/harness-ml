import pandas as pd
import numpy as np
import pytest
from harness.data.expressions.engine import ExpressionEngine


class TestMathFunctions:
    def test_abs(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "abs(score - 90)")
        assert all(result >= 0)

    def test_log(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "log(score)")
        assert all(result > 0)

    def test_sqrt(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "sqrt(score)")
        assert all(result > 0)


class TestStatsFunctions:
    def test_zscore(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "zscore(score)")
        assert abs(result.mean()) < 1e-10

    def test_rank_pct(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "rank_pct(score)")
        assert result.max() <= 1.0
        assert result.min() > 0.0


class TestComparisonFunctions:
    def test_safe_div(self, sample_df):
        engine = ExpressionEngine()
        df = sample_df.copy()
        df["zero"] = 0
        result = engine.evaluate(df, "safe_div(score, zero)")
        assert all(result == 0.0)

    def test_where(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "where(score > 90, 1, 0)")
        assert result.sum() == 2
