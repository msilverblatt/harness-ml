import pytest
from harness.data.expressions.validator import ExpressionValidator


class TestExpressionValidator:
    def test_valid_expression(self):
        validator = ExpressionValidator()
        schema = {"columns": ["score", "rating"], "column_types": {"score": "float64", "rating": "float64"}}
        result = validator.validate("score * 2", schema)
        assert result.is_valid

    def test_missing_column(self):
        validator = ExpressionValidator()
        schema = {"columns": ["score"], "column_types": {"score": "float64"}}
        result = validator.validate("momentum * 2", schema)
        assert not result.is_valid
        assert "momentum" in result.errors[0]
        assert "score" in result.suggestion

    def test_unknown_function(self):
        validator = ExpressionValidator()
        schema = {"columns": ["score"], "column_types": {"score": "float64"}}
        result = validator.validate("bad_func(score)", schema)
        assert not result.is_valid
        assert "bad_func" in result.errors[0]

    def test_valid_function(self):
        validator = ExpressionValidator()
        schema = {"columns": ["score"], "column_types": {"score": "float64"}}
        result = validator.validate("abs(score - 50)", schema)
        assert result.is_valid
