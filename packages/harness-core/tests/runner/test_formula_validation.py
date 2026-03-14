"""Tests for formula syntax validation at feature add time."""

from harnessml.core.runner.config_writer.features import _validate_formula_syntax


class TestValidateFormulaSyntax:
    """Unit tests for _validate_formula_syntax helper."""

    def test_valid_simple_formula(self):
        assert _validate_formula_syntax("a + b") is None

    def test_valid_arithmetic(self):
        assert _validate_formula_syntax("(home_elo - away_elo) / 100") is None

    def test_valid_function_call(self):
        assert _validate_formula_syntax("abs(x - y)") is None

    def test_valid_ternary(self):
        assert _validate_formula_syntax("x if x > 0 else 0") is None

    def test_valid_comparison(self):
        assert _validate_formula_syntax("a > b") is None

    def test_valid_complex_formula(self):
        assert _validate_formula_syntax("(a + b) * c / (d - e + 1)") is None

    def test_invalid_unbalanced_parens(self):
        result = _validate_formula_syntax("(a + b")
        assert result is not None
        assert "Invalid formula syntax" in result

    def test_invalid_trailing_operator(self):
        result = _validate_formula_syntax("a + ")
        assert result is not None
        assert "Invalid formula syntax" in result

    def test_invalid_mismatched_brackets(self):
        result = _validate_formula_syntax("a[0 + b")
        assert result is not None
        assert "Invalid formula syntax" in result

    def test_invalid_empty_string(self):
        result = _validate_formula_syntax("")
        assert result is not None
        assert "Invalid formula syntax" in result

    def test_error_includes_formula_text(self):
        result = _validate_formula_syntax("a + (")
        assert result is not None
        assert "a + (" in result

    def test_valid_attribute_access(self):
        assert _validate_formula_syntax("df.col1 - df.col2") is None

    def test_valid_list_index(self):
        assert _validate_formula_syntax("a[0] + b[1]") is None
