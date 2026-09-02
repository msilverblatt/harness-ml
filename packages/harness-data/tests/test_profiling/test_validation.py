"""Tests for SchemaValidator."""
import numpy as np
import pandas as pd
import pytest

from harness.data.profiling.validation import SchemaValidator, ValidationResult


@pytest.fixture
def basic_df():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"],
        "score": [1.0, 2.0, 3.0],
    })


def test_valid_result_type(basic_df):
    v = SchemaValidator(required_columns=["id", "name"])
    result = v.validate(basic_df)
    assert isinstance(result, ValidationResult)
    assert result.is_valid is True
    assert result.errors == []


def test_required_columns_present(basic_df):
    v = SchemaValidator(required_columns=["id", "name", "score"])
    result = v.validate(basic_df)
    assert result.is_valid is True


def test_missing_required_column(basic_df):
    v = SchemaValidator(required_columns=["id", "missing_col"])
    result = v.validate(basic_df)
    assert result.is_valid is False
    assert any("missing_col" in e for e in result.errors)


def test_no_null_columns_pass(basic_df):
    v = SchemaValidator(no_null_columns=["id", "name"])
    result = v.validate(basic_df)
    assert result.is_valid is True


def test_no_null_columns_fail():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    v = SchemaValidator(no_null_columns=["a"])
    result = v.validate(df)
    assert result.is_valid is False
    assert any("'a'" in e for e in result.errors)


def test_min_rows_pass(basic_df):
    v = SchemaValidator(min_rows=2)
    result = v.validate(basic_df)
    assert result.is_valid is True


def test_min_rows_fail():
    df = pd.DataFrame({"x": [1]})
    v = SchemaValidator(min_rows=5)
    result = v.validate(df)
    assert result.is_valid is False
    assert any("minimum" in e for e in result.errors)


def test_column_type_mismatch_is_warning(basic_df):
    v = SchemaValidator(column_types={"score": "int64"})
    result = v.validate(basic_df)
    # type mismatch → warning, not error
    assert result.is_valid is True
    assert any("score" in w for w in result.warnings)


def test_column_type_match_no_warning(basic_df):
    v = SchemaValidator(column_types={"score": "float64"})
    result = v.validate(basic_df)
    assert result.warnings == [] or not any("score" in w for w in result.warnings)


def test_missing_column_in_type_check_skipped(basic_df):
    # column_types entry for non-existent column should not raise
    v = SchemaValidator(column_types={"nonexistent": "int64"})
    result = v.validate(basic_df)
    assert result.is_valid is True


def test_multiple_errors_accumulated():
    df = pd.DataFrame({"a": [1]})
    v = SchemaValidator(
        required_columns=["b", "c"],
        min_rows=10,
    )
    result = v.validate(df)
    assert result.is_valid is False
    assert len(result.errors) >= 3  # min_rows + 2 missing cols


def test_no_null_column_missing_from_df():
    df = pd.DataFrame({"a": [1, 2]})
    v = SchemaValidator(no_null_columns=["nonexistent"])
    result = v.validate(df)
    # Column doesn't exist — skip check, no error
    assert result.is_valid is True


def test_empty_validator(basic_df):
    v = SchemaValidator()
    result = v.validate(basic_df)
    assert result.is_valid is True
    assert result.errors == []
    assert result.warnings == []
