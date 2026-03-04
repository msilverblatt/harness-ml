"""Tests for source schema validation."""
from __future__ import annotations

import pandas as pd

from easyml.core.runner.sources.registry import SourceDef
from easyml.core.runner.sources.validation import validate_source, ValidationViolation


def _make_source(schema: dict) -> SourceDef:
    return SourceDef(name="test", source_type="file", schema=schema)


def test_no_schema_returns_no_violations():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    src = SourceDef(name="test", source_type="file")
    violations = validate_source(src, df)
    assert violations == []


def test_required_columns_present():
    df = pd.DataFrame({"team_id": [1], "score": [80]})
    src = _make_source({"required_columns": ["team_id", "score"]})
    violations = validate_source(src, df)
    assert violations == []


def test_required_columns_missing():
    df = pd.DataFrame({"team_id": [1]})
    src = _make_source({"required_columns": ["team_id", "score", "margin"]})
    violations = validate_source(src, df)
    assert len(violations) == 2
    missing_cols = {v.column for v in violations}
    assert missing_cols == {"score", "margin"}
    assert all(v.severity == "error" for v in violations)


def test_type_check_passes():
    df = pd.DataFrame({"value": [1, 2, 3]})
    src = _make_source({"types": {"value": "float"}})
    violations = validate_source(src, df)
    assert violations == []


def test_type_check_fails():
    df = pd.DataFrame({"value": ["hello", "world"]})
    src = _make_source({"types": {"value": "int"}})
    violations = validate_source(src, df)
    assert len(violations) == 1
    assert violations[0].column == "value"
    assert "cannot be cast" in violations[0].message


def test_type_check_skips_missing_column():
    """Type check for a column not in the DataFrame is silently skipped."""
    df = pd.DataFrame({"a": [1]})
    src = _make_source({"types": {"b": "float"}})
    violations = validate_source(src, df)
    assert violations == []


def test_min_rows_passes():
    df = pd.DataFrame({"a": range(100)})
    src = _make_source({"min_rows": 50})
    violations = validate_source(src, df)
    assert violations == []


def test_min_rows_fails():
    df = pd.DataFrame({"a": range(10)})
    src = _make_source({"min_rows": 50})
    violations = validate_source(src, df)
    assert len(violations) == 1
    assert "__dataframe__" == violations[0].column
    assert "at least 50 rows" in violations[0].message


def test_combined_violations():
    df = pd.DataFrame({"a": ["x"]})
    src = _make_source({
        "required_columns": ["a", "b"],
        "types": {"a": "int"},
        "min_rows": 10,
    })
    violations = validate_source(src, df)
    # missing "b", type cast "a" fails, min_rows fails
    assert len(violations) == 3


def test_validation_violation_dataclass():
    v = ValidationViolation(column="score", message="bad type", severity="warning")
    assert v.column == "score"
    assert v.message == "bad type"
    assert v.severity == "warning"
