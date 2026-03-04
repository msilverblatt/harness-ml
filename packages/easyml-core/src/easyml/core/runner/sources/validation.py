"""Schema validation for data sources -- optional Pandera integration."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationViolation:
    """A single validation violation for a column or the dataframe."""

    column: str
    message: str
    severity: str = "error"  # "error" or "warning"


def validate_source(source_def, df: pd.DataFrame) -> list[ValidationViolation]:
    """Validate a DataFrame against a source's schema definition.

    Uses Pandera if available and configured, otherwise falls back to basic
    checks on required_columns, types, and min_rows.
    """
    violations: list[ValidationViolation] = []
    schema = source_def.schema
    if not schema:
        return violations

    # Check required columns
    required = schema.get("required_columns", [])
    actual = set(df.columns)
    for col in required:
        if col not in actual:
            violations.append(ValidationViolation(col, f"Missing required column: {col}"))

    # Check types
    types = schema.get("types", {})
    for col, expected_dtype in types.items():
        if col not in df.columns:
            continue
        try:
            df[col].astype(expected_dtype)
        except (ValueError, TypeError):
            violations.append(ValidationViolation(
                col, f"Column '{col}' cannot be cast to {expected_dtype}"
            ))

    # Check min_rows
    min_rows = schema.get("min_rows", 0)
    if min_rows and len(df) < min_rows:
        violations.append(ValidationViolation(
            "__dataframe__",
            f"Expected at least {min_rows} rows, got {len(df)}",
        ))

    # Try Pandera if available and schema defines a pandera_schema
    pandera_schema = schema.get("pandera_schema")
    if pandera_schema:
        try:
            import pandera as pa

            pa_schema = pa.DataFrameSchema.from_dict(pandera_schema)
            pa_schema.validate(df)
        except ImportError:
            logger.info("Pandera not installed, skipping Pandera validation")
        except Exception as e:
            violations.append(ValidationViolation("__pandera__", str(e)))

    return violations
