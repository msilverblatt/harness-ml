"""Schema validator — validates a DataFrame against expectations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SchemaValidator:
    """Validates a DataFrame against a declared schema."""

    def __init__(
        self,
        required_columns: list[str] | None = None,
        column_types: dict[str, Any] | None = None,
        no_null_columns: list[str] | None = None,
        min_rows: int | None = None,
    ) -> None:
        self._required_columns: list[str] = required_columns or []
        self._column_types: dict[str, Any] = column_types or {}
        self._no_null_columns: list[str] = no_null_columns or []
        self._min_rows: int | None = min_rows

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        # min_rows check
        if self._min_rows is not None and len(df) < self._min_rows:
            errors.append(
                f"DataFrame has {len(df)} rows, minimum required is {self._min_rows}"
            )

        # required columns
        for col in self._required_columns:
            if col not in df.columns:
                errors.append(f"Required column missing: '{col}'")

        # column type checks (only for columns that exist)
        for col, expected_dtype in self._column_types.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            expected_str = str(expected_dtype)
            if actual != expected_str:
                warnings.append(
                    f"Column '{col}' has dtype '{actual}', expected '{expected_str}'"
                )

        # no-null checks
        for col in self._no_null_columns:
            if col not in df.columns:
                continue
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                errors.append(
                    f"Column '{col}' must not contain nulls, found {null_count}"
                )

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
