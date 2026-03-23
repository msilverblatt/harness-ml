"""DataFrame profiler — computes per-column statistics and infers types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    null_pct: float
    n_unique: int
    inferred_type: str
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None


@dataclass
class DataProfile:
    row_count: int
    column_count: int
    columns: list[ColumnProfile] = field(default_factory=list)
    high_null_columns: list[str] = field(default_factory=list)
    zero_variance_columns: list[str] = field(default_factory=list)


class DataProfiler:
    """Profiles a DataFrame and returns a DataProfile."""

    def profile(self, df: pd.DataFrame, high_null_threshold: float = 50.0) -> DataProfile:
        row_count = len(df)
        column_count = len(df.columns)
        columns: list[ColumnProfile] = []
        high_null: list[str] = []
        zero_var: list[str] = []

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_pct = (null_count / row_count * 100.0) if row_count > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))
            dtype_str = str(series.dtype)

            inferred = self._infer_type(series, n_unique)

            mean = std = min_ = max_ = median = None
            if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
                numeric = series.dropna()
                if len(numeric) > 0:
                    mean = float(numeric.mean())
                    std = float(numeric.std())
                    min_ = float(numeric.min())
                    max_ = float(numeric.max())
                    median = float(numeric.median())
                if std == 0.0 or (std is None and len(numeric) > 0 and len(numeric.unique()) == 1):
                    zero_var.append(col)
            elif pd.api.types.is_bool_dtype(series):
                # Check zero variance for bool too
                if series.dropna().nunique() <= 1:
                    zero_var.append(col)

            cp = ColumnProfile(
                name=col,
                dtype=dtype_str,
                null_count=null_count,
                null_pct=null_pct,
                n_unique=n_unique,
                inferred_type=inferred,
                mean=mean,
                std=std,
                min=min_,
                max=max_,
                median=median,
            )
            columns.append(cp)

            if null_pct >= high_null_threshold:
                high_null.append(col)

        return DataProfile(
            row_count=row_count,
            column_count=column_count,
            columns=columns,
            high_null_columns=high_null,
            zero_variance_columns=zero_var,
        )

    def _infer_type(self, series: pd.Series, n_unique: int) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            if n_unique == 2:
                return "binary"
            # Only classify as categorical if it looks like an enum, not just a small dataset.
            # Use ratio: if >50% of values are unique, it's numeric, not categorical.
            # Also check if it's a float — floats are almost never categorical.
            row_count = len(series.dropna())
            is_float = pd.api.types.is_float_dtype(series)
            unique_ratio = n_unique / row_count if row_count > 0 else 0
            if not is_float and n_unique <= 20 and unique_ratio < 0.5:
                return "categorical"
            return "numeric"
        # object / string columns
        row_count = len(series)
        if row_count == 0:
            return "categorical"
        unique_ratio = n_unique / row_count * 100.0
        if unique_ratio > 90:
            return "id"
        if n_unique <= 50:
            return "categorical"
        return "high_cardinality"
