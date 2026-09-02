"""Cast step — change column data types."""
from __future__ import annotations
import pandas as pd

NAME = "cast"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    columns = params.get("columns")
    if not columns or not isinstance(columns, dict):
        raise ValueError("cast step requires 'columns' dict of {column: type}")
    result = df.copy()
    for col, dtype in columns.items():
        if col not in result.columns:
            raise ValueError(f"Column not found: {col}")
        result[col] = result[col].astype(dtype)
    return result
