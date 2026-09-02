"""Null indicator step — create binary columns indicating missing values."""
from __future__ import annotations
import pandas as pd

NAME = "null_indicator"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    columns = params.get("columns")
    prefix = params.get("prefix", "missing_")
    if not columns:
        raise ValueError("null_indicator step requires 'columns' parameter")
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column not found: {col}")
        result[f"{prefix}{col}"] = result[col].isna().astype(int)
    return result
