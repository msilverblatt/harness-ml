"""Fill step — fill missing values."""
from __future__ import annotations

import pandas as pd

NAME = "fill"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    result = df.copy()
    columns = params.get("columns")  # {col: value}
    strategy = params.get("strategy")  # "median", "mean", "zero", "mode", "ffill"
    if columns and isinstance(columns, dict):
        for col, value in columns.items():
            if col in result.columns:
                result[col] = result[col].fillna(value)
    elif strategy:
        numeric_cols = result.select_dtypes(include="number").columns
        if strategy == "median":
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
        elif strategy == "mean":
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
        elif strategy == "zero":
            result[numeric_cols] = result[numeric_cols].fillna(0)
        elif strategy == "mode":
            for col in result.columns:
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val.iloc[0])
        elif strategy == "ffill":
            result = result.ffill()
        else:
            raise ValueError(f"Unknown fill strategy: {strategy}")
    return result
