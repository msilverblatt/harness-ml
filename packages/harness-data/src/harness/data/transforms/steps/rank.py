"""Rank step — compute rank of column values."""
from __future__ import annotations
import pandas as pd

NAME = "rank"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    columns = params.get("columns")
    if not columns or not isinstance(columns, dict):
        raise ValueError("rank step requires 'columns' dict of {new_col: source_col}")
    keys = params.get("keys")
    method = params.get("method", "average")
    ascending = params.get("ascending", True)
    pct = params.get("pct", False)
    result = df.copy()
    for new_col, source_col in columns.items():
        if source_col not in result.columns:
            raise ValueError(f"Column not found: {source_col}")
        if keys:
            result[new_col] = result.groupby(keys)[source_col].rank(
                method=method, ascending=ascending, pct=pct
            )
        else:
            result[new_col] = result[source_col].rank(
                method=method, ascending=ascending, pct=pct
            )
    return result
