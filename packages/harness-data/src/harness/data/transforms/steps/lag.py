"""Lag step — shift column values by N periods."""
from __future__ import annotations

import pandas as pd

NAME = "lag"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    columns = params.get("columns", {})
    if not columns:
        raise ValueError("lag step requires 'columns' dict of {new_col: 'source_col:periods'}")
    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])
    for new_col, spec in columns.items():
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Lag spec must be 'col:periods', got: {spec}")
        source_col, periods = parts[0], int(parts[1])
        if keys:
            result[new_col] = result.groupby(keys, sort=False)[source_col].shift(periods)
        else:
            result[new_col] = result[source_col].shift(periods)
    return result.sort_index().reset_index(drop=True)
