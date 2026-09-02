"""Rolling step — rolling window aggregations."""
from __future__ import annotations
import numpy as np
import pandas as pd

NAME = "rolling"
BUILTIN_AGGS = {"mean", "std", "sum", "min", "max", "count", "median"}


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    window = params.get("window")
    aggs = params.get("aggs", {})
    min_periods = params.get("min_periods", 1)
    if not window or not aggs:
        raise ValueError("rolling step requires 'window' and 'aggs' parameters")
    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])
    for new_col, spec in aggs.items():
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Rolling agg spec must be 'col:func', got: {spec}")
        source_col, func = parts
        if keys:
            grouped = result.groupby(keys, sort=False)[source_col]
            rolling_obj = grouped.rolling(window=window, min_periods=min_periods)
        else:
            rolling_obj = result[source_col].rolling(window=window, min_periods=min_periods)
        if func in BUILTIN_AGGS:
            values = getattr(rolling_obj, func)()
        else:
            raise ValueError(f"Unknown rolling function: {func}")
        if keys:
            values = values.reset_index(level=list(range(len(keys))), drop=True)
        result[new_col] = values
    return result.sort_index().reset_index(drop=True)
