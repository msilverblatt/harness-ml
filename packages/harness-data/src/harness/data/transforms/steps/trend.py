"""Trend step — OLS slope over a rolling window."""
from __future__ import annotations
import numpy as np
import pandas as pd

NAME = "trend"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    window = params.get("window")
    columns = params.get("columns", {})
    if not window or not columns:
        raise ValueError("trend step requires 'window' and 'columns' parameters")
    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])

    def slope(arr):
        y = arr.values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        x = np.arange(len(y), dtype=float)[mask]
        y = y[mask]
        return np.polyfit(x, y, 1)[0]

    for new_col, source_col in columns.items():
        if keys:
            result[new_col] = result.groupby(keys, sort=False)[source_col].transform(
                lambda s: s.rolling(window, min_periods=2).apply(slope, raw=False)
            )
        else:
            result[new_col] = result[source_col].rolling(window, min_periods=2).apply(slope, raw=False)
    return result.sort_index().reset_index(drop=True)
