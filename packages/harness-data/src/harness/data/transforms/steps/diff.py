"""Diff step — compute differences or percent changes between rows."""
from __future__ import annotations

import pandas as pd

NAME = "diff"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    columns = params.get("columns", {})
    pct = params.get("pct", False)
    if not columns:
        raise ValueError("diff step requires 'columns' dict")
    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])
    for new_col, spec in columns.items():
        parts = spec.split(":")
        source_col = parts[0]
        periods = int(parts[1]) if len(parts) > 1 else 1
        if pct:
            if keys:
                result[new_col] = result.groupby(keys, sort=False)[source_col].pct_change(periods)
            else:
                result[new_col] = result[source_col].pct_change(periods)
        else:
            if keys:
                result[new_col] = result.groupby(keys, sort=False)[source_col].diff(periods)
            else:
                result[new_col] = result[source_col].diff(periods)
    return result.sort_index().reset_index(drop=True)
