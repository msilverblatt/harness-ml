"""Head step — take the first N rows, optionally per group."""
from __future__ import annotations
import pandas as pd

NAME = "head"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    n = params.get("n", 10)
    order_by = params.get("order_by")
    ascending = params.get("ascending", True)
    keys = params.get("keys")
    result = df
    if order_by:
        if isinstance(order_by, str):
            order_by = [order_by]
        result = result.sort_values(by=order_by, ascending=ascending)
    if keys:
        return result.groupby(keys, sort=False).head(n).reset_index(drop=True)
    return result.head(n).reset_index(drop=True)
