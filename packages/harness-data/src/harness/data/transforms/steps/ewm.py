"""EWM step — exponentially weighted moving statistics."""
from __future__ import annotations

import pandas as pd

NAME = "ewm"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    span = params.get("span")
    aggs = params.get("aggs", {})
    if not span or not aggs:
        raise ValueError("ewm step requires 'span' and 'aggs' parameters")
    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])
    for new_col, spec in aggs.items():
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"EWM spec must be 'col:stat', got: {spec}")
        source_col, stat = parts
        if keys:
            ewm_obj = result.groupby(keys, sort=False)[source_col].transform(
                lambda s: getattr(s.ewm(span=span), stat)()
            )
        else:
            ewm_obj = getattr(result[source_col].ewm(span=span), stat)()
        result[new_col] = ewm_obj
    return result.sort_index().reset_index(drop=True)
