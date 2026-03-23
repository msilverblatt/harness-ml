"""Conditional aggregate step — agg with per-agg conditions."""
from __future__ import annotations

import pandas as pd

NAME = "conditional_agg"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    keys = params.get("keys")
    aggs = params.get("aggs", {})
    if not keys or not aggs:
        raise ValueError("conditional_agg step requires 'keys' and 'aggs'")
    result_parts = []
    for new_col, spec in aggs.items():
        parts = spec.split(":")
        if len(parts) == 2:
            source_col, func = parts
            condition = None
        elif len(parts) == 3:
            source_col, func, condition = parts
        else:
            raise ValueError(f"Spec must be 'col:func' or 'col:func:condition', got: {spec}")
        subset = df
        if condition:
            subset = df.query(condition.strip())
        agg_result = subset.groupby(keys, as_index=False)[source_col].agg(func)
        agg_result = agg_result.rename(columns={source_col: new_col})
        result_parts.append(agg_result)
    result = result_parts[0]
    for part in result_parts[1:]:
        result = result.merge(part, on=keys, how="outer")
    return result.reset_index(drop=True)
