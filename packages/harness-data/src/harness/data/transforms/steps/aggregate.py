"""Aggregate step — group by + agg."""
from __future__ import annotations

import pandas as pd

NAME = "aggregate"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    keys = params.get("keys")
    aggs = params.get("aggs")
    if not keys or not aggs:
        raise ValueError("aggregate step requires 'keys' and 'aggs' parameters")
    agg_dict = {}
    for col, funcs in aggs.items():
        if isinstance(funcs, str):
            funcs = [funcs]
        agg_dict[col] = funcs
    result = df.groupby(keys, as_index=False).agg(agg_dict)
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [f"{col}_{func}" if func else col for col, func in result.columns]
    return result.reset_index(drop=True)
