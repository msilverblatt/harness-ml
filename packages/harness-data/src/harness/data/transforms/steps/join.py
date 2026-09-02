"""Join step — merge with another DataFrame via resolver callback."""
from __future__ import annotations

import pandas as pd

NAME = "join"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    other_name = params.get("other")
    on = params.get("on")
    how = params.get("how", "left")
    select = params.get("select")
    prefix = params.get("prefix")
    resolver = params.get("_resolver")
    if not other_name or not on:
        raise ValueError("join step requires 'other' and 'on' parameters")
    if resolver is None:
        raise ValueError("join step requires a resolver to load the other source/view")
    other_df = resolver(other_name)
    if select:
        keep_cols = list(set(select + (on if isinstance(on, list) else list(on.keys()))))
        other_df = other_df[[c for c in keep_cols if c in other_df.columns]]
    if isinstance(on, dict):
        result = df.merge(other_df, left_on=list(on.keys()), right_on=list(on.values()), how=how)
    else:
        result = df.merge(other_df, on=on, how=how)
    if prefix:
        if isinstance(on, dict):
            on_cols = list(on.keys())
        elif isinstance(on, list):
            on_cols = on
        else:
            on_cols = [on]
        new_cols = [c for c in result.columns if c not in df.columns and c not in on_cols]
        result = result.rename(columns={c: f"{prefix}{c}" for c in new_cols})
    return result
