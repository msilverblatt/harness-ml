"""Isin step — filter rows where column value is in a set."""
from __future__ import annotations

import pandas as pd

NAME = "isin"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = params.get("column")
    values = params.get("values")
    negate = params.get("negate", False)
    if not column or values is None:
        raise ValueError("isin step requires 'column' and 'values' parameters")
    mask = df[column].isin(values)
    if negate:
        mask = ~mask
    return df[mask].reset_index(drop=True)
