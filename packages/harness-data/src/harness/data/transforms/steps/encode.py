"""Encode step — categorical encoding."""
from __future__ import annotations

import pandas as pd

NAME = "encode"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = params.get("column")
    method = params.get("method", "frequency")
    output = params.get("output", f"{column}_encoded")
    if not column:
        raise ValueError("encode step requires 'column' parameter")
    result = df.copy()
    if method == "frequency":
        freq = result[column].value_counts(normalize=True)
        result[output] = result[column].map(freq)
    elif method == "ordinal":
        categories = sorted(result[column].dropna().unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        result[output] = result[column].map(mapping)
    else:
        raise ValueError(f"Unknown encode method: {method}")
    return result
