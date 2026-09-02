"""Bin step — discretize a numeric column."""
from __future__ import annotations

import pandas as pd

NAME = "bin"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = params.get("column")
    method = params.get("method", "quantile")
    n_bins = params.get("n_bins", 5)
    output = params.get("output", f"{column}_binned")
    boundaries = params.get("boundaries")
    if not column:
        raise ValueError("bin step requires 'column' parameter")
    result = df.copy()
    if method == "quantile":
        result[output] = pd.qcut(result[column], q=n_bins, labels=False, duplicates="drop")
    elif method == "uniform":
        result[output] = pd.cut(result[column], bins=n_bins, labels=False)
    elif method == "custom" and boundaries:
        result[output] = pd.cut(result[column], bins=boundaries, labels=False)
    else:
        raise ValueError(f"Unknown bin method: {method}")
    return result
