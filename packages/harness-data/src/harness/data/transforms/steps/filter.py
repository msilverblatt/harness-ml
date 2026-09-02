"""Filter step — row filtering via pandas query."""
from __future__ import annotations

import pandas as pd

NAME = "filter"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Filter rows using a pandas query expression."""
    expr = params.get("expr")
    if not expr:
        raise ValueError("filter step requires 'expr' parameter")
    return df.query(expr).reset_index(drop=True)
