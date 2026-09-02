"""Sort step — sort rows by columns."""
from __future__ import annotations
import pandas as pd

NAME = "sort"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    by = params.get("by")
    if not by:
        raise ValueError("sort step requires 'by' parameter")
    ascending = params.get("ascending", True)
    if isinstance(by, str):
        by = [by]
    return df.sort_values(by=by, ascending=ascending).reset_index(drop=True)
