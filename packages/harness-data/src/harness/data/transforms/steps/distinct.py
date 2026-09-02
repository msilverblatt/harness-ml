"""Distinct step — drop duplicate rows."""
from __future__ import annotations
import pandas as pd

NAME = "distinct"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    columns = params.get("columns")
    keep = params.get("keep", "first")
    subset = columns if columns else None
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
