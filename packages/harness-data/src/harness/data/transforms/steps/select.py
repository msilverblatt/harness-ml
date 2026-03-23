"""Select step — column selection and renaming."""
from __future__ import annotations

import pandas as pd

NAME = "select"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Select and optionally rename columns."""
    columns = params.get("columns")
    if columns is None:
        raise ValueError("select step requires 'columns' parameter")
    if isinstance(columns, dict):
        df = df[list(columns.values())]
        df = df.rename(columns={v: k for k, v in columns.items()})
        return df
    elif isinstance(columns, list):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        return df[columns]
    else:
        raise TypeError(f"columns must be list or dict, got {type(columns)}")
