"""Datetime step — extract calendar features."""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "datetime"

EXTRACTORS = {
    "year": lambda s: s.dt.year,
    "month": lambda s: s.dt.month,
    "day": lambda s: s.dt.day,
    "dayofweek": lambda s: s.dt.dayofweek,
    "hour": lambda s: s.dt.hour,
    "quarter": lambda s: s.dt.quarter,
    "weekofyear": lambda s: s.dt.isocalendar().week.astype(int),
}


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    column = params.get("column")
    extract = params.get("extract", [])
    cyclical = params.get("cyclical", [])
    if not column:
        raise ValueError("datetime step requires 'column' parameter")
    result = df.copy()
    col_dt = pd.to_datetime(result[column])
    for component in extract:
        extractor = EXTRACTORS.get(component)
        if extractor is None:
            raise ValueError(f"Unknown datetime component: {component}")
        result[f"{column}_{component}"] = extractor(col_dt)
    for component in cyclical:
        extractor = EXTRACTORS.get(component)
        if extractor is None:
            raise ValueError(f"Unknown datetime component: {component}")
        values = extractor(col_dt).astype(float)
        max_val = {"month": 12, "dayofweek": 7, "hour": 24, "quarter": 4}.get(component, values.max())
        result[f"{column}_{component}_sin"] = np.sin(2 * np.pi * values / max_val)
        result[f"{column}_{component}_cos"] = np.cos(2 * np.pi * values / max_val)
    return result
