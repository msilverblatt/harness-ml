"""Union step — vertical concat of two DataFrames."""
from __future__ import annotations

import pandas as pd

NAME = "union"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    other_name = params.get("other")
    resolver = params.get("_resolver")
    if not other_name:
        raise ValueError("union step requires 'other' parameter")
    if resolver is None:
        raise ValueError("union step requires a resolver")
    other_df = resolver(other_name)
    return pd.concat([df, other_df], ignore_index=True)
