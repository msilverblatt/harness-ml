"""Unpivot step — melt wide to long."""
from __future__ import annotations

import pandas as pd

NAME = "unpivot"


def step(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    unpivot_columns = params.get("unpivot_columns", {})
    if not unpivot_columns:
        raise ValueError("unpivot step requires 'unpivot_columns' dict")
    result = df.copy()
    for value_name, cols in unpivot_columns.items():
        result = pd.melt(
            result,
            id_vars=[c for c in result.columns if c not in cols],
            value_vars=cols,
            var_name=f"{value_name}_source",
            value_name=value_name,
        )
    return result.reset_index(drop=True)
