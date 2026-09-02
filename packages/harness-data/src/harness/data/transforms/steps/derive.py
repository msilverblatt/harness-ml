"""Derive step — create new columns from expressions."""
from __future__ import annotations

from typing import Any

import pandas as pd
from harness.data.expressions.engine import ExpressionEngine

NAME = "derive"
_engine = ExpressionEngine()


def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns")
    if not columns or not isinstance(columns, dict):
        raise ValueError("derive step requires 'columns' dict of {name: expression}")
    result = df.copy()
    for col_name, expr in columns.items():
        result[col_name] = _engine.evaluate(result, expr)
    return result
