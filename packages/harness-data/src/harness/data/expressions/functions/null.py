"""Null-handling functions for the expression engine."""
from __future__ import annotations

import pandas as pd

FUNCTIONS = {}


def _register(name: str, description: str):
    """Decorator to register a function."""
    def decorator(fn):
        FUNCTIONS[name] = {"fn": fn, "description": description}
        return fn
    return decorator


@_register("isna", "Check if values are null/NaN")
def fn_isna(x):
    return pd.Series(x).isna()


@_register("fillna", "Fill null/NaN values with a replacement")
def fn_fillna(x, fill_value=0):
    return pd.Series(x).fillna(fill_value)


@_register("coalesce", "Return first non-null value from arguments")
def fn_coalesce(*args):
    result = pd.Series(args[0])
    for other in args[1:]:
        result = result.fillna(pd.Series(other))
    return result
