"""Comparison functions for the expression engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

FUNCTIONS = {}


def _register(name: str, description: str):
    """Decorator to register a function."""
    def decorator(fn):
        FUNCTIONS[name] = {"fn": fn, "description": description}
        return fn
    return decorator


@_register("where", "Conditional: where(condition, true_val, false_val)")
def fn_where(condition, true_val, false_val):
    return pd.Series(np.where(condition, true_val, false_val))


@_register("safe_div", "Safe division: returns 0 when denominator is 0")
def fn_safe_div(numerator, denominator):
    num = pd.Series(numerator, dtype=float)
    den = pd.Series(denominator, dtype=float)
    result = num / den
    result = result.fillna(0.0)
    result = result.replace([np.inf, -np.inf], 0.0)
    return result


@_register("minimum", "Element-wise minimum of two series")
def fn_minimum(a, b):
    return np.minimum(a, b)


@_register("maximum", "Element-wise maximum of two series")
def fn_maximum(a, b):
    return np.maximum(a, b)
