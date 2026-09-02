"""Statistics functions for the expression engine."""
from __future__ import annotations

import pandas as pd


FUNCTIONS = {}


def _register(name: str, description: str):
    """Decorator to register a function."""
    def decorator(fn):
        FUNCTIONS[name] = {"fn": fn, "description": description}
        return fn
    return decorator


@_register("zscore", "Z-score normalization: (x - mean) / std")
def fn_zscore(x):
    s = pd.Series(x)
    mean = s.mean()
    std = s.std()
    if std == 0:
        return s * 0.0
    return (s - mean) / std


@_register("rank_pct", "Percentile rank (0-1)")
def fn_rank_pct(x):
    s = pd.Series(x)
    return s.rank(pct=True)
