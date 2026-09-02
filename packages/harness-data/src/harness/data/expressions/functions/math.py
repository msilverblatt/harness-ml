"""Math functions for the expression engine."""
from __future__ import annotations

import numpy as np

FUNCTIONS = {}


def _register(name: str, description: str):
    """Decorator to register a function."""
    def decorator(fn):
        FUNCTIONS[name] = {"fn": fn, "description": description}
        return fn
    return decorator


@_register("abs", "Absolute value")
def fn_abs(x):
    return np.abs(x)


@_register("log", "Natural logarithm (uses log1p for safety)")
def fn_log(x):
    return np.log1p(x)


@_register("sqrt", "Square root")
def fn_sqrt(x):
    return np.sqrt(x)


@_register("exp", "Exponential (e^x)")
def fn_exp(x):
    return np.exp(x)


@_register("clip", "Clip values to [lower, upper]")
def fn_clip(x, lower, upper):
    return np.clip(x, lower, upper)


@_register("sign", "Sign of values (-1, 0, or 1)")
def fn_sign(x):
    return np.sign(x)


@_register("floor", "Floor (round down)")
def fn_floor(x):
    return np.floor(x)


@_register("ceil", "Ceiling (round up)")
def fn_ceil(x):
    return np.ceil(x)


@_register("round", "Round to given decimals")
def fn_round(x, decimals=0):
    return np.round(x, int(decimals))
