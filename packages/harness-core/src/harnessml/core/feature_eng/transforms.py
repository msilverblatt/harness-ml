"""Feature transforms including cyclical encoding."""
from __future__ import annotations

import numpy as np


def cyclical_encode(values, period):
    """Encode cyclical features as sin/cos pairs.

    Parameters
    ----------
    values : array-like of numeric values
    period : the period of the cycle (e.g., 12 for months, 24 for hours)

    Returns
    -------
    tuple of (sin_values, cos_values) as numpy arrays
    """
    values = np.asarray(values, dtype=float)
    sin_vals = np.sin(2 * np.pi * values / period)
    cos_vals = np.cos(2 * np.pi * values / period)
    return sin_vals, cos_vals
