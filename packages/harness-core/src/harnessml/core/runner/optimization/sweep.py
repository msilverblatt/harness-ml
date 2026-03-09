"""Sweep expansion for experiment overlays.

Expands a single overlay with a ``sweep`` key into multiple concrete overlays,
one per parameter combination (cartesian product across axes).
"""
from __future__ import annotations

import copy
import itertools
from typing import Any


def set_nested_key(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated key path.

    Creates intermediate dicts as needed.

    Example::

        d = {}
        set_nested_key(d, "a.b.c", 42)
        # d == {"a": {"b": {"c": 42}}}
    """
    keys = dotted_key.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def get_nested_key(d: dict, dotted_key: str, default: Any = None) -> Any:
    """Read a value from a nested dict using a dot-separated key path."""
    keys = dotted_key.split(".")
    current = d
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def expand_sweep(overlay: dict) -> list[dict]:
    """Expand an overlay with ``sweep`` config into concrete overlay variants.

    Parameters
    ----------
    overlay:
        Experiment overlay dict, optionally containing a ``sweep`` key.

    Returns
    -------
    list[dict]
        One overlay per combination.  If no ``sweep`` key, returns ``[overlay]``.
        Each variant has the sweep key removed and the swept values injected.

    The ``sweep`` value may be:

    * A single axis dict: ``{"key": "models.xgb.params.lr", "values": [0.01, 0.05]}``
    * A list of axis dicts for multi-axis (cartesian product) sweeps.

    Each axis dict must have:

    * ``key`` — dot-separated path into the overlay
    * ``values`` — list of values to sweep over

    A ``description`` field on each variant is auto-generated from the swept
    values appended to the base description.
    """
    if "sweep" not in overlay:
        return [overlay]

    sweep_spec = overlay["sweep"]
    base = {k: v for k, v in overlay.items() if k != "sweep"}
    base_desc = base.get("description", "sweep")

    # Normalize to list of axes
    if isinstance(sweep_spec, dict):
        axes = [sweep_spec]
    elif isinstance(sweep_spec, list):
        axes = sweep_spec
    else:
        raise ValueError(
            f"sweep must be a dict or list of dicts, got {type(sweep_spec).__name__}"
        )

    for i, ax in enumerate(axes):
        if "key" not in ax or "values" not in ax:
            raise ValueError(
                f"Sweep axis {i} must have 'key' and 'values', got {sorted(ax.keys())}"
            )
        if not isinstance(ax["values"], list) or len(ax["values"]) == 0:
            raise ValueError(
                f"Sweep axis {i} 'values' must be a non-empty list"
            )

    axis_keys = [ax["key"] for ax in axes]
    axis_values = [ax["values"] for ax in axes]

    variants: list[dict] = []
    for combo in itertools.product(*axis_values):
        variant = copy.deepcopy(base)
        parts: list[str] = []
        for key, val in zip(axis_keys, combo):
            set_nested_key(variant, key, val)
            short_key = key.split(".")[-1]
            parts.append(f"{short_key}={val}")
        variant["description"] = f"{base_desc} ({', '.join(parts)})"
        variants.append(variant)

    return variants
