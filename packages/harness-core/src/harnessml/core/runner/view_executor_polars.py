"""Polars-based view step executor.

Each step operates on a pl.LazyFrame and returns a pl.LazyFrame.
No .copy() needed -- Polars lazy frames are immutable query plans.
"""
from __future__ import annotations

import polars as pl


def execute_step(
    lf: pl.LazyFrame,
    step: dict,
    context: dict[str, pl.LazyFrame] | None = None,
) -> pl.LazyFrame:
    """Dispatch a single view step."""
    op = step["op"]
    executor = _DISPATCH.get(op)
    if executor is None:
        raise ValueError(f"Unknown view step: {op}")
    return executor(lf, step, context)


def execute_steps(
    lf: pl.LazyFrame,
    steps: list[dict],
    context: dict[str, pl.LazyFrame] | None = None,
) -> pl.LazyFrame:
    """Execute a sequence of view steps."""
    for step in steps:
        lf = execute_step(lf, step, context)
    return lf


# ---------------------------------------------------------------------------
# Core steps: filter, select, derive, sort, distinct, head
# ---------------------------------------------------------------------------


def _filter(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    return lf.filter(pl.sql_expr(step["expr"]))


def _select(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    columns = step["columns"]
    if isinstance(columns, list):
        return lf.select(columns)
    # dict = rename: {new_name: old_name}
    return lf.select([pl.col(old).alias(new) for new, old in columns.items()])


def _derive(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    exprs = []
    for name, expr_str in step["columns"].items():
        exprs.append(pl.sql_expr(expr_str).alias(name))
    return lf.with_columns(exprs)


def _sort(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    by = step["by"] if isinstance(step["by"], list) else [step["by"]]
    ascending = step.get("ascending", True)
    if isinstance(ascending, bool):
        descending = [not ascending] * len(by)
    else:
        descending = [not a for a in ascending]
    return lf.sort(by, descending=descending)


def _distinct(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    columns = step.get("columns")
    keep = step.get("keep", "first")
    if columns:
        return lf.unique(subset=columns, keep=keep)
    return lf.unique(keep=keep)


def _head(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    keys = step["keys"]
    n = step.get("n", 1)
    order_by = step.get("order_by")
    ascending = step.get("ascending", True)
    position = step.get("position", "first")

    if order_by:
        if isinstance(order_by, str):
            order_by = [order_by]
        if isinstance(ascending, bool):
            descending = [not ascending] * len(order_by)
        else:
            descending = [not a for a in ascending]
        if position == "last":
            descending = [not d for d in descending]
        lf = lf.sort(order_by, descending=descending)

    return lf.group_by(keys).head(n)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, callable] = {
    "filter": _filter,
    "select": _select,
    "derive": _derive,
    "sort": _sort,
    "distinct": _distinct,
    "head": _head,
}
