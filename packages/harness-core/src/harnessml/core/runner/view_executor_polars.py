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
# Aggregation steps: group_by, rolling, cond_agg, ewm, rank
# ---------------------------------------------------------------------------

_AGG_MAP = {
    "mean": lambda c: c.mean(),
    "sum": lambda c: c.sum(),
    "min": lambda c: c.min(),
    "max": lambda c: c.max(),
    "count": lambda c: c.count(),
    "std": lambda c: c.std(),
    "var": lambda c: c.var(),
    "median": lambda c: c.median(),
    "first": lambda c: c.first(),
    "last": lambda c: c.last(),
}


def _group_by(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    keys = step["keys"]
    agg_exprs = []
    for col, func in step["aggs"].items():
        if isinstance(func, list):
            for f in func:
                agg_fn = _AGG_MAP.get(f)
                if agg_fn is None:
                    raise ValueError(f"Unknown aggregation: {f}")
                agg_exprs.append(agg_fn(pl.col(col)).alias(f"{col}_{f}"))
        else:
            agg_fn = _AGG_MAP.get(func)
            if agg_fn is None:
                raise ValueError(f"Unknown aggregation: {func}")
            agg_exprs.append(agg_fn(pl.col(col)).alias(col))
    return lf.group_by(keys).agg(agg_exprs)


def _rolling(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Rolling window aggregation partitioned by keys."""
    keys = step["keys"]
    order_by = step["order_by"]
    window = step["window"]
    lf = lf.sort([*keys, order_by])

    for new_col, spec in step["aggs"].items():
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Rolling agg spec must be 'column:func', got {spec!r}")
        src_col, func = parts[0].strip(), parts[1].strip()

        agg_fn = _AGG_MAP.get(func)
        if agg_fn is None:
            raise ValueError(f"Unknown rolling aggregation: {func}")

        expr = agg_fn(
            pl.col(src_col).rolling(index_column=order_by, period=f"{window}i")
        ).over(keys).alias(new_col)
        lf = lf.with_columns(expr)

    return lf


def _cond_agg(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Group and aggregate with optional per-agg conditions."""
    keys = step["keys"]
    agg_exprs = []

    for new_col, spec in step["aggs"].items():
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise ValueError(
                f"cond_agg spec must be 'column:func' or 'column:func:condition', "
                f"got {spec!r}"
            )
        src_col, func = parts[0].strip(), parts[1].strip()
        condition = parts[2].strip() if len(parts) == 3 else None

        agg_fn = _AGG_MAP.get(func)
        if agg_fn is None:
            raise ValueError(f"Unknown aggregation: {func}")

        if condition:
            expr = agg_fn(
                pl.col(src_col).filter(pl.sql_expr(condition))
            ).alias(new_col)
        else:
            expr = agg_fn(pl.col(src_col)).alias(new_col)
        agg_exprs.append(expr)

    return lf.group_by(keys).agg(agg_exprs)


def _ewm(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Exponentially weighted moving statistic within groups."""
    keys = step["keys"]
    order_by = step["order_by"]
    span = step["span"]

    lf = lf.sort([*keys, order_by])

    for new_col, spec in step["aggs"].items():
        src_col, stat = spec.rsplit(":", 1)
        src_col, stat = src_col.strip(), stat.strip()

        if stat == "mean":
            expr = (
                pl.col(src_col)
                .ewm_mean(span=span, adjust=False)
                .over(keys)
                .alias(new_col)
            )
        elif stat == "std":
            expr = (
                pl.col(src_col)
                .ewm_std(span=span, adjust=False)
                .over(keys)
                .alias(new_col)
            )
        elif stat == "var":
            expr = (
                pl.col(src_col)
                .ewm_var(span=span, adjust=False)
                .over(keys)
                .alias(new_col)
            )
        else:
            raise ValueError(f"Unsupported EWM stat: {stat}")

        lf = lf.with_columns(expr)

    return lf


def _rank(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Add rank columns, optionally within groups."""
    ascending = step.get("ascending", True)
    keys = step.get("keys")
    method = step.get("method", "average")
    pct = step.get("pct", False)
    descending = not ascending

    for new_col, src_col in step["columns"].items():
        if keys:
            rank_expr = pl.col(src_col).rank(method=method, descending=descending).over(keys)
        else:
            rank_expr = pl.col(src_col).rank(method=method, descending=descending)

        if pct:
            if keys:
                count_expr = pl.col(src_col).count().over(keys)
            else:
                count_expr = pl.col(src_col).count()
            rank_expr = rank_expr / count_expr

        lf = lf.with_columns(rank_expr.alias(new_col))

    return lf


# ---------------------------------------------------------------------------
# Join + Union steps
# ---------------------------------------------------------------------------


def _join(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Join with another table resolved from context."""
    if ctx is None:
        raise ValueError("A context is required for join steps")
    other_name = step["other"]
    other_lf = ctx.get(other_name)
    if other_lf is None:
        raise ValueError(f"Table '{other_name}' not found in context")

    on = step["on"]
    how = step.get("how", "left")
    select_cols = step.get("select")
    prefix = step.get("prefix")

    if isinstance(on, dict):
        left_on = list(on.keys())
        right_on = list(on.values())
    else:
        left_on = list(on) if isinstance(on, list) else [on]
        right_on = left_on

    # Apply select to narrow the right side before join
    if select_cols:
        keep_cols = [*right_on, *select_cols]
        other_lf = other_lf.select([c for c in keep_cols])

    # Apply prefix to non-key columns
    if prefix:
        schema = other_lf.collect_schema()
        rename_map = {
            c: f"{prefix}{c}" for c in schema.names() if c not in right_on
        }
        if rename_map:
            other_lf = other_lf.rename(rename_map)

    result = lf.join(other_lf, left_on=left_on, right_on=right_on, how=how)
    return result


def _union(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Concatenate with another table from context."""
    if ctx is None:
        raise ValueError("A context is required for union steps")
    other_name = step["other"]
    other_lf = ctx.get(other_name)
    if other_lf is None:
        raise ValueError(f"Table '{other_name}' not found in context")
    return pl.concat([lf, other_lf])


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
    "group_by": _group_by,
    "rolling": _rolling,
    "cond_agg": _cond_agg,
    "ewm": _ewm,
    "rank": _rank,
    "join": _join,
    "union": _union,
}
