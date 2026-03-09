"""Polars-based view step executor.

Each step operates on a pl.LazyFrame and returns a pl.LazyFrame.
No .copy() needed -- Polars lazy frames are immutable query plans.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
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
        return lf.unique(subset=columns, keep=keep, maintain_order=True)
    return lf.unique(keep=keep, maintain_order=True)


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

    return lf.group_by(keys, maintain_order=True).head(n)


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
            # Always use {col}_{func} naming to match pandas behavior
            agg_exprs.append(agg_fn(pl.col(col)).alias(f"{col}_{func}"))
    return lf.group_by(keys, maintain_order=True).agg(agg_exprs)


_ROLLING_EXPR_MAP = {
    "mean": lambda c, w, mp: c.rolling_mean(window_size=w, min_samples=mp),
    "std": lambda c, w, mp: c.rolling_std(window_size=w, min_samples=mp),
    "sum": lambda c, w, mp: c.rolling_sum(window_size=w, min_samples=mp),
    "min": lambda c, w, mp: c.rolling_min(window_size=w, min_samples=mp),
    "max": lambda c, w, mp: c.rolling_max(window_size=w, min_samples=mp),
    "var": lambda c, w, mp: c.rolling_var(window_size=w, min_samples=mp),
    "median": lambda c, w, mp: c.rolling_median(window_size=w, min_samples=mp),
}


def _rolling(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Rolling window aggregation partitioned by keys."""
    keys = step["keys"]
    order_by = step["order_by"]
    window = step["window"]
    min_periods = step.get("min_periods") or window
    lf = lf.sort([*keys, order_by])

    for new_col, spec in step["aggs"].items():
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Rolling agg spec must be 'column:func', got {spec!r}")
        src_col, func = parts[0].strip(), parts[1].strip()

        builder = _ROLLING_EXPR_MAP.get(func)
        if builder is None:
            raise ValueError(
                f"Polars rolling does not support '{func}' -- falling back to pandas"
            )

        expr = builder(pl.col(src_col), window, min_periods).over(keys).alias(new_col)
        lf = lf.with_columns(expr)

    return lf


def _cond_agg(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Group and aggregate with optional per-agg conditions."""
    keys = step["keys"]
    agg_exprs = []
    conditional_cols = []

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
            conditional_cols.append(new_col)
        else:
            expr = agg_fn(pl.col(src_col)).alias(new_col)
        agg_exprs.append(expr)

    result = lf.group_by(keys, maintain_order=True).agg(agg_exprs)

    # Drop groups where ALL conditional columns are null (matches pandas behavior
    # where groups not matching a query condition are absent from the result)
    if conditional_cols:
        any_non_null = pl.lit(False)
        for col_name in conditional_cols:
            any_non_null = any_non_null | pl.col(col_name).is_not_null()
        result = result.filter(any_non_null)

    return result


def _ewm(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Exponentially weighted moving statistic within groups.

    Only supports mean -- std/var have subtle behavioral differences vs pandas
    (e.g. first-row NaN handling), so those fall back to the pandas executor.
    """
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
        else:
            raise ValueError(
                f"Polars EWM does not support '{stat}' -- falling back to pandas"
            )

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
# Remaining steps: lag, diff, trend, encode, bin, datetime, cast, unpivot,
#                  isin, null_indicator
# ---------------------------------------------------------------------------

# Type-name to Polars type mapping for CastStep
_PL_TYPE_MAP: dict[str, pl.DataType] = {
    "int": pl.Int64,
    "float": pl.Float64,
    "str": pl.Utf8,
    "bool": pl.Boolean,
}


def _lag(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Shift column values within groups (lag/lead)."""
    keys = step["keys"]
    order_by = step["order_by"]
    lf = lf.sort([*keys, order_by])

    for new_col, spec in step["columns"].items():
        src_col, lag_str = spec.rsplit(":", 1)
        lag_n = int(lag_str)
        lf = lf.with_columns(
            pl.col(src_col).shift(lag_n).over(keys).alias(new_col)
        )
    return lf


def _diff(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """First/second differences or percent change within groups."""
    keys = step["keys"]
    order_by = step["order_by"]
    pct = step.get("pct", False)
    lf = lf.sort([*keys, order_by])

    for new_col, spec in step["columns"].items():
        src_col, periods_str = spec.rsplit(":", 1)
        periods = int(periods_str)
        if pct:
            lf = lf.with_columns(
                pl.col(src_col).pct_change(n=periods).over(keys).alias(new_col)
            )
        else:
            lf = lf.with_columns(
                pl.col(src_col).diff(n=periods).over(keys).alias(new_col)
            )
    return lf


def _trend(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """OLS slope over a rolling window within groups.

    Uses Polars map_batches with numpy for the OLS computation within each
    rolling window, since vectorized rolling slope requires window-local indices.
    """
    keys = step["keys"]
    order_by = step["order_by"]
    window = step["window"]
    lf = lf.sort([*keys, order_by])

    # Precompute x values for the window (constant)
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_denom = ((x - x_mean) ** 2).sum()

    for new_col, src_col in step["columns"].items():
        # Use rolling_map via struct + map_batches
        expr = (
            pl.col(src_col)
            .rolling_map(
                lambda s: _ols_slope(s.to_numpy(), x, x_mean, x_denom),
                window_size=window,
                min_samples=window,
            )
            .over(keys)
            .alias(new_col)
        )
        lf = lf.with_columns(expr)

    return lf


def _ols_slope(
    y: np.ndarray, x: np.ndarray, x_mean: float, x_denom: float
) -> float:
    """Compute OLS slope for a single rolling window."""
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    y_vals = y[mask]
    x_vals = x[: len(y_vals)]
    y_mean = y_vals.mean()
    x_m = x_vals.mean()
    denom = ((x_vals - x_m) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x_vals - x_m) * (y_vals - y_mean)).sum() / denom)


def _encode(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Categorical encoding (frequency or ordinal)."""
    col = step["column"]
    out = step.get("output") or f"{col}_encoded"
    method = step["method"]

    if method == "frequency":
        # Count per category / total count
        counts = lf.group_by(col).agg(pl.len().alias("__freq_count__"))
        total = lf.select(pl.len().alias("__total__"))
        counts = counts.with_columns(
            (pl.col("__freq_count__") / total.collect().item()).alias(out)
        ).drop("__freq_count__")
        lf = lf.join(counts, on=col, how="left")
    elif method == "ordinal":
        counts = lf.group_by(col).agg(pl.len().alias("__ord_count__"))
        counts = counts.sort("__ord_count__", descending=True)
        counts = counts.with_row_index("__ord_rank__")
        counts = counts.with_columns(
            (pl.col("__ord_rank__") + 1).alias(out)
        ).select([col, out])
        lf = lf.join(counts, on=col, how="left")
    else:
        raise ValueError(f"Polars executor does not support encode method: {method!r}")

    return lf


def _bin(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Discretize a continuous column into bins."""
    col = step["column"]
    out = step.get("output") or f"{col}_binned"
    method = step["method"]
    n_bins = step["n_bins"]

    if method == "quantile":
        # Collect to compute quantile boundaries, then apply
        collected = lf.collect()
        result = collected.to_pandas()

        result[out] = pd.qcut(result[col], n_bins, labels=False, duplicates="drop")
        return pl.from_pandas(result).lazy()
    elif method == "uniform":
        collected = lf.collect()
        result = collected.to_pandas()

        result[out] = pd.cut(result[col], n_bins, labels=False)
        return pl.from_pandas(result).lazy()
    elif method == "custom":
        boundaries = step.get("boundaries")
        if boundaries is None:
            raise ValueError("custom binning requires 'boundaries'")
        collected = lf.collect()
        result = collected.to_pandas()

        result[out] = pd.cut(result[col], bins=boundaries, labels=False)
        return pl.from_pandas(result).lazy()
    else:
        raise ValueError(f"Unknown bin method: {method!r}")


def _datetime(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Extract calendar features from a datetime column."""
    col = step["column"]
    extract = step.get("extract", [])
    cyclical = step.get("cyclical", [])

    # Ensure column is datetime type
    lf = lf.with_columns(pl.col(col).cast(pl.Datetime).alias(col))

    _dt_accessor_map = {
        "year": lambda c: c.dt.year(),
        "month": lambda c: c.dt.month(),
        "day": lambda c: c.dt.day(),
        "hour": lambda c: c.dt.hour(),
        "minute": lambda c: c.dt.minute(),
        "second": lambda c: c.dt.second(),
        "dayofweek": lambda c: c.dt.weekday(),
        "quarter": lambda c: c.dt.quarter(),
        "weekofyear": lambda c: c.dt.week(),
    }

    _cyclical_periods = {
        "month": 12,
        "dayofweek": 7,
        "hour": 24,
        "quarter": 4,
        "weekofyear": 53,
        "day": 31,
        "minute": 60,
        "second": 60,
    }

    for part in extract:
        accessor = _dt_accessor_map.get(part)
        if accessor is None:
            raise ValueError(f"Unknown datetime part: {part}")
        lf = lf.with_columns(accessor(pl.col(col)).alias(f"{col}_{part}"))

    for part in cyclical:
        period = _cyclical_periods.get(part)
        if period is None:
            raise ValueError(f"No known period for cyclical encoding of {part!r}")
        accessor = _dt_accessor_map.get(part)
        if accessor is None:
            raise ValueError(f"Unknown datetime part: {part}")
        raw = accessor(pl.col(col)).cast(pl.Float64)
        lf = lf.with_columns([
            (raw * (2 * np.pi / period)).sin().alias(f"{col}_{part}_sin"),
            (raw * (2 * np.pi / period)).cos().alias(f"{col}_{part}_cos"),
        ])

    return lf


def _cast(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Cast columns to different types."""
    for col, type_spec in step["columns"].items():
        if ":" in type_spec:
            # Not supported in Polars executor -- fall back to pandas
            raise ValueError(
                f"Complex cast expressions ('{type_spec}') not supported in Polars executor"
            )
        target = _PL_TYPE_MAP.get(type_spec)
        if target is None:
            raise ValueError(f"Unknown type: {type_spec}")
        lf = lf.with_columns(pl.col(col).cast(target).alias(col))
    return lf


def _unpivot(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Unpivot (melt) columns."""
    id_columns = step["id_columns"]
    unpivot_columns = step["unpivot_columns"]
    names_column = step.get("names_column")
    names_map = step.get("names_map", {})

    # Determine N variants from first entry
    lengths = [len(sources) for sources in unpivot_columns.values()]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"All unpivot_columns entries must have the same number of source columns, "
            f"got lengths: {lengths}"
        )
    n_variants = lengths[0]
    first_sources = next(iter(unpivot_columns.values()))

    parts = []
    for i in range(n_variants):
        select_exprs = [pl.col(c) for c in id_columns]
        for new_col, sources in unpivot_columns.items():
            select_exprs.append(pl.col(sources[i]).alias(new_col))
        if names_column is not None:
            raw_name = first_sources[i]
            mapped_name = names_map.get(raw_name, raw_name)
            select_exprs.append(pl.lit(mapped_name).alias(names_column))
        parts.append(lf.select(select_exprs))

    return pl.concat(parts)


def _isin(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Filter rows where column value is (or is not) in a list."""
    col = step["column"]
    values = step["values"]
    negate = step.get("negate", False)
    expr = pl.col(col).is_in(values)
    if negate:
        expr = ~expr
    return lf.filter(expr)


def _null_indicator(lf: pl.LazyFrame, step: dict, ctx: dict | None) -> pl.LazyFrame:
    """Create binary indicators for missing values."""
    columns = step["columns"]
    prefix = step.get("prefix", "missing_")
    exprs = []
    for col in columns:
        exprs.append(pl.col(col).is_null().cast(pl.Int8).alias(f"{prefix}{col}"))
    return lf.with_columns(exprs)


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
    "lag": _lag,
    "diff": _diff,
    "trend": _trend,
    "encode": _encode,
    "bin": _bin,
    "datetime": _datetime,
    "cast": _cast,
    "unpivot": _unpivot,
    "isin": _isin,
    "null_indicator": _null_indicator,
}
