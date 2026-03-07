"""View step execution engine -- pure functions mapping TransformStep to pandas operations."""
from __future__ import annotations

import logging
import re
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from harnessml.core.runner.schema import (
        BinStep,
        CastStep,
        ConditionalAggStep,
        DatetimeStep,
        DeriveStep,
        DiffStep,
        DistinctStep,
        EncodeStep,
        EwmStep,
        FilterStep,
        GroupByStep,
        HeadStep,
        IsInStep,
        JoinStep,
        LagStep,
        NullIndicatorStep,
        RankStep,
        RollingStep,
        SelectStep,
        SortStep,
        TransformStep,
        TrendStep,
        UnionStep,
        UnpivotStep,
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type-name to Python type mapping for CastStep
# ---------------------------------------------------------------------------
_TYPE_MAP: dict[str, type] = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------


def execute_step(
    df: pd.DataFrame,
    step: TransformStep,
    resolver: Callable[[str], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Execute a single transform step on a DataFrame.

    Parameters
    ----------
    df : DataFrame to transform
    step : The step to execute (discriminated on step.op)
    resolver : Callable that resolves a source/view name to a DataFrame.
               Required for join and union steps.
    """
    _dispatch = {
        "filter": _execute_filter,
        "select": _execute_select,
        "derive": _execute_derive,
        "group_by": _execute_group_by,
        "join": _execute_join,
        "union": _execute_union,
        "unpivot": _execute_unpivot,
        "cast": _execute_cast,
        "sort": _execute_sort,
        "distinct": _execute_distinct,
        "rolling": _execute_rolling,
        "head": _execute_head,
        "rank": _execute_rank,
        "cond_agg": _execute_cond_agg,
        "isin": _execute_isin,
        "lag": _execute_lag,
        "ewm": _execute_ewm,
        "diff": _execute_diff,
        "trend": _execute_trend,
        "encode": _execute_encode,
        "bin": _execute_bin,
        "datetime": _execute_datetime,
        "null_indicator": _execute_null_indicator,
    }
    handler = _dispatch.get(step.op)
    if handler is None:
        raise ValueError(f"Unknown step op: {step.op!r}")

    if step.op in ("join", "union"):
        if resolver is None:
            raise ValueError(f"A resolver is required for {step.op!r} steps")
        return handler(df, step, resolver)

    return handler(df, step)


# ---------------------------------------------------------------------------
# Per-operation executors
# ---------------------------------------------------------------------------


def _execute_filter(df: pd.DataFrame, step: FilterStep) -> pd.DataFrame:
    return df.query(step.expr).reset_index(drop=True)


def _execute_select(df: pd.DataFrame, step: SelectStep) -> pd.DataFrame:
    if isinstance(step.columns, list):
        return df[step.columns]
    # dict form: {new_name: old_name}
    old_cols = list(step.columns.values())
    rename_map = {v: k for k, v in step.columns.items()}
    return df[old_cols].rename(columns=rename_map)


def _execute_derive(df: pd.DataFrame, step: DeriveStep) -> pd.DataFrame:
    result = df.copy()
    for col_name, expr in step.columns.items():
        result[col_name] = _eval_derive_expr(result, expr)
    return result


def _execute_group_by(df: pd.DataFrame, step: GroupByStep) -> pd.DataFrame:
    agg_dict: dict[str, list[str]] = {}
    for col, agg_spec in step.aggs.items():
        if isinstance(agg_spec, str):
            agg_dict[col] = [agg_spec]
        else:
            agg_dict[col] = list(agg_spec)
    grouped = df.groupby(step.keys).agg(agg_dict)
    # Flatten multi-index columns: (col, agg) -> col_agg
    grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
    grouped = grouped.reset_index()
    return grouped


def _execute_join(
    df: pd.DataFrame,
    step: JoinStep,
    resolver: Callable[[str], pd.DataFrame],
) -> pd.DataFrame:
    other_df = resolver(step.other)

    if isinstance(step.on, dict):
        join_keys_other = list(step.on.values())
    else:
        join_keys_other = list(step.on)

    if step.select:
        other_df = other_df[[*join_keys_other, *step.select]]

    if step.prefix:
        rename = {
            c: f"{step.prefix}{c}"
            for c in other_df.columns
            if c not in join_keys_other
        }
        other_df = other_df.rename(columns=rename)

    if isinstance(step.on, dict):
        result = df.merge(
            other_df,
            left_on=list(step.on.keys()),
            right_on=list(step.on.values()),
            how=step.how,
        )
        # Drop right-side key columns that differ from left-side keys
        # (e.g. joining TeamA->TeamID means TeamID is redundant)
        drop_cols = [v for k, v in step.on.items() if k != v and v in result.columns]
        if drop_cols:
            result = result.drop(columns=drop_cols)
    else:
        result = df.merge(other_df, on=step.on, how=step.how)
    return result


def _execute_union(
    df: pd.DataFrame,
    step: UnionStep,
    resolver: Callable[[str], pd.DataFrame],
) -> pd.DataFrame:
    other_df = resolver(step.other)
    return pd.concat([df, other_df], ignore_index=True)


def _execute_unpivot(df: pd.DataFrame, step: UnpivotStep) -> pd.DataFrame:
    # Determine N (the number of variants) from the first unpivot column list
    lengths = [len(sources) for sources in step.unpivot_columns.values()]
    if len(set(lengths)) != 1:
        raise ValueError(
            "All unpivot_columns entries must have the same number of source columns, "
            f"got lengths: {lengths}"
        )
    n_variants = lengths[0]

    # Determine the first unpivot column's sources for default naming
    first_sources = next(iter(step.unpivot_columns.values()))

    parts: list[pd.DataFrame] = []
    for i in range(n_variants):
        sub = df[step.id_columns].copy()
        for new_col, sources in step.unpivot_columns.items():
            sub[new_col] = df[sources[i]].values
        if step.names_column is not None:
            raw_name = first_sources[i]
            if step.names_map and raw_name in step.names_map:
                sub[step.names_column] = step.names_map[raw_name]
            else:
                sub[step.names_column] = raw_name
        parts.append(sub)

    return pd.concat(parts, ignore_index=True)


def _execute_cast(df: pd.DataFrame, step: CastStep) -> pd.DataFrame:
    result = df.copy()
    for col, type_spec in step.columns.items():
        if ":" in type_spec:
            # Format: "target_type:source_expr"  e.g. "int:str[1:3]"
            target_type_str, source_expr = type_spec.split(":", 1)
            target_type = _TYPE_MAP.get(target_type_str, target_type_str)
            # Evaluate the source expression against the column
            result[col] = _eval_cast_source(result[col], source_expr).astype(target_type)
        else:
            target_type = _TYPE_MAP.get(type_spec, type_spec)
            result[col] = result[col].astype(target_type)
    return result


def _execute_sort(df: pd.DataFrame, step: SortStep) -> pd.DataFrame:
    by = step.by if isinstance(step.by, list) else [step.by]
    return df.sort_values(by=by, ascending=step.ascending).reset_index(drop=True)


def _execute_distinct(df: pd.DataFrame, step: DistinctStep) -> pd.DataFrame:
    return df.drop_duplicates(subset=step.columns, keep=step.keep).reset_index(drop=True)


def _rolling_slope(window: pd.Series) -> float:
    """OLS slope over a rolling window."""
    n = len(window)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = window.values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x, y = x[mask], y[mask]
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y_mean)).sum() / denom)


def _ema_last(window: pd.Series, alpha: float = 0.3) -> float:
    """Exponential moving average, return last value."""
    if len(window) == 0:
        return np.nan
    return float(window.ewm(alpha=alpha, adjust=False).mean().iloc[-1])


# Aggregations that require `.apply()` with a per-window lambda rather than
# the built-in pandas `.agg(name)` path.
_CUSTOM_ROLLING_AGGS: dict[str, Callable] = {
    "median": lambda w: w.median(),
    "skew": lambda w: w.skew(),
    "kurt": lambda w: w.kurt(),
    "slope": _rolling_slope,
    "ema": _ema_last,
    "range": lambda w: w.max() - w.min(),
    "cv": lambda w: w.std() / (abs(w.mean()) + 1e-8),
    "pct_change": lambda w: (
        (w.iloc[-1] - w.iloc[0]) / (abs(w.iloc[0]) + 1e-8)
        if len(w) > 0
        else 0.0
    ),
    "first": lambda w: w.iloc[0] if len(w) > 0 else np.nan,
    "last": lambda w: w.iloc[-1] if len(w) > 0 else np.nan,
}


def _execute_rolling(df: pd.DataFrame, step: RollingStep) -> pd.DataFrame:
    """Rolling window aggregation partitioned by keys.

    Agg format: ``{new_col: "source_col:func"}``.

    Built-in pandas aggregation names (mean, std, sum, min, max, count) are
    dispatched via ``.agg(name)``.  Custom aggregations (median, skew, kurt,
    slope, ema, range, cv, pct_change, first, last) use ``.apply(fn)``.
    """
    result = df.sort_values(by=[*step.keys, step.order_by]).copy()
    min_periods = step.min_periods if step.min_periods is not None else step.window

    for new_col, spec in step.aggs.items():
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Rolling agg spec must be 'column:func', got {spec!r}"
            )
        src_col, func = parts[0].strip(), parts[1].strip()

        rolling_obj = (
            result.groupby(step.keys)[src_col]
            .rolling(window=step.window, min_periods=min_periods)
        )

        custom_fn = _CUSTOM_ROLLING_AGGS.get(func)
        if custom_fn is not None:
            rolled = rolling_obj.apply(custom_fn, raw=False)
        else:
            rolled = rolling_obj.agg(func)

        # rolling inside groupby produces multi-index; align back via droplevel
        result[new_col] = rolled.droplevel(list(range(len(step.keys)))).values

    return result.reset_index(drop=True)


def _execute_head(df: pd.DataFrame, step: HeadStep) -> pd.DataFrame:
    """Take first or last N rows per group."""
    if step.order_by is not None:
        by = step.order_by if isinstance(step.order_by, list) else [step.order_by]
        asc = step.ascending
        if step.position == "last":
            # Reverse sort so .head() grabs the "last" rows
            asc = [not a for a in asc] if isinstance(asc, list) else not asc
        df = df.sort_values(by=by, ascending=asc)
    elif step.position == "last":
        df = df.iloc[::-1]

    grouped = df.groupby(step.keys, sort=False)
    return grouped.head(step.n).reset_index(drop=True)


def _execute_rank(df: pd.DataFrame, step: RankStep) -> pd.DataFrame:
    """Add rank columns, optionally within groups."""
    result = df.copy()
    for new_col, src_col in step.columns.items():
        if step.keys:
            result[new_col] = result.groupby(step.keys)[src_col].rank(
                method=step.method, ascending=step.ascending, pct=step.pct,
            )
        else:
            result[new_col] = result[src_col].rank(
                method=step.method, ascending=step.ascending, pct=step.pct,
            )
    return result


def _execute_cond_agg(df: pd.DataFrame, step: ConditionalAggStep) -> pd.DataFrame:
    """Group and aggregate with optional per-agg conditions.

    Spec format: ``"source_col:func"`` or ``"source_col:func:where_expr"``.
    """
    agg_frames: list[pd.Series] = []
    agg_names: list[str] = []

    for new_col, spec in step.aggs.items():
        parts = spec.split(":", 2)
        if len(parts) < 2:
            raise ValueError(
                f"cond_agg spec must be 'column:func' or 'column:func:condition', "
                f"got {spec!r}"
            )
        src_col, func = parts[0].strip(), parts[1].strip()
        condition = parts[2].strip() if len(parts) == 3 else None

        subset = df if condition is None else df.query(condition)
        series = subset.groupby(step.keys)[src_col].agg(func)
        series.name = new_col
        agg_frames.append(series)
        agg_names.append(new_col)

    result = pd.concat(agg_frames, axis=1).reset_index()
    return result


def _execute_isin(df: pd.DataFrame, step: IsInStep) -> pd.DataFrame:
    """Filter rows where column value is (or is not) in a list."""
    mask = df[step.column].isin(step.values)
    if step.negate:
        mask = ~mask
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lag step
# ---------------------------------------------------------------------------


def _execute_lag(df: pd.DataFrame, step: LagStep) -> pd.DataFrame:
    """Shift column values within groups (lag/lead)."""
    result = df.sort_values(step.keys + [step.order_by]).copy()
    for new_col, spec in step.columns.items():
        src_col, lag_str = spec.rsplit(":", 1)
        lag_n = int(lag_str)
        result[new_col] = result.groupby(step.keys)[src_col].shift(lag_n)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# EWM step
# ---------------------------------------------------------------------------


def _execute_ewm(df: pd.DataFrame, step: EwmStep) -> pd.DataFrame:
    """Exponentially weighted moving statistic within groups."""
    result = df.sort_values(step.keys + [step.order_by]).copy()
    span = step.span
    for new_col, spec in step.aggs.items():
        src_col, stat = spec.rsplit(":", 1)
        result[new_col] = result.groupby(step.keys)[src_col].transform(
            lambda s: getattr(s.ewm(span=span, adjust=False), stat)()
        )
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Diff step
# ---------------------------------------------------------------------------


def _execute_diff(df: pd.DataFrame, step: DiffStep) -> pd.DataFrame:
    """First/second differences or percent change within groups."""
    result = df.sort_values(step.keys + [step.order_by]).copy()
    for new_col, spec in step.columns.items():
        src_col, periods_str = spec.rsplit(":", 1)
        periods = int(periods_str)
        if step.pct:
            result[new_col] = result.groupby(step.keys)[src_col].pct_change(
                periods=periods
            )
        else:
            result[new_col] = result.groupby(step.keys)[src_col].diff(
                periods=periods
            )
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trend step
# ---------------------------------------------------------------------------


def _execute_trend(df: pd.DataFrame, step: TrendStep) -> pd.DataFrame:
    """OLS slope over a rolling window within groups."""
    result = df.sort_values(step.keys + [step.order_by]).copy()
    for new_col, src_col in step.columns.items():
        rolling_obj = (
            result.groupby(step.keys)[src_col]
            .rolling(window=step.window, min_periods=step.window)
        )
        rolled = rolling_obj.apply(_rolling_slope, raw=False)
        result[new_col] = rolled.droplevel(list(range(len(step.keys)))).values
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Encode step
# ---------------------------------------------------------------------------

_CYCLICAL_PERIODS: dict[str, int] = {
    "month": 12,
    "dayofweek": 7,
    "hour": 24,
    "quarter": 4,
    "weekofyear": 53,
    "day": 31,
    "minute": 60,
    "second": 60,
}


def _execute_encode(df: pd.DataFrame, step: EncodeStep) -> pd.DataFrame:
    """Categorical encoding."""
    result = df.copy()
    col = step.column
    out = step.output if step.output is not None else f"{col}_encoded"
    method = step.method

    if method == "frequency":
        freq = result[col].value_counts(normalize=True)
        result[out] = result[col].map(freq)
    elif method == "ordinal":
        freq = result[col].value_counts()
        # most common = 1
        rank_map = {v: i + 1 for i, v in enumerate(freq.index)}
        result[out] = result[col].map(rank_map)
    elif method == "target_loo":
        if "target" not in result.columns:
            raise ValueError("target_loo encoding requires a 'target' column")
        global_mean = result["target"].mean()
        group_sum = result.groupby(col)["target"].transform("sum")
        group_count = result.groupby(col)["target"].transform("count")
        # leave-one-out: (sum - this_row) / (count - 1)
        result[out] = (group_sum - result["target"]) / (group_count - 1)
        # Groups with only 1 member get NaN; fill with global mean
        result[out] = result[out].fillna(global_mean)
    elif method == "target_temporal":
        if "target" not in result.columns:
            raise ValueError("target_temporal encoding requires a 'target' column")
        # For each row, mean of target for all prior rows with same category
        result[out] = np.nan
        for cat in result[col].unique():
            mask = result[col] == cat
            targets = result.loc[mask, "target"]
            expanding_mean = targets.expanding().mean().shift(1)
            result.loc[mask, out] = expanding_mean.values
    else:
        raise ValueError(f"Unknown encode method: {method!r}")

    return result


# ---------------------------------------------------------------------------
# Bin step
# ---------------------------------------------------------------------------


def _execute_bin(df: pd.DataFrame, step: BinStep) -> pd.DataFrame:
    """Discretize a continuous column into bins."""
    result = df.copy()
    col = step.column
    out = step.output if step.output is not None else f"{col}_binned"
    method = step.method

    if method == "quantile":
        result[out] = pd.qcut(result[col], step.n_bins, labels=False, duplicates="drop")
    elif method == "uniform":
        result[out] = pd.cut(result[col], step.n_bins, labels=False)
    elif method == "custom":
        if step.boundaries is None:
            raise ValueError("custom binning requires 'boundaries'")
        result[out] = pd.cut(result[col], bins=step.boundaries, labels=False)
    elif method == "kmeans":
        from sklearn.cluster import KMeans

        vals = result[col].dropna().values.reshape(-1, 1)
        km = KMeans(n_clusters=step.n_bins, random_state=42, n_init=10)
        km.fit(vals)
        # Predict labels for all rows (including NaN handled separately)
        labels = pd.Series(np.nan, index=result.index)
        non_null_mask = result[col].notna()
        labels[non_null_mask] = km.predict(
            result.loc[non_null_mask, col].values.reshape(-1, 1)
        )
        result[out] = labels
    else:
        raise ValueError(f"Unknown bin method: {method!r}")

    return result


# ---------------------------------------------------------------------------
# Datetime step
# ---------------------------------------------------------------------------


def _execute_datetime(df: pd.DataFrame, step: DatetimeStep) -> pd.DataFrame:
    """Extract calendar features from a datetime column."""
    result = df.copy()
    dt = pd.to_datetime(result[step.column])

    if step.extract:
        for part in step.extract:
            if part == "weekofyear":
                result[f"{step.column}_{part}"] = dt.dt.isocalendar().week.astype(int)
            else:
                result[f"{step.column}_{part}"] = getattr(dt.dt, part)

    if step.cyclical:
        for part in step.cyclical:
            period = _CYCLICAL_PERIODS.get(part)
            if period is None:
                raise ValueError(
                    f"No known period for cyclical encoding of {part!r}. "
                    f"Known: {sorted(_CYCLICAL_PERIODS)}"
                )
            if part == "weekofyear":
                raw = dt.dt.isocalendar().week.astype(int)
            else:
                raw = getattr(dt.dt, part)
            result[f"{step.column}_{part}_sin"] = np.sin(2 * np.pi * raw / period)
            result[f"{step.column}_{part}_cos"] = np.cos(2 * np.pi * raw / period)

    return result


# ---------------------------------------------------------------------------
# Null indicator step
# ---------------------------------------------------------------------------


def _execute_null_indicator(df: pd.DataFrame, step: NullIndicatorStep) -> pd.DataFrame:
    """Create binary indicators for missing values."""
    result = df.copy()
    prefix = step.prefix
    for col in step.columns:
        result[f"{prefix}{col}"] = result[col].isna().astype(int)
    return result


# ---------------------------------------------------------------------------
# Derive expression evaluator
# ---------------------------------------------------------------------------

# Regex patterns for derive expression parsing
_WHERE_RE = re.compile(r"^where\((.+)\)$", re.DOTALL)
_STR_ACCESSOR_RE = re.compile(
    r"^(\w+)((?:\.str(?:\[.+?\]|\.\w+\(.*?\)))+(?:\.astype\(\w+\))?)$"
)
_SIMPLE_ASTYPE_RE = re.compile(r"^(\w+)\.astype\((\w+)\)$")


def _eval_derive_expr(df: pd.DataFrame, expr: str) -> pd.Series:
    """Evaluate a derive expression against a DataFrame.

    Supports:
    - where(condition, true_val, false_val)
    - col.str[...], col.str.method()
    - col.astype(type)
    - Chained: col.str[1:3].astype(int)
    - Arithmetic via pd.eval
    """
    expr = expr.strip()

    # 1) where(...) calls
    m = _WHERE_RE.match(expr)
    if m:
        return _eval_where(df, m.group(1).strip())

    # 2) Chained .str / .astype patterns
    m = _STR_ACCESSOR_RE.match(expr)
    if m:
        col_name = m.group(1)
        chain = m.group(2)
        return _eval_chain(df, col_name, chain)

    # 3) Simple col.astype(type) without .str
    m = _SIMPLE_ASTYPE_RE.match(expr)
    if m:
        col_name, type_name = m.group(1), m.group(2)
        target = _TYPE_MAP.get(type_name, type_name)
        return df[col_name].astype(target)

    # 4) Fallback: pd.eval for arithmetic
    try:
        return df.eval(expr)
    except Exception:
        raise ValueError(f"Cannot evaluate derive expression: {expr!r}")


def _eval_where(df: pd.DataFrame, args_str: str) -> pd.Series:
    """Evaluate where(condition, true_val, false_val)."""
    # Split the top-level arguments by comma, respecting parentheses
    parts = _split_where_args(args_str)
    if len(parts) != 3:
        raise ValueError(
            f"where() requires exactly 3 arguments, got {len(parts)}: {args_str!r}"
        )
    cond_str, true_str, false_str = [p.strip() for p in parts]

    condition = df.eval(cond_str)
    true_val = _resolve_where_operand(df, true_str)
    false_val = _resolve_where_operand(df, false_str)

    return pd.Series(np.where(condition, true_val, false_val), index=df.index)


def _split_where_args(args_str: str) -> list[str]:
    """Split where() arguments by top-level commas (respecting parens)."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in args_str:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _resolve_where_operand(df: pd.DataFrame, token: str) -> pd.Series | float | int | str:
    """Resolve a where() operand -- column reference, number, or string literal."""
    token = token.strip()
    # Numeric literal
    try:
        return float(token) if "." in token else int(token)
    except ValueError:
        pass
    # String literal (quoted)
    if (token.startswith("'") and token.endswith("'")) or (
        token.startswith('"') and token.endswith('"')
    ):
        return token[1:-1]
    # Column reference
    if token in df.columns:
        return df[token]
    # Try eval as expression
    try:
        return df.eval(token)
    except Exception:
        return token


def _eval_chain(df: pd.DataFrame, col_name: str, chain: str) -> pd.Series:
    """Evaluate a chain of .str and .astype operations on a column."""
    result = df[col_name]

    # Tokenize the chain into individual operations
    # Matches: .str[...], .str.method(...), .astype(...)
    token_re = re.compile(r"\.(str\[.+?\]|str\.\w+\(.*?\)|astype\(\w+\))")
    tokens = token_re.findall(chain)

    for tok in tokens:
        if tok.startswith("str["):
            # e.g. str[1:3]
            slice_expr = tok[4:-1]  # extract "1:3"
            parts = slice_expr.split(":")
            if len(parts) == 2:
                start = int(parts[0]) if parts[0] else None
                stop = int(parts[1]) if parts[1] else None
                result = result.str[start:stop]
            elif len(parts) == 1:
                idx = int(parts[0])
                result = result.str[idx]
            else:
                raise ValueError(f"Unsupported str slice: {tok!r}")
        elif tok.startswith("str."):
            # e.g. str.lower(), str.upper(), str.strip()
            method_match = re.match(r"str\.(\w+)\((.*?)\)", tok)
            if method_match:
                method_name = method_match.group(1)
                method = getattr(result.str, method_name)
                result = method()
            else:
                raise ValueError(f"Unsupported str method: {tok!r}")
        elif tok.startswith("astype("):
            type_name = tok[7:-1]  # extract type from "astype(int)"
            target = _TYPE_MAP.get(type_name, type_name)
            result = result.astype(target)
        else:
            raise ValueError(f"Unsupported chain token: {tok!r}")

    return result


# ---------------------------------------------------------------------------
# Cast source-expression evaluator
# ---------------------------------------------------------------------------


def _eval_cast_source(series: pd.Series, source_expr: str) -> pd.Series:
    """Evaluate a source expression in a cast spec, e.g. 'str[1:3]'."""
    source_expr = source_expr.strip()

    if source_expr.startswith("str["):
        slice_expr = source_expr[4:-1]
        parts = slice_expr.split(":")
        if len(parts) == 2:
            start = int(parts[0]) if parts[0] else None
            stop = int(parts[1]) if parts[1] else None
            return series.astype(str).str[start:stop]
        elif len(parts) == 1:
            idx = int(parts[0])
            return series.astype(str).str[idx]

    raise ValueError(f"Unsupported cast source expression: {source_expr!r}")
