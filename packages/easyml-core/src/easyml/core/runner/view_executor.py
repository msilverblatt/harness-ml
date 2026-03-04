"""View step execution engine -- pure functions mapping TransformStep to pandas operations."""
from __future__ import annotations

import logging
import re
from typing import Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from easyml.core.runner.schema import (
        CastStep,
        ConditionalAggStep,
        DeriveStep,
        DistinctStep,
        FilterStep,
        GroupByStep,
        HeadStep,
        IsInStep,
        JoinStep,
        RankStep,
        RollingStep,
        SelectStep,
        SortStep,
        TransformStep,
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


def _execute_rolling(df: pd.DataFrame, step: RollingStep) -> pd.DataFrame:
    """Rolling window aggregation partitioned by keys.

    Agg format: ``{new_col: "source_col:func"}``.
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
        rolled = (
            result.groupby(step.keys)[src_col]
            .rolling(window=step.window, min_periods=min_periods)
            .agg(func)
        )
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
