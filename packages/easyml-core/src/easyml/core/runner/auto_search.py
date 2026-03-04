"""Automated feature search -- systematically test feature transformations.

Searches over interactions (pairwise arithmetic), lags, and rolling means
to discover new candidate features ranked by absolute correlation with
the target variable.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_SEARCH_TYPES = {"interactions", "lags", "rolling"}

_DEFAULT_LAG_WINDOWS = [1, 3, 5, 10]
_DEFAULT_ROLLING_WINDOWS = [3, 5, 10, 20]


@dataclass
class SearchResult:
    """A single candidate feature discovered by auto-search."""

    name: str
    formula_or_spec: str
    score: float
    search_type: str


def _safe_corr(series: pd.Series, target: np.ndarray) -> float:
    """Compute Pearson correlation, returning 0.0 on failure or insufficient data."""
    try:
        mask = series.notna()
        if mask.sum() < 10:
            return 0.0
        corr = float(np.corrcoef(series.values[mask], target[mask])[0, 1])
        return corr if not np.isnan(corr) else 0.0
    except (TypeError, ValueError, IndexError):
        return 0.0


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise division with epsilon to avoid division by zero."""
    return a / (b + np.sign(b) * 1e-6)


def _search_interactions(
    df: pd.DataFrame,
    target: np.ndarray,
    feature_cols: list[str],
) -> list[SearchResult]:
    """Test all pairs of numeric features with +, -, *, /, abs_diff."""
    results: list[SearchResult] = []
    ops = [
        ("add", lambda a, b: a + b, "{a} + {b}"),
        ("sub", lambda a, b: a - b, "{a} - {b}"),
        ("mul", lambda a, b: a * b, "{a} * {b}"),
        ("div", lambda a, b: _safe_div(a, b), "{a} / {b}"),
        ("abs_diff", lambda a, b: (a - b).abs(), "abs({a} - {b})"),
    ]

    for col_a, col_b in itertools.combinations(feature_cols, 2):
        if col_a not in df.columns or col_b not in df.columns:
            continue
        series_a = df[col_a].astype(float)
        series_b = df[col_b].astype(float)
        for op_name, op_func, formula_tmpl in ops:
            try:
                combined = op_func(series_a, series_b)
                score = _safe_corr(combined, target)
                name = f"{op_name}_{col_a}_{col_b}"
                formula = formula_tmpl.format(a=col_a, b=col_b)
                results.append(SearchResult(
                    name=name,
                    formula_or_spec=formula,
                    score=score,
                    search_type="interactions",
                ))
            except Exception as exc:
                logger.debug("Interaction %s(%s, %s) failed: %s", op_name, col_a, col_b, exc)
    return results


def _search_lags(
    df: pd.DataFrame,
    target: np.ndarray,
    feature_cols: list[str],
) -> list[SearchResult]:
    """Test each feature shifted by lag windows."""
    results: list[SearchResult] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        for lag in _DEFAULT_LAG_WINDOWS:
            try:
                shifted = series.shift(lag)
                score = _safe_corr(shifted, target)
                name = f"lag{lag}_{col}"
                formula = f"shift({col}, {lag})"
                results.append(SearchResult(
                    name=name,
                    formula_or_spec=formula,
                    score=score,
                    search_type="lags",
                ))
            except Exception as exc:
                logger.debug("Lag %d on %s failed: %s", lag, col, exc)
    return results


def _search_rolling(
    df: pd.DataFrame,
    target: np.ndarray,
    feature_cols: list[str],
) -> list[SearchResult]:
    """Test each feature with rolling mean windows."""
    results: list[SearchResult] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        series = df[col].astype(float)
        for window in _DEFAULT_ROLLING_WINDOWS:
            try:
                rolled = series.rolling(window, min_periods=1).mean()
                score = _safe_corr(rolled, target)
                name = f"rolling{window}_{col}"
                formula = f"rolling_mean({col}, {window})"
                results.append(SearchResult(
                    name=name,
                    formula_or_spec=formula,
                    score=score,
                    search_type="rolling",
                ))
            except Exception as exc:
                logger.debug("Rolling %d on %s failed: %s", window, col, exc)
    return results


def auto_search(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    search_types: list[str] | None = None,
    top_n: int = 20,
) -> list[SearchResult]:
    """Run automated feature search over given columns.

    search_types:
        - "interactions": all numeric pairs x {+, -, *, /, abs_diff}
        - "lags": each feature x lag(1, 3, 5, 10)
        - "rolling": each feature x rolling_mean(3, 5, 10, 20)

    Returns top_n candidates ranked by absolute correlation with target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing feature columns and the target column.
    target_col : str
        Name of the target column.
    feature_cols : list[str]
        Column names to search over.
    search_types : list[str] | None
        Which search types to run. Defaults to all three.
    top_n : int
        Number of top results to return.

    Returns
    -------
    list[SearchResult]
        Top candidates sorted by absolute score descending.
    """
    if search_types is None:
        search_types = list(VALID_SEARCH_TYPES)

    invalid = set(search_types) - VALID_SEARCH_TYPES
    if invalid:
        raise ValueError(
            f"Invalid search types: {invalid}. "
            f"Valid types: {sorted(VALID_SEARCH_TYPES)}"
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    y = df[target_col].values.astype(float)

    results: list[SearchResult] = []

    if "interactions" in search_types:
        results.extend(_search_interactions(df, y, feature_cols))

    if "lags" in search_types:
        results.extend(_search_lags(df, y, feature_cols))

    if "rolling" in search_types:
        results.extend(_search_rolling(df, y, feature_cols))

    # Sort by absolute score descending, take top_n
    results.sort(key=lambda r: abs(r.score), reverse=True)
    return results[:top_n]


def format_auto_search_report(results: list[SearchResult]) -> str:
    """Format auto-search results as a markdown table.

    Parameters
    ----------
    results : list[SearchResult]
        Output of auto_search().

    Returns
    -------
    str
        Markdown-formatted report.
    """
    if not results:
        return "No auto-search results found."

    lines = [
        "## Auto Feature Search Results",
        "",
        "| Rank | Name | Formula / Spec | |Corr| | Type |",
        "|------|------|---------------|-------|------|",
    ]
    for i, r in enumerate(results, 1):
        lines.append(
            f"| {i} | {r.name} | `{r.formula_or_spec}` "
            f"| {abs(r.score):.4f} | {r.search_type} |"
        )

    return "\n".join(lines)
