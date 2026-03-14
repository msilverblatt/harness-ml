"""Data profiling utilities for exploring datasets.

Provides summary statistics, null rates, column types, and feature
analysis without requiring ad-hoc scripts.  Designed for both
programmatic use and CLI output.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from harnessml.core.runner.hooks import get_id_patterns, get_label_candidates, get_margin_candidates
from harnessml.core.runner.schema import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class PluginSection:
    """Output from a profiler plugin."""
    name: str
    content: str


@dataclass
class ColumnProfile:
    """Profile of a single column."""

    name: str
    dtype: str
    null_count: int
    null_pct: float
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    cardinality: int | None = None
    inferred_type: str | None = None


@dataclass
class DataProfile:
    """Full profile of a dataset."""

    path: str
    n_rows: int
    n_cols: int
    periods: list[int] = field(default_factory=list)
    period_counts: dict[int, int] = field(default_factory=dict)
    label_column: str | None = None
    label_distribution: dict[str, Any] = field(default_factory=dict)
    margin_column: str | None = None
    margin_stats: dict[str, float] = field(default_factory=dict)
    time_column: str | None = None
    feature_columns: list[ColumnProfile] = field(default_factory=list)
    high_null_columns: list[ColumnProfile] = field(default_factory=list)
    zero_variance_columns: list[str] = field(default_factory=list)
    plugin_sections: list[PluginSection] = field(default_factory=list)

    @property
    def diff_columns(self) -> list[ColumnProfile]:
        """Feature columns whose names start with 'diff_'."""
        return [c for c in self.feature_columns if c.name.startswith("diff_")]

    @property
    def non_diff_columns(self) -> list[ColumnProfile]:
        """Feature columns whose names do not start with 'diff_'."""
        return [c for c in self.feature_columns if not c.name.startswith("diff_")]

    def format_summary(self) -> str:
        """Format as human-readable summary."""
        lines: list[str] = []
        lines.append(f"Dataset: {self.path}")
        lines.append(f"Shape: {self.n_rows} rows x {self.n_cols} columns")

        if self.periods:
            lines.append(f"Periods: {self.periods[0]}-{self.periods[-1]} ({len(self.periods)} periods)")
            # Show per-period counts for all periods
            if self.period_counts:
                counts = [f"{p}:{self.period_counts.get(p, 0)}" for p in self.periods]
                lines.append(f"Period counts: {', '.join(counts)}")

        if self.label_column:
            dist = self.label_distribution
            lines.append(f"Label: {self.label_column} (win_rate={dist.get('mean', 0):.1%})")

        if self.margin_column:
            ms = self.margin_stats
            lines.append(
                f"Margin: {self.margin_column} "
                f"(mean={ms.get('mean', 0):.1f}, std={ms.get('std', 0):.1f}, "
                f"range=[{ms.get('min', 0):.0f}, {ms.get('max', 0):.0f}])"
            )

        n_diff = len(self.diff_columns)
        n_other = len(self.non_diff_columns)
        lines.append(f"Feature columns: {n_diff} diff_ + {n_other} other = {n_diff + n_other} total")

        if self.high_null_columns:
            lines.append("\nHigh-null columns (>50%):")
            # Group by null percentage ranges
            for col in sorted(self.high_null_columns, key=lambda c: -c.null_pct):
                lines.append(f"  {col.name}: {col.null_pct:.1f}% null")

        if self.zero_variance_columns:
            lines.append(f"\nZero-variance columns ({len(self.zero_variance_columns)}):")
            for name in self.zero_variance_columns:
                lines.append(f"  {name}")

        if self.plugin_sections:
            for section in self.plugin_sections:
                lines.append(f"\n{section.content}")

        return "\n".join(lines)

    def format_columns(self, category: str | None = None) -> str:
        """Format column details, optionally filtered by category.

        Categories: 'diff', 'non_diff', 'high_null', 'all'.
        """
        if category == "diff":
            cols = self.diff_columns
        elif category == "non_diff":
            cols = self.non_diff_columns
        elif category == "high_null":
            cols = self.high_null_columns
        else:
            cols = self.diff_columns + self.non_diff_columns

        if not cols:
            return "No columns found."

        lines: list[str] = []
        header = (
            f"{'Column':<45} {'Type':<10} {'Inferred':<15} "
            f"{'Null%':>6} {'Card':>6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for col in sorted(cols, key=lambda c: c.name):
            mean_s = f"{col.mean:.4f}" if col.mean is not None else ""
            std_s = f"{col.std:.4f}" if col.std is not None else ""
            min_s = f"{col.min:.2f}" if col.min is not None else ""
            max_s = f"{col.max:.2f}" if col.max is not None else ""
            inferred = col.inferred_type or ""
            card_s = str(col.cardinality) if col.cardinality is not None else ""
            lines.append(
                f"{col.name:<45} {col.dtype:<10} {inferred:<15} "
                f"{col.null_pct:>5.1f}% {card_s:>6} "
                f"{mean_s:>10} {std_s:>10} {min_s:>10} {max_s:>10}"
            )

        return "\n".join(lines)

    def format_null_tiers(self) -> str:
        """Format columns grouped by null percentage tier."""
        all_cols = self.diff_columns + self.non_diff_columns
        tiers: dict[str, list[str]] = {
            "0%": [],
            "1-25%": [],
            "25-50%": [],
            "50-75%": [],
            "75-100%": [],
        }

        for col in all_cols:
            if col.null_pct == 0:
                tiers["0%"].append(col.name)
            elif col.null_pct <= 25:
                tiers["1-25%"].append(col.name)
            elif col.null_pct <= 50:
                tiers["25-50%"].append(col.name)
            elif col.null_pct <= 75:
                tiers["50-75%"].append(col.name)
            else:
                tiers["75-100%"].append(col.name)

        lines: list[str] = []
        for tier_name, cols in tiers.items():
            if cols:
                lines.append(f"\n{tier_name} null ({len(cols)} columns):")
                for name in sorted(cols):
                    lines.append(f"  {name}")

        return "\n".join(lines)


def detect_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr",
    threshold: float | None = None,
) -> list[dict]:
    """Detect statistical outliers in numeric columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    columns : list[str] | None
        Columns to check. If None, checks all numeric columns.
    method : str
        Detection method: "iqr" (interquartile range) or "zscore".
    threshold : float | None
        Sensitivity threshold. Defaults to 1.5 for IQR, 3.0 for Z-score.

    Returns
    -------
    list[dict]
        Per-column results with keys: column, n_outliers, pct_outliers,
        lower_bound, upper_bound, total_rows.
    """

    if threshold is None:
        threshold = 1.5 if method == "iqr" else 3.0

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    results = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            if std == 0:
                continue
            lower = mean - threshold * std
            upper = mean + threshold * std
        else:
            raise ValueError(f"Unknown method {method!r}. Use 'iqr' or 'zscore'.")

        outlier_mask = (series < lower) | (series > upper)
        n_outliers = int(outlier_mask.sum())

        results.append({
            "column": col,
            "n_outliers": n_outliers,
            "pct_outliers": round(100 * n_outliers / len(df), 1),
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "total_rows": len(df),
        })

    return results


def profile_dataset(
    path: str | Path | None = None,
    high_null_threshold: float = 50.0,
    config: DataConfig | None = None,
    plugins: list | None = None,
    *,
    df: pd.DataFrame | None = None,
) -> DataProfile:
    """Profile a features dataset.

    Parameters
    ----------
    path : str | Path | None
        Path to the parquet file.  Ignored when *df* is provided.
    high_null_threshold : float
        Columns with null percentage above this are flagged.
    config : DataConfig | None
        If provided, uses config fields for column identification
        instead of hardcoded heuristics.
    plugins : list | None
        Optional list of plugin instances.  Each must have a ``name``
        attribute and an ``analyze(df, config)`` method that returns
        a markdown string.
    df : pd.DataFrame | None
        Pre-loaded DataFrame.  When given, *path* is used only as a
        display label (defaults to ``"<DataFrame>"``).

    Returns
    -------
    DataProfile
        Comprehensive dataset profile.
    """
    if df is None:
        if path is None:
            raise ValueError("Either path or df must be provided.")
        path = Path(path)
        df = pd.read_parquet(path)

    label = str(path) if path is not None else "<DataFrame>"

    profile = DataProfile(
        path=label,
        n_rows=len(df),
        n_cols=len(df.columns),
    )

    if config is not None:
        _profile_with_config(df, profile, config, high_null_threshold)
    else:
        _profile_with_heuristics(df, profile, high_null_threshold)

    # Zero-variance columns
    for col in profile.feature_columns:
        if col.std is not None and col.std == 0.0:
            profile.zero_variance_columns.append(col.name)

    # Run plugins
    if plugins:
        for plugin in plugins:
            try:
                content = plugin.analyze(df, config)
                profile.plugin_sections.append(
                    PluginSection(name=plugin.name, content=content)
                )
            except Exception as exc:
                logger.warning("Plugin '%s' failed: %s", getattr(plugin, 'name', '?'), exc)

    return profile


def _profile_with_config(
    df: pd.DataFrame,
    profile: DataProfile,
    config: DataConfig,
    high_null_threshold: float,
) -> None:
    """Profile using DataConfig fields for column identification."""
    # Time/period analysis
    time_col = config.time_column
    if time_col and time_col in df.columns:
        profile.time_column = time_col
        periods = sorted(df[time_col].dropna().unique().astype(int))
        profile.periods = periods
        profile.period_counts = df[time_col].value_counts().to_dict()
        profile.period_counts = {int(k): int(v) for k, v in profile.period_counts.items()}

    # Label analysis
    label_col = config.target_column
    if label_col and label_col in df.columns:
        profile.label_column = label_col
        profile.label_distribution = {
            "mean": float(df[label_col].mean()),
            "count_1": int(df[label_col].sum()),
            "count_0": int((1 - df[label_col]).sum()),
        }

    # Build skip set from config
    skip_cols: set[str] = set()
    if label_col:
        skip_cols.add(label_col)
    if time_col:
        skip_cols.add(time_col)
    skip_cols.update(config.key_columns)
    skip_cols.update(config.exclude_columns)

    # Column profiles — all non-skipped columns are features
    for col_name in df.columns:
        if col_name in skip_cols:
            continue

        col = df[col_name]
        cp = _profile_column(col_name, col, len(df))
        profile.feature_columns.append(cp)

        if cp.null_pct >= high_null_threshold:
            profile.high_null_columns.append(cp)


def _profile_with_heuristics(
    df: pd.DataFrame,
    profile: DataProfile,
    high_null_threshold: float,
) -> None:
    """Profile using hardcoded heuristics (backward-compat path)."""
    # Time/period analysis
    time_col = _find_column(df, ["Season", "season", "year", "Year", "period", "Period", "date"])
    if time_col:
        profile.time_column = time_col
        periods = sorted(df[time_col].dropna().unique().astype(int))
        profile.periods = periods
        profile.period_counts = df[time_col].value_counts().to_dict()
        profile.period_counts = {int(k): int(v) for k, v in profile.period_counts.items()}

    # Label analysis
    label_col = _find_column(df, get_label_candidates())
    if label_col:
        profile.label_column = label_col
        profile.label_distribution = {
            "mean": float(df[label_col].mean()),
            "count_1": int(df[label_col].sum()),
            "count_0": int((1 - df[label_col]).sum()),
        }

    # Margin analysis
    margin_col = _find_column(df, get_margin_candidates())
    if margin_col:
        profile.margin_column = margin_col
        margin = df[margin_col].dropna()
        profile.margin_stats = {
            "mean": float(margin.mean()),
            "std": float(margin.std()),
            "min": float(margin.min()),
            "max": float(margin.max()),
            "median": float(margin.median()),
        }

    # Column profiles
    skip_cols = {time_col, label_col, margin_col} - {None}
    # Also skip ID-like columns (via hook system for extensibility)
    id_patterns = get_id_patterns()

    for col_name in df.columns:
        if col_name in skip_cols:
            continue
        if any(col_name.startswith(p) for p in id_patterns):
            continue

        col = df[col_name]
        cp = _profile_column(col_name, col, len(df))
        profile.feature_columns.append(cp)

        if cp.null_pct >= high_null_threshold:
            profile.high_null_columns.append(cp)


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _infer_column_type(series: pd.Series, n_unique: int, n_rows: int) -> str:
    """Infer a semantic type hint for a column.

    Returns one of: 'numeric', 'binary', 'categorical', 'high_cardinality',
    'boolean', 'datetime', 'text', 'id'.
    """
    dtype = series.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"

    if pd.api.types.is_numeric_dtype(dtype):
        if n_unique == 2:
            return "binary"
        if n_unique <= 20 and n_rows > 0:
            return "categorical"
        return "numeric"

    # Object / string columns
    if n_rows > 0:
        unique_ratio = n_unique / n_rows
        if unique_ratio > 0.9:
            return "id"
        if n_unique <= 50:
            return "categorical"
        return "high_cardinality"

    return "text"


def _profile_column(name: str, series: pd.Series, n_rows: int) -> ColumnProfile:
    """Build a ColumnProfile for a single column."""
    null_count = int(series.isna().sum())
    null_pct = null_count / n_rows * 100 if n_rows > 0 else 0.0
    n_unique = int(series.nunique())

    # Cardinality for non-numeric columns (categorical/object)
    cardinality = None
    if not pd.api.types.is_numeric_dtype(series) or n_unique <= 50:
        cardinality = n_unique

    # Infer semantic type
    inferred_type = _infer_column_type(series, n_unique, n_rows)

    cp = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        null_count=null_count,
        null_pct=null_pct,
        n_unique=n_unique,
        cardinality=cardinality,
        inferred_type=inferred_type,
    )

    if pd.api.types.is_numeric_dtype(series):
        valid = series.dropna()
        if len(valid) > 0:
            cp.mean = float(valid.mean())
            cp.std = float(valid.std())
            cp.min = float(valid.min())
            cp.max = float(valid.max())
            cp.median = float(valid.median())

    return cp
