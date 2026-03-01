"""Data profiling utilities for exploring datasets.

Provides summary statistics, null rates, column types, and feature
analysis without requiring ad-hoc scripts.  Designed for both
programmatic use and CLI output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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


@dataclass
class DataProfile:
    """Full profile of a dataset."""

    path: str
    n_rows: int
    n_cols: int
    seasons: list[int] = field(default_factory=list)
    season_counts: dict[int, int] = field(default_factory=dict)
    label_column: str | None = None
    label_distribution: dict[str, Any] = field(default_factory=dict)
    margin_column: str | None = None
    margin_stats: dict[str, float] = field(default_factory=dict)
    diff_columns: list[ColumnProfile] = field(default_factory=list)
    non_diff_columns: list[ColumnProfile] = field(default_factory=list)
    high_null_columns: list[ColumnProfile] = field(default_factory=list)
    zero_variance_columns: list[str] = field(default_factory=list)

    def format_summary(self) -> str:
        """Format as human-readable summary."""
        lines: list[str] = []
        lines.append(f"Dataset: {self.path}")
        lines.append(f"Shape: {self.n_rows} rows x {self.n_cols} columns")

        if self.seasons:
            lines.append(f"Seasons: {self.seasons[0]}-{self.seasons[-1]} ({len(self.seasons)} seasons)")
            # Show per-season counts for backtest seasons
            bt_seasons = [s for s in self.seasons if s >= 2015 and s != 2020]
            if bt_seasons:
                counts = [f"{s}:{self.season_counts.get(s, 0)}" for s in bt_seasons]
                lines.append(f"Backtest season games: {', '.join(counts)}")

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
            lines.append(f"\nHigh-null columns (>50%):")
            # Group by null percentage ranges
            for col in sorted(self.high_null_columns, key=lambda c: -c.null_pct):
                lines.append(f"  {col.name}: {col.null_pct:.1f}% null")

        if self.zero_variance_columns:
            lines.append(f"\nZero-variance columns ({len(self.zero_variance_columns)}):")
            for name in self.zero_variance_columns:
                lines.append(f"  {name}")

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
        header = f"{'Column':<45} {'Type':<10} {'Null%':>6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}"
        lines.append(header)
        lines.append("-" * len(header))

        for col in sorted(cols, key=lambda c: c.name):
            mean_s = f"{col.mean:.4f}" if col.mean is not None else ""
            std_s = f"{col.std:.4f}" if col.std is not None else ""
            min_s = f"{col.min:.2f}" if col.min is not None else ""
            max_s = f"{col.max:.2f}" if col.max is not None else ""
            lines.append(
                f"{col.name:<45} {col.dtype:<10} {col.null_pct:>5.1f}% "
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


def profile_dataset(
    path: str | Path,
    high_null_threshold: float = 50.0,
) -> DataProfile:
    """Profile a matchup features parquet file.

    Parameters
    ----------
    path : str | Path
        Path to the parquet file.
    high_null_threshold : float
        Columns with null percentage above this are flagged.

    Returns
    -------
    DataProfile
        Comprehensive dataset profile.
    """
    path = Path(path)
    df = pd.read_parquet(path)

    profile = DataProfile(
        path=str(path),
        n_rows=len(df),
        n_cols=len(df.columns),
    )

    # Season analysis
    season_col = _find_column(df, ["Season", "season"])
    if season_col:
        seasons = sorted(df[season_col].dropna().unique().astype(int))
        profile.seasons = seasons
        profile.season_counts = df[season_col].value_counts().to_dict()
        profile.season_counts = {int(k): int(v) for k, v in profile.season_counts.items()}

    # Label analysis
    label_col = _find_column(df, ["result", "TeamAWon"])
    if label_col:
        profile.label_column = label_col
        profile.label_distribution = {
            "mean": float(df[label_col].mean()),
            "count_1": int(df[label_col].sum()),
            "count_0": int((1 - df[label_col]).sum()),
        }

    # Margin analysis
    margin_col = _find_column(df, ["margin", "TeamAMargin"])
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
    skip_cols = {season_col, label_col, margin_col} - {None}
    # Also skip ID-like columns
    id_patterns = ["TeamA", "TeamB", "game_id", "matchup_id"]

    for col_name in df.columns:
        if col_name in skip_cols:
            continue
        if any(col_name.startswith(p) for p in id_patterns):
            continue

        col = df[col_name]
        cp = _profile_column(col_name, col, len(df))

        if col_name.startswith("diff_"):
            profile.diff_columns.append(cp)
        else:
            profile.non_diff_columns.append(cp)

        if cp.null_pct >= high_null_threshold:
            profile.high_null_columns.append(cp)

    # Zero-variance columns
    for col in profile.diff_columns + profile.non_diff_columns:
        if col.std is not None and col.std == 0.0:
            profile.zero_variance_columns.append(col.name)

    return profile


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _profile_column(name: str, series: pd.Series, n_rows: int) -> ColumnProfile:
    """Build a ColumnProfile for a single column."""
    null_count = int(series.isna().sum())
    null_pct = null_count / n_rows * 100 if n_rows > 0 else 0.0
    n_unique = int(series.nunique())

    cp = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        null_count=null_count,
        null_pct=null_pct,
        n_unique=n_unique,
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
