"""Config-driven data pipeline orchestrator.

Reads SourceConfig declarations from DataConfig, executes the full
source -> clean -> merge -> feature store chain.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from harnessml.core.runner.data_utils import get_features_path
from harnessml.core.runner.schema import ColumnCleaningRule, DataConfig, SourceConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline refresh."""

    sources_processed: int = 0
    sources_skipped: int = 0
    errors: dict[str, str] = field(default_factory=dict)
    columns_added: dict[str, list[str]] = field(default_factory=dict)


def resolve_cleaning_rule(
    column: str,
    source: SourceConfig,
    global_default: ColumnCleaningRule,
) -> ColumnCleaningRule:
    """Resolve the cleaning rule for a column via cascade.

    Priority: column-level > source-level > global-level.
    """
    if source.columns and column in source.columns:
        return source.columns[column]
    if "default_cleaning" in source.model_fields_set:
        return source.default_cleaning
    return global_default


def apply_cleaning_rule(series: pd.Series, rule: ColumnCleaningRule) -> pd.Series:
    """Apply a cleaning rule to a pandas Series."""
    s = series.copy()

    if rule.coerce_numeric:
        s = pd.to_numeric(s, errors="coerce")

    if rule.clip_outliers is not None:
        lower_pct, upper_pct = rule.clip_outliers
        lower = s.quantile(lower_pct / 100)
        upper = s.quantile(upper_pct / 100)
        s = s.clip(lower, upper)

    if rule.log_transform and pd.api.types.is_numeric_dtype(s):
        s = np.log1p(s.clip(lower=0))

    null_count = s.isna().sum()
    if null_count > 0:
        if rule.null_strategy == "median" and pd.api.types.is_numeric_dtype(s):
            s = s.fillna(s.median())
        elif rule.null_strategy == "mode":
            modes = s.mode()
            if len(modes) > 0:
                s = s.fillna(modes.iloc[0])
        elif rule.null_strategy == "zero":
            s = s.fillna(0)
        elif rule.null_strategy == "ffill":
            s = s.ffill()
        elif rule.null_strategy == "constant":
            if rule.null_fill_value is not None:
                s = s.fillna(rule.null_fill_value)
        elif rule.null_strategy == "drop":
            pass  # handled at DataFrame level by caller

    if rule.normalize == "zscore" and pd.api.types.is_numeric_dtype(s):
        std = s.std()
        if std > 0:
            s = (s - s.mean()) / std
    elif rule.normalize == "minmax" and pd.api.types.is_numeric_dtype(s):
        smin, smax = s.min(), s.max()
        if smax > smin:
            s = (s - smin) / (smax - smin)

    return s


def _read_source(project_dir: Path, source: SourceConfig) -> pd.DataFrame:
    """Read a source file based on its config."""
    if source.path is None:
        raise ValueError(f"Source '{source.name}' has no path configured")

    path = Path(source.path)
    if not path.is_absolute():
        path = project_dir / path

    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    fmt = source.format
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            fmt = "parquet"
        elif suffix == ".csv":
            fmt = "csv"
        elif suffix in (".xlsx", ".xls"):
            fmt = "excel"
        else:
            raise ValueError(f"Cannot auto-detect format for {path}")

    if fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "csv":
        return pd.read_csv(path)
    elif fmt == "excel":
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


class DataPipeline:
    """Config-driven data pipeline orchestrator."""

    def __init__(self, project_dir: Path, config: DataConfig) -> None:
        self.project_dir = Path(project_dir)
        self.config = config

    def refresh(self, sources: list[str] | None = None) -> PipelineResult:
        """Run the full pipeline for all (or specified) sources."""
        result = PipelineResult()

        if sources is not None:
            for name in sources:
                if name not in self.config.sources:
                    raise ValueError(
                        f"Source '{name}' not found in config. "
                        f"Available: {list(self.config.sources.keys())}"
                    )
            to_process = {k: v for k, v in self.config.sources.items() if k in sources}
        else:
            to_process = {k: v for k, v in self.config.sources.items() if v.enabled}

        features_path = get_features_path(self.project_dir, self.config)
        features_path.parent.mkdir(parents=True, exist_ok=True)

        existing_df = None
        if features_path.exists():
            existing_df = pd.read_parquet(features_path)

        for name, source in to_process.items():
            try:
                df = _read_source(self.project_dir, source)
                df = self._apply_cleaning(df, source)

                if existing_df is None:
                    existing_df = df
                    result.columns_added[name] = list(df.columns)
                else:
                    existing_df, new_cols = self._merge(existing_df, df, source)
                    result.columns_added[name] = new_cols

                result.sources_processed += 1
            except Exception as exc:
                result.errors[name] = str(exc)
                logger.error("Failed to process source '%s': %s", name, exc)

        if existing_df is not None:
            existing_df.to_parquet(features_path, index=False)

        return result

    def add_source(self, name: str, path: str, **kwargs) -> SourceConfig:
        """Register a new source and run initial ingest."""
        source = SourceConfig(name=name, path=path, **kwargs)
        self.config.sources[name] = source
        self.refresh(sources=[name])
        return source

    def remove_source(self, name: str) -> None:
        """Remove a source from config."""
        self.config.sources.pop(name, None)

    def _apply_cleaning(self, df: pd.DataFrame, source: SourceConfig) -> pd.DataFrame:
        """Apply cleaning rules to all columns in a DataFrame."""
        df = df.copy()
        drop_cols = []

        for col in df.columns:
            rule = resolve_cleaning_rule(col, source, self.config.default_cleaning)
            df[col] = apply_cleaning_rule(df[col], rule)
            if rule.null_strategy == "drop" and df[col].isna().any():
                drop_cols.append(col)

        if drop_cols:
            df = df.dropna(subset=drop_cols)

        df = df.drop_duplicates()
        return df

    def _merge(
        self,
        existing: pd.DataFrame,
        new: pd.DataFrame,
        source: SourceConfig,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Merge new data into existing features."""
        join_on = source.join_on
        if join_on is None:
            common = sorted(set(new.columns) & set(existing.columns))
            if self.config.key_columns:
                join_on = [k for k in self.config.key_columns if k in common]
            if not join_on:
                join_on = common
            if not join_on:
                raise ValueError(
                    f"Cannot auto-detect join keys for source '{source.name}'"
                )

        existing_cols = set(existing.columns)
        new_columns = [c for c in new.columns if c not in join_on and c not in existing_cols]

        if not new_columns:
            return existing, []

        merge_cols = join_on + new_columns
        merge_df = new[merge_cols].drop_duplicates(subset=join_on, keep="first")
        merged = existing.merge(merge_df, on=join_on, how="left")

        return merged, new_columns
