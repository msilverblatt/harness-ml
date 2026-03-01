"""Shared data utilities for feature identification and path resolution.

Provides the canonical way to identify feature columns and resolve
the features parquet path from a DataConfig. All feature tools
should use these helpers instead of hardcoding column prefixes or
parquet filenames.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from easyml.runner.schema import DataConfig


def get_feature_columns(
    df: pd.DataFrame,
    config: DataConfig,
    *,
    numeric_only: bool = True,
) -> list[str]:
    """Identify feature columns by excluding keys, target, time, and explicit exclusions.

    Features = all columns except key_columns + target_column + time_column + exclude_columns.
    No prefix convention — works for any naming scheme.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
    config : DataConfig
        Project data configuration.
    numeric_only : bool
        If True (default), only return numeric columns.

    Returns
    -------
    list[str]
        Column names identified as features.
    """
    non_feature: set[str] = set(config.key_columns) | {config.target_column}
    if config.time_column:
        non_feature.add(config.time_column)
    non_feature.update(config.exclude_columns)

    if numeric_only:
        candidates = df.select_dtypes(include=[np.number]).columns
    else:
        candidates = df.columns

    return [c for c in candidates if c not in non_feature]


def get_features_path(project_dir: Path, config: DataConfig | None = None) -> Path:
    """Resolve the features parquet path from config.

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    config : DataConfig | None
        If provided, use its features_dir and features_file.
        If None, load from pipeline.yaml.

    Returns
    -------
    Path
        Absolute path to the features parquet file.
    """
    if config is None:
        config = load_data_config(project_dir)

    features_dir = Path(config.features_dir)
    if not features_dir.is_absolute():
        features_dir = project_dir / features_dir

    return features_dir / config.features_file


def load_data_config(project_dir: Path) -> DataConfig:
    """Load DataConfig from the project's pipeline.yaml.

    Parameters
    ----------
    project_dir : Path
        Root project directory.

    Returns
    -------
    DataConfig
        Parsed data configuration.
    """
    pipeline_path = project_dir / "config" / "pipeline.yaml"
    if not pipeline_path.exists():
        return DataConfig()

    raw = yaml.safe_load(pipeline_path.read_text()) or {}
    data_section = raw.get("data", {})
    return DataConfig(**data_section)
