"""Data ingestion: bootstrap, auto-clean, merge, and source tracking.

Designed for MCP tool calls — every function returns structured results
that an AI agent can present to the user.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from harnessml.core.logging import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------
# File I/O
# -----------------------------------------------------------------------

def _read_file(path: Path) -> pd.DataFrame:
    """Read CSV, parquet, or Excel file based on extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


# -----------------------------------------------------------------------
# Join key detection — generic, no hardcoded candidates
# -----------------------------------------------------------------------

def _detect_join_keys(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    key_columns: list[str] | None = None,
    exclude_cols: list[str] | None = None,
) -> list[str] | None:
    """Auto-detect join keys between new data and existing features."""
    new_cols = set(new_df.columns)
    existing_cols = set(existing_df.columns)

    # Strategy 1: use configured key_columns
    if key_columns:
        overlap = [k for k in key_columns if k in new_cols and k in existing_cols]
        if overlap:
            return overlap

    # Strategy 2: find all overlapping columns
    common = sorted(new_cols & existing_cols)
    if not common:
        return None

    # Build exclusion set
    skip = set()
    if exclude_cols:
        skip.update(c.lower() for c in exclude_cols)

    key_candidates = []
    for col in common:
        if col.lower() in skip:
            continue
        key_candidates.append(col)

    return key_candidates if key_candidates else None


# -----------------------------------------------------------------------
# Auto-cleaning
# -----------------------------------------------------------------------

def _auto_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply automatic data cleaning. Returns (cleaned_df, actions_taken).

    Actions:
    1. Detect and coerce numeric columns stored as strings
    2. Fill nulls: median for numeric, mode for categorical
    3. Drop exact duplicate rows
    """
    actions: list[str] = []
    df = df.copy()

    # 1. Coerce numeric columns stored as strings
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                non_null_original = df[col].notna().sum()
                non_null_converted = numeric.notna().sum()
                # If most values convert successfully, treat as numeric
                if non_null_original > 0 and non_null_converted / non_null_original > 0.8:
                    df[col] = numeric
                    actions.append(f"Coerced '{col}' from string to numeric")
            except (TypeError, ValueError):
                continue

    # 2. Fill nulls
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count == 0:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            actions.append(f"Filled {null_count} nulls in '{col}' with median ({median_val:.4g})")
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            mode_vals = df[col].mode()
            if len(mode_vals) > 0:
                df[col] = df[col].fillna(mode_vals.iloc[0])
                actions.append(f"Filled {null_count} nulls in '{col}' with mode ('{mode_vals.iloc[0]}')")

    # 3. Drop exact duplicate rows
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        actions.append(f"Dropped {n_dropped} exact duplicate rows")

    return df, actions


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------

def _compute_null_rates(df: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    """Compute null rates for specified columns."""
    rates = {}
    for col in columns:
        if col in df.columns:
            rates[col] = float(df[col].isna().mean())
    return rates


def _compute_correlation_preview(
    df: pd.DataFrame,
    new_columns: list[str],
    target_col: str,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Compute correlation of new columns with target, return top N."""
    if target_col not in df.columns:
        return []

    correlations = []
    for col in new_columns:
        if col in df.columns and df[col].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
            try:
                valid = df[[col, target_col]].dropna()
                if len(valid) > 10:
                    corr = valid[col].astype(float).corr(valid[target_col].astype(float))
                    if not np.isnan(corr):
                        correlations.append((col, round(float(corr), 4)))
            except (TypeError, ValueError):
                continue

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    return correlations[:top_n]


def _register_source(
    project_dir: Path,
    name: str,
    data_path: str,
    columns_added: list[str],
    rows: int,
    is_bootstrap: bool,
) -> bool:
    """Track ingested source in source_registry.json."""
    registry_path = project_dir / "data" / "source_registry.json"
    try:
        if registry_path.exists():
            registry = json.loads(registry_path.read_text())
        else:
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry = {"sources": []}

        registry["sources"].append({
            "name": name,
            "path": str(data_path),
            "columns_added": columns_added,
            "rows": rows,
            "is_bootstrap": is_bootstrap,
        })
        registry_path.write_text(json.dumps(registry, indent=2))
        return True
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("failed to register source", path=str(registry_path), error=str(exc))
        return False


# -----------------------------------------------------------------------
# IngestResult
# -----------------------------------------------------------------------

@dataclass
class IngestResult:
    """Result of a dataset ingestion."""

    name: str
    columns_added: list[str]
    rows_matched: int
    rows_total: int
    null_rates: dict[str, float]
    correlation_preview: list[tuple[str, float]]
    warnings: list[str] = field(default_factory=list)
    cleaning_actions: list[str] = field(default_factory=list)
    is_bootstrap: bool = False
    source_registered: bool = False
    auto_clean: bool = False

    def format_summary(self) -> str:
        """Markdown summary for tool response."""
        lines = [f"## Ingested: {self.name}\n"]

        if self.is_bootstrap:
            lines.append("- **Mode**: Bootstrap (first dataset)")
        lines.append(f"- **Rows matched**: {self.rows_matched} / {self.rows_total}")
        lines.append(f"- **Columns added**: {len(self.columns_added)}")

        if self.columns_added:
            cols_preview = ", ".join(self.columns_added[:10])
            if len(self.columns_added) > 10:
                cols_preview += f", ... (+{len(self.columns_added) - 10} more)"
            lines.append(f"- **Columns**: {cols_preview}")

        if self.auto_clean and self.cleaning_actions:
            lines.append("\n### Auto-Clean Applied\n")
            for action in self.cleaning_actions:
                lines.append(f"- {action}")
        elif not self.auto_clean:
            # Report null summary when auto_clean is off
            cols_with_nulls = {
                k: v for k, v in self.null_rates.items() if v > 0.0
            }
            if cols_with_nulls:
                lines.append("\n### Null Summary\n")
                for col, rate in sorted(
                    cols_with_nulls.items(), key=lambda x: -x[1]
                ):
                    count = int(round(rate * self.rows_total))
                    lines.append(
                        f"- {col}: {count} nulls ({rate:.1%})"
                    )
                lines.append(
                    "\nUse fill_nulls or null_indicator features to handle these."
                )

        if self.correlation_preview:
            lines.append("\n### Top Correlations with Target\n")
            lines.append("| Feature | Correlation |")
            lines.append("|---------|-------------|")
            for col, corr in self.correlation_preview:
                lines.append(f"| {col} | {corr:+.4f} |")

        if self.warnings:
            lines.append("\n### Warnings\n")
            for w in self.warnings:
                lines.append(f"- {w}")

        high_null = {k: v for k, v in self.null_rates.items() if v > 0.1}
        if high_null and self.auto_clean:
            lines.append("\n### High Null Columns (>10%)\n")
            for col, rate in sorted(high_null.items(), key=lambda x: -x[1]):
                lines.append(f"- {col}: {rate:.1%}")

        return "\n".join(lines)


# -----------------------------------------------------------------------
# Main ingestion function
# -----------------------------------------------------------------------

def ingest_dataset(
    project_dir: Path,
    data_path: str,
    *,
    join_on: list[str] | None = None,
    target_col: str = "result",
    name: str | None = None,
    prefix: str | None = None,
    features_dir: str | None = None,
    auto_clean: bool = False,
) -> IngestResult:
    """Add a new dataset to the project's feature store.

    Steps:
    1. Read the file (CSV, parquet, Excel -- auto-detect)
    2. If no existing features parquet, bootstrap (save as initial dataset)
    3. If existing features exist, merge on join keys
    4. Auto-clean if enabled (coerce types, fill nulls, dedup)
    5. Compute correlation preview with target
    6. Register source

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    data_path : str
        Path to the new dataset file.
    join_on : list[str] | None
        Columns to join on. Auto-detected if None.
    target_col : str
        Target column name for correlation preview.
    name : str | None
        Name for this dataset. Defaults to filename stem.
    prefix : str | None
        Prefix to add to new columns (avoids collisions).
    features_dir : str | None
        Override path to features directory.
    auto_clean : bool
        If True, auto-clean the data (coerce types, fill nulls, dedup).
        Defaults to False to preserve null signal.

    Returns
    -------
    IngestResult
        Summary of the ingestion.
    """
    project_dir = Path(project_dir)
    data_file = Path(data_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if name is None:
        name = data_file.stem

    # Resolve features path
    from harnessml.core.runner.data_utils import load_data_config

    if features_dir is not None:
        feat_dir = Path(features_dir)
    else:
        try:
            config = load_data_config(project_dir)
            feat_dir = project_dir / config.features_dir
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features dir", error=str(exc))
            feat_dir = project_dir / "data" / "features"

    try:
        config = load_data_config(project_dir)
        features_filename = config.features_file
        configured_keys = config.key_columns
        target_col = config.target_column or target_col
    except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
        logger.debug("config load failed, using default features settings", error=str(exc))
        features_filename = "features.parquet"
        configured_keys = []

    parquet_path = feat_dir / features_filename

    # Read the new data
    new_df = _read_file(data_file)
    logger.info("data_loaded", rows=len(new_df), columns=len(new_df.columns), path=str(data_file))

    # Preserve raw copy before any processing/cleaning
    raw_dir = project_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"raw_{data_file.stem}.parquet"
    new_df.to_parquet(raw_path, index=False)
    logger.info("raw_preserved", path=str(raw_path), rows=len(new_df))

    # Auto-clean the incoming data if requested
    cleaning_actions: list[str] = []
    if auto_clean:
        before_clean = len(new_df)
        new_df, cleaning_actions = _auto_clean(new_df)
        if len(new_df) < before_clean:
            logger.info("rows_filtered", before=before_clean, after=len(new_df), reason="auto_clean")

    # ---------------------------------------------------------------
    # Bootstrap: if no existing features file, save directly
    # ---------------------------------------------------------------
    if not parquet_path.exists():
        feat_dir.mkdir(parents=True, exist_ok=True)

        columns_added = list(new_df.columns)
        rows_total = len(new_df)

        # Compute correlation preview
        correlation_preview = _compute_correlation_preview(
            new_df, columns_added, target_col=target_col,
        )
        null_rates = _compute_null_rates(new_df, columns_added)

        new_df.to_parquet(parquet_path, index=False)
        logger.info("bootstrap saved", rows=rows_total, columns=len(columns_added), path=str(parquet_path))

        source_registered = _register_source(
            project_dir, name, data_path, columns_added, rows_total, is_bootstrap=True,
        )

        return IngestResult(
            name=name,
            columns_added=columns_added,
            rows_matched=rows_total,
            rows_total=rows_total,
            null_rates=null_rates,
            correlation_preview=correlation_preview,
            cleaning_actions=cleaning_actions,
            is_bootstrap=True,
            source_registered=source_registered,
            auto_clean=auto_clean,
        )

    # ---------------------------------------------------------------
    # Merge: existing features file exists
    # ---------------------------------------------------------------
    existing_df = pd.read_parquet(parquet_path)
    warnings: list[str] = []
    rows_total = len(existing_df)

    # Auto-detect join keys if not specified
    if join_on is None:
        # Build exclusion list from config
        exclude_cols = [target_col]
        try:
            exclude_cols.extend(config.exclude_columns)
        except (AttributeError, TypeError) as exc:
            logger.debug("could not read exclude_columns from config", error=str(exc))
        join_on = _detect_join_keys(
            new_df, existing_df,
            key_columns=configured_keys,
            exclude_cols=exclude_cols,
        )
        if join_on is None:
            raise ValueError(
                "Could not auto-detect join keys. "
                f"New columns: {list(new_df.columns)}. "
                f"Existing columns: {list(existing_df.columns)[:20]}... "
                "Specify join_on explicitly."
            )
        logger.info("auto-detected join keys", keys=join_on)

    # Validate join keys exist in both DataFrames
    for key in join_on:
        if key not in new_df.columns:
            raise ValueError(f"Join key '{key}' not found in new dataset")
        if key not in existing_df.columns:
            raise ValueError(f"Join key '{key}' not found in existing features")

    # Determine new columns (exclude join keys and columns already present)
    existing_cols = set(existing_df.columns)
    new_columns = [
        c for c in new_df.columns
        if c not in join_on and c not in existing_cols
    ]

    if not new_columns:
        return IngestResult(
            name=name,
            columns_added=[],
            rows_matched=rows_total,
            rows_total=rows_total,
            null_rates={},
            correlation_preview=[],
            warnings=["No new columns to add (all columns already exist)."],
            cleaning_actions=cleaning_actions,
            auto_clean=auto_clean,
        )

    # Apply prefix to new columns
    if prefix:
        rename_map = {c: f"{prefix}{c}" for c in new_columns}
        new_df = new_df.rename(columns=rename_map)
        new_columns = [f"{prefix}{c}" for c in new_columns]

    # Select only join keys + new columns for the merge
    merge_cols = join_on + new_columns
    merge_df = new_df[merge_cols].copy()

    # Log columns that already exist and were excluded
    existing_only = [c for c in new_df.columns if c not in join_on and c in existing_cols]
    if existing_only:
        logger.info("columns_dropped", columns=existing_only)

    # Drop duplicates on join keys (keep first)
    n_before = len(merge_df)
    merge_df = merge_df.drop_duplicates(subset=join_on, keep="first")
    if len(merge_df) < n_before:
        logger.info("rows_filtered", before=n_before, after=len(merge_df), reason="duplicate_join_keys")
        warnings.append(
            f"Dropped {n_before - len(merge_df)} duplicate rows on join keys."
        )

    # Merge
    merged = existing_df.merge(merge_df, on=join_on, how="left")
    rows_matched = int(merged[new_columns[0]].notna().sum())

    if rows_matched == 0:
        warnings.append("No rows matched on join keys -- all new columns are null.")
    elif rows_matched < rows_total * 0.5:
        warnings.append(
            f"Low match rate: {rows_matched}/{rows_total} "
            f"({rows_matched/rows_total:.0%}) rows matched."
        )

    # Compute null rates for new columns
    null_rates = _compute_null_rates(merged, new_columns)

    # Compute correlation preview
    correlation_preview = _compute_correlation_preview(
        merged, new_columns, target_col=target_col,
    )

    # Save updated parquet
    merged.to_parquet(parquet_path, index=False)
    logger.info(
        "features updated",
        path=str(parquet_path),
        columns_added=len(new_columns),
        rows_matched=rows_matched,
        rows_total=rows_total,
    )

    source_registered = _register_source(
        project_dir, name, data_path, new_columns, rows_total, is_bootstrap=False,
    )

    return IngestResult(
        name=name,
        columns_added=new_columns,
        rows_matched=rows_matched,
        rows_total=rows_total,
        null_rates=null_rates,
        correlation_preview=correlation_preview,
        warnings=warnings,
        cleaning_actions=cleaning_actions,
        source_registered=source_registered,
        auto_clean=auto_clean,
    )


# -----------------------------------------------------------------------
# Granular data tools (for opt-out mode / fine-grained control)
# -----------------------------------------------------------------------

def validate_dataset(
    project_dir: Path,
    data_path: str,
    *,
    features_dir: str | None = None,
) -> str:
    """Preview a dataset without ingesting. Reports schema, types, nulls, issues.

    Returns markdown-formatted report.
    """
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = _read_file(data_file)

    lines = [f"## Dataset Preview: {data_file.name}\n"]
    lines.append(f"- **Rows**: {len(df)}")
    lines.append(f"- **Columns**: {len(df.columns)}")

    lines.append("\n### Schema\n")
    lines.append("| Column | Type | Nulls | Unique |")
    lines.append("|--------|------|-------|--------|")
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isna().sum())
        null_pct = f"{df[col].isna().mean():.1%}" if null_count > 0 else "0"
        unique = df[col].nunique()
        lines.append(f"| {col} | {dtype} | {null_count} ({null_pct}) | {unique} |")

    # Check for potential issues
    issues: list[str] = []
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                if numeric.notna().sum() / max(df[col].notna().sum(), 1) > 0.8:
                    issues.append(f"'{col}' looks numeric but stored as string")
            except (TypeError, ValueError):
                pass
        if df[col].isna().mean() > 0.5:
            issues.append(f"'{col}' has >50% null values")

    # Check for duplicates
    n_dupes = len(df) - len(df.drop_duplicates())
    if n_dupes > 0:
        issues.append(f"{n_dupes} exact duplicate rows found")

    if issues:
        lines.append("\n### Potential Issues\n")
        for issue in issues:
            lines.append(f"- {issue}")

    # Check overlap with existing features
    project_dir = Path(project_dir)
    from harnessml.core.runner.data_utils import load_data_config

    try:
        config = load_data_config(project_dir)
        feat_dir = project_dir / config.features_dir if features_dir is None else Path(features_dir)
        parquet_path = feat_dir / config.features_file
    except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
        logger.debug("config load failed, using default features path", error=str(exc))
        feat_dir = project_dir / "data" / "features" if features_dir is None else Path(features_dir)
        parquet_path = feat_dir / "features.parquet"

    if parquet_path.exists():
        existing = pd.read_parquet(parquet_path)
        common_cols = sorted(set(df.columns) & set(existing.columns))
        new_cols = sorted(set(df.columns) - set(existing.columns))
        lines.append("\n### Overlap with Existing Features\n")
        lines.append(f"- **Common columns** (potential join keys): {', '.join(common_cols) if common_cols else 'none'}")
        lines.append(f"- **New columns**: {', '.join(new_cols[:15]) if new_cols else 'none'}")
        if len(new_cols) > 15:
            lines.append(f"  ... +{len(new_cols) - 15} more")
    else:
        lines.append("\n*No existing features file — this would be a bootstrap ingestion.*")

    return "\n".join(lines)


def fill_nulls(
    project_dir: Path,
    column: str,
    *,
    strategy: str = "median",
    value: float | str | None = None,
    features_dir: str | None = None,
) -> str:
    """Fill nulls in a column of the features file.

    Parameters
    ----------
    strategy : str
        "median", "mean", "mode", "zero", or "value" (requires value param).
    value : float | str | None
        Fill value when strategy is "value".

    Returns markdown summary.
    """
    project_dir = Path(project_dir)
    from harnessml.core.runner.data_utils import get_features_path, load_data_config

    if features_dir is not None:
        try:
            config = load_data_config(project_dir)
            parquet_path = Path(features_dir) / config.features_file
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features file", error=str(exc))
            parquet_path = Path(features_dir) / "features.parquet"
    else:
        try:
            config = load_data_config(project_dir)
            parquet_path = get_features_path(project_dir, config)
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features path", error=str(exc))
            parquet_path = project_dir / "data" / "features" / "features.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Features file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in features file")

    null_count = int(df[column].isna().sum())
    if null_count == 0:
        return f"Column '{column}' has no nulls — nothing to fill."

    if strategy == "median":
        fill_val = df[column].median()
    elif strategy == "mean":
        fill_val = df[column].mean()
    elif strategy == "mode":
        modes = df[column].mode()
        fill_val = modes.iloc[0] if len(modes) > 0 else None
    elif strategy == "zero":
        fill_val = 0
    elif strategy == "value":
        if value is None:
            raise ValueError("Must provide 'value' when strategy is 'value'")
        fill_val = value
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use median, mean, mode, zero, or value.")

    df[column] = df[column].fillna(fill_val)
    df.to_parquet(parquet_path, index=False)

    return f"Filled {null_count} nulls in '{column}' with {strategy} ({fill_val})"


def drop_duplicates(
    project_dir: Path,
    *,
    columns: list[str] | None = None,
    features_dir: str | None = None,
) -> str:
    """Drop duplicate rows from the features file.

    Parameters
    ----------
    columns : list[str] | None
        Subset of columns to check for duplicates. None = all columns.

    Returns markdown summary.
    """
    project_dir = Path(project_dir)
    from harnessml.core.runner.data_utils import get_features_path, load_data_config

    if features_dir is not None:
        try:
            config = load_data_config(project_dir)
            parquet_path = Path(features_dir) / config.features_file
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features file", error=str(exc))
            parquet_path = Path(features_dir) / "features.parquet"
    else:
        try:
            config = load_data_config(project_dir)
            parquet_path = get_features_path(project_dir, config)
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features path", error=str(exc))
            parquet_path = project_dir / "data" / "features" / "features.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Features file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    n_before = len(df)
    df = df.drop_duplicates(subset=columns)
    n_dropped = n_before - len(df)

    if n_dropped == 0:
        subset_desc = f"on columns {columns}" if columns else "across all columns"
        return f"No duplicates found {subset_desc}."

    df.to_parquet(parquet_path, index=False)
    subset_desc = f"on columns {columns}" if columns else "across all columns"
    return f"Dropped {n_dropped} duplicate rows {subset_desc}. {len(df)} rows remaining."


def derive_column(
    project_dir: Path,
    name: str,
    expression: str,
    *,
    group_by: str | None = None,
    dtype: str | None = None,
    features_dir: str | None = None,
) -> str:
    """Derive a new column from a pandas expression and save to the feature store.

    Supports arithmetic (``close - open``), shifts with groupby
    (``close.shift(-1) / close - 1``), boolean thresholds
    (``(value > 0).astype(int)``), and datetime accessors (``date.dt.year``).

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    name : str
        Name for the new column.
    expression : str
        Python expression referencing existing columns by name.
    group_by : str | None
        If provided, the expression is evaluated per-group (enables ``.shift()``
        and other group-aware operations).
    dtype : str | None
        Optional dtype to cast the result to (e.g. ``"int"``, ``"float"``).
    features_dir : str | None
        Override path to features directory.

    Returns
    -------
    str
        Markdown summary of the derived column.
    """
    project_dir = Path(project_dir)
    from harnessml.core.runner.data_utils import get_features_path, load_data_config

    if features_dir is not None:
        try:
            config = load_data_config(project_dir)
            parquet_path = Path(features_dir) / config.features_file
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features file", error=str(exc))
            parquet_path = Path(features_dir) / "features.parquet"
    else:
        try:
            config = load_data_config(project_dir)
            parquet_path = get_features_path(project_dir, config)
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features path", error=str(exc))
            parquet_path = project_dir / "data" / "features" / "features.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Features file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    def _build_eval_ns(frame):
        """Build a restricted namespace for eval from a DataFrame."""
        ns = {col: frame[col] for col in frame.columns}
        ns["np"] = np
        ns["pd"] = pd
        for builtin_name, builtin_obj in [
            ("int", int), ("float", float), ("str", str), ("bool", bool),
            ("abs", abs), ("round", round), ("min", min), ("max", max), ("len", len),
        ]:
            ns[builtin_name] = builtin_obj
        return ns

    try:
        if group_by is not None:
            result = df.groupby(group_by, group_keys=False).apply(
                lambda grp: grp.assign(
                    **{name: eval(expression, {"__builtins__": {}}, _build_eval_ns(grp))}  # noqa: S307
                )
            )[name]
        else:
            result = eval(expression, {"__builtins__": {}}, _build_eval_ns(df))  # noqa: S307
    except (NameError, SyntaxError, TypeError, ValueError, KeyError, ZeroDivisionError) as exc:
        raise ValueError(
            f"Failed to evaluate expression '{expression}': {exc}"
        ) from exc

    if dtype is not None:
        result = result.astype(dtype)

    df[name] = result.values if hasattr(result, "values") else result
    df.to_parquet(parquet_path, index=False)

    n_null = int(df[name].isna().sum())
    summary = f"Derived column `{name}` from `{expression}`."
    if group_by:
        summary += f" Grouped by `{group_by}`."
    summary += f"\n- Non-null: {len(df) - n_null}/{len(df)}"
    if n_null > 0:
        summary += f" ({n_null} NaN values)"
    summary += f"\n- dtype: {df[name].dtype}"
    sample = df[name].dropna()
    if len(sample) > 0:
        summary += f"\n- Sample values: {list(sample.head(5).values)}"
    return summary


def drop_rows(
    project_dir: Path,
    *,
    column: str | None = None,
    condition: str = "null",
    features_dir: str | None = None,
) -> str:
    """Drop rows from the features file by condition.

    Two modes:
    1. condition="null" with column="col_name" — drops rows where that column is NaN.
    2. condition="<pandas query expression>" — drops rows where expression is True.

    Parameters
    ----------
    column : str | None
        Column to check for NaN (required when condition="null").
    condition : str
        "null" to drop NaN rows in the specified column, or a pandas query
        expression (e.g. "value < 0") to drop matching rows.
    features_dir : str | None
        Override path to features directory.

    Returns
    -------
    str
        Markdown summary of the operation.
    """
    project_dir = Path(project_dir)
    from harnessml.core.runner.data_utils import get_features_path, load_data_config

    if features_dir is not None:
        try:
            config = load_data_config(project_dir)
            parquet_path = Path(features_dir) / config.features_file
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features file", error=str(exc))
            parquet_path = Path(features_dir) / "features.parquet"
    else:
        try:
            config = load_data_config(project_dir)
            parquet_path = get_features_path(project_dir, config)
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features path", error=str(exc))
            parquet_path = project_dir / "data" / "features" / "features.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Features file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    n_before = len(df)

    if condition == "null":
        if column is None:
            return "**Error**: `column` is required when condition is 'null'."
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in features file")
        df = df.dropna(subset=[column])
    else:
        try:
            mask = df.eval(condition)
        except (NameError, SyntaxError, TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"Failed to evaluate condition '{condition}': {exc}"
            ) from exc
        df = df[~mask]

    n_dropped = n_before - len(df)

    if n_dropped == 0:
        return f"No rows matched condition — nothing dropped. {len(df)} rows remain."

    logger.info("rows_filtered", before=n_before, after=len(df), reason=condition)
    df.to_parquet(parquet_path, index=False)
    return f"Dropped {n_dropped} rows ({condition}). {len(df)} rows remaining."


def rename_columns(
    project_dir: Path,
    mapping: dict[str, str],
    *,
    features_dir: str | None = None,
) -> str:
    """Rename columns in the features file.

    Parameters
    ----------
    mapping : dict[str, str]
        {old_name: new_name} mapping.

    Returns markdown summary.
    """
    project_dir = Path(project_dir)
    from harnessml.core.runner.data_utils import get_features_path, load_data_config

    if features_dir is not None:
        try:
            config = load_data_config(project_dir)
            parquet_path = Path(features_dir) / config.features_file
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features file", error=str(exc))
            parquet_path = Path(features_dir) / "features.parquet"
    else:
        try:
            config = load_data_config(project_dir)
            parquet_path = get_features_path(project_dir, config)
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError) as exc:
            logger.debug("config load failed, using default features path", error=str(exc))
            parquet_path = project_dir / "data" / "features" / "features.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Features file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    # Validate all old columns exist
    missing = [k for k in mapping if k not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")

    df = df.rename(columns=mapping)
    df.to_parquet(parquet_path, index=False)

    renamed = [f"'{old}' → '{new}'" for old, new in mapping.items()]
    return f"Renamed {len(mapping)} columns: {', '.join(renamed)}"


# -----------------------------------------------------------------------
# Sampling for fast iteration
# -----------------------------------------------------------------------

def sample_data(project_dir, *, fraction=0.1, stratify_column=None, seed=42):
    """Sample the feature store for fast iteration. Saves backup as features_full.parquet."""
    project_dir = Path(project_dir)
    features_dir = project_dir / "data" / "features"
    features_path = features_dir / "features.parquet"
    backup_path = features_dir / "features_full.parquet"

    if not features_path.exists():
        return "**Error**: No features.parquet found."

    df = pd.read_parquet(features_path)
    original_len = len(df)

    # Save backup if not already backed up
    if not backup_path.exists():
        df.to_parquet(backup_path, index=False)

    # Sample
    if stratify_column and stratify_column in df.columns:
        parts = []
        for _, group in df.groupby(stratify_column):
            parts.append(group.sample(frac=fraction, random_state=seed))
        sample = pd.concat(parts, ignore_index=True)
    else:
        sample = df.sample(frac=fraction, random_state=seed)

    sample.to_parquet(features_path, index=False)
    return f"Sampled {len(sample):,} rows from {original_len:,} ({fraction:.0%}). Backup saved as `features_full.parquet`."


def restore_full_data(project_dir):
    """Restore the full feature store from backup."""
    import shutil

    project_dir = Path(project_dir)
    features_dir = project_dir / "data" / "features"
    features_path = features_dir / "features.parquet"
    backup_path = features_dir / "features_full.parquet"

    if not backup_path.exists():
        return "**Error**: No backup found (`features_full.parquet` does not exist)."

    shutil.move(str(backup_path), str(features_path))
    df = pd.read_parquet(features_path)
    return f"Restored full dataset: {len(df):,} rows."
