from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Common join key candidates (checked in order)
_JOIN_KEY_CANDIDATES = [
    ["season", "game_id"],        # matchup-level join
    ["Season", "GameID"],
    ["season"],                   # season-level join (broadcasts)
    ["Season"],
]

# Common team ID columns
_TEAM_ID_CANDIDATES = ["team_id", "TeamID", "team"]


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


def _detect_join_keys(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
) -> list[str] | None:
    """Auto-detect join keys between new data and existing features."""
    new_cols = set(new_df.columns)
    existing_cols = set(existing_df.columns)

    for candidates in _JOIN_KEY_CANDIDATES:
        if all(c in new_cols and c in existing_cols for c in candidates):
            return candidates

    return None


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
    target_col: str = "result",
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Compute correlation of new columns with target, return top N."""
    if target_col not in df.columns:
        return []

    correlations = []
    target = df[target_col].astype(float)
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
    source_registered: bool = False

    def format_summary(self) -> str:
        """Markdown summary for tool response."""
        lines = [f"## Ingested: {self.name}\n"]
        lines.append(f"- **Rows matched**: {self.rows_matched} / {self.rows_total}")
        lines.append(f"- **Columns added**: {len(self.columns_added)}")

        if self.columns_added:
            cols_preview = ", ".join(self.columns_added[:10])
            if len(self.columns_added) > 10:
                cols_preview += f", ... (+{len(self.columns_added) - 10} more)"
            lines.append(f"- **Columns**: {cols_preview}")

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
        if high_null:
            lines.append("\n### High Null Columns (>10%)\n")
            for col, rate in sorted(high_null.items(), key=lambda x: -x[1]):
                lines.append(f"- {col}: {rate:.1%}")

        return "\n".join(lines)


def ingest_dataset(
    project_dir: Path,
    data_path: str,
    *,
    join_on: list[str] | None = None,
    target_col: str = "result",
    name: str | None = None,
    prefix: str | None = None,
    features_dir: str | None = None,
) -> IngestResult:
    """Add a new dataset to the project's feature store.

    Steps:
    1. Read the file (CSV, parquet, Excel -- auto-detect)
    2. Detect schema: columns, types, null rates, seasons present
    3. Auto-detect join keys if not specified
    4. Merge with existing matchup_features.parquet
    5. Auto-prefix new columns if prefix specified
    6. Compute correlation preview with target
    7. Return IngestResult with summary

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
        Override path to features directory. If None, uses
        project_dir / "data" / "features".

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

    # Determine features directory
    if features_dir is not None:
        feat_dir = Path(features_dir)
    else:
        feat_dir = project_dir / "data" / "features"

    parquet_path = feat_dir / "matchup_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Existing features not found: {parquet_path}"
        )

    # Read files
    new_df = _read_file(data_file)
    existing_df = pd.read_parquet(parquet_path)

    warnings: list[str] = []
    rows_total = len(existing_df)

    # Auto-detect join keys if not specified
    if join_on is None:
        join_on = _detect_join_keys(new_df, existing_df)
        if join_on is None:
            raise ValueError(
                "Could not auto-detect join keys. "
                f"New columns: {list(new_df.columns)}. "
                f"Existing columns: {list(existing_df.columns)[:20]}... "
                "Specify join_on explicitly."
            )
        logger.info("Auto-detected join keys: %s", join_on)

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
        )

    # Apply prefix to new columns
    if prefix:
        rename_map = {c: f"{prefix}{c}" for c in new_columns}
        new_df = new_df.rename(columns=rename_map)
        new_columns = [f"{prefix}{c}" for c in new_columns]

    # Select only join keys + new columns for the merge
    merge_cols = join_on + new_columns
    merge_df = new_df[merge_cols].copy()

    # Drop duplicates on join keys (keep first)
    n_before = len(merge_df)
    merge_df = merge_df.drop_duplicates(subset=join_on, keep="first")
    if len(merge_df) < n_before:
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
        "Updated %s: added %d columns, %d/%d rows matched",
        parquet_path, len(new_columns), rows_matched, rows_total,
    )

    return IngestResult(
        name=name,
        columns_added=new_columns,
        rows_matched=rows_matched,
        rows_total=rows_total,
        null_rates=null_rates,
        correlation_preview=correlation_preview,
        warnings=warnings,
    )
