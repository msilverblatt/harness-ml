"""File source adapter — loads CSV, Parquet, and Excel files."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from harness.data.sources.protocol import SourceConfig


class FileSource:
    """Load data from local files (CSV, Parquet, Excel)."""

    LOADERS = {
        "csv": pd.read_csv,
        "parquet": pd.read_parquet,
        "excel": lambda path, **kw: pd.read_excel(path, **kw),
    }

    FORMAT_EXTENSIONS = {
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".xlsx": "excel",
        ".xls": "excel",
    }

    def load(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        """Load data from a file path."""
        path = self._resolve_path(config.path, base_dir)
        fmt = self._detect_format(path, config.format)
        loader = self.LOADERS.get(fmt)
        if loader is None:
            raise ValueError(f"Unsupported format: {fmt}")
        return loader(str(path), **config.params)

    def validate(self, config: SourceConfig) -> list[str]:
        """Validate that the file exists and format is supported."""
        errors = []
        if not config.path:
            errors.append("Source path is required for file sources")
            return errors
        path = Path(config.path)
        if not path.is_absolute() and not path.exists():
            pass
        elif path.is_absolute() and not path.exists():
            errors.append(f"File not found: {config.path}")
        fmt = self._detect_format(path, config.format)
        if fmt not in self.LOADERS:
            errors.append(f"Unsupported format: {fmt}")
        return errors

    def refresh(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        """Re-fetch data. For files, this is the same as load (re-read from disk)."""
        return self.load(config, base_dir)

    def _resolve_path(self, path: str | None, base_dir: str | None) -> Path:
        if path is None:
            raise ValueError("Source path is required")
        p = Path(path)
        if not p.is_absolute() and base_dir:
            p = Path(base_dir) / p
        if not p.exists():
            raise FileNotFoundError(f"Source file not found: {p}")
        return p

    def _detect_format(self, path: Path, configured_format: str) -> str:
        if configured_format != "auto":
            return configured_format
        suffix = Path(path).suffix.lower()
        fmt = self.FORMAT_EXTENSIONS.get(suffix)
        if fmt is None:
            raise ValueError(f"Cannot auto-detect format for extension: {suffix}")
        return fmt
