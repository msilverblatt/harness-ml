"""Source protocol — the contract all source adapters implement."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    """Metadata about a loaded source."""

    name: str
    source_type: str  # "file", "url", "api"
    row_count: int | None = None
    columns: list[str] = Field(default_factory=list)
    column_types: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_dataframe(cls, name: str, source_type: str, df: pd.DataFrame) -> SourceMetadata:
        """Create metadata from a loaded DataFrame."""
        return cls(
            name=name,
            source_type=source_type,
            row_count=len(df),
            columns=list(df.columns),
            column_types={col: str(dtype) for col, dtype in df.dtypes.items()},
        )


class SourceConfig(BaseModel):
    """Configuration for a data source."""

    name: str
    source_type: str = "file"  # "file", "url", "api"
    path: str | None = None
    url: str | None = None
    format: str = "auto"  # "csv", "parquet", "excel", "json", "auto"
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


@runtime_checkable
class Source(Protocol):
    """Protocol that all source adapters implement."""

    def load(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        """Load data from the source and return a DataFrame."""
        ...

    def validate(self, config: SourceConfig) -> list[str]:
        """Validate source config. Returns list of error messages (empty = valid)."""
        ...

    def refresh(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        """Re-fetch data from the source. Default: delegates to load()."""
        ...
