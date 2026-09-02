"""URL source adapter — fetches CSV, JSON, or Parquet from HTTP."""

from __future__ import annotations

import io
from pathlib import PurePosixPath

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

from harness.data.sources.protocol import SourceConfig

_FORMAT_EXTENSIONS: dict[str, str] = {
    ".csv": "csv",
    ".tsv": "csv",
    ".json": "json",
    ".parquet": "parquet",
    ".pq": "parquet",
}


class UrlSource:
    """Load data from a remote URL (CSV, JSON, Parquet)."""

    def load(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        if not config.url:
            raise ValueError("url is required for URL sources")
        resp = requests.get(config.url)
        resp.raise_for_status()
        fmt = self._detect_format(config.url, config.format)
        return self._parse(resp.content, fmt)

    def validate(self, config: SourceConfig) -> list[str]:
        errors: list[str] = []
        if not config.url:
            errors.append("url is required for URL sources")
        return errors

    def refresh(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        return self.load(config, base_dir)

    def _detect_format(self, url: str, configured_format: str) -> str:
        if configured_format != "auto":
            return configured_format
        suffix = PurePosixPath(url.split("?")[0]).suffix.lower()
        fmt = _FORMAT_EXTENSIONS.get(suffix)
        if fmt is None:
            raise ValueError(f"Unsupported or unrecognisable format for URL extension: '{suffix}'")
        return fmt

    def _parse(self, content: bytes, fmt: str) -> pd.DataFrame:
        buf = io.BytesIO(content)
        if fmt == "csv":
            return pd.read_csv(buf)
        if fmt == "json":
            return pd.read_json(buf)
        if fmt == "parquet":
            return pd.read_parquet(buf)
        raise ValueError(f"Unsupported format: {fmt}")
