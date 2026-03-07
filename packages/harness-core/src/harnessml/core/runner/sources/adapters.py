"""Built-in data source adapters."""
from __future__ import annotations

import io
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


class FileAdapter:
    """Load data from local CSV/parquet/Excel files."""

    @staticmethod
    def load(path_pattern: str, **kwargs: Any) -> pd.DataFrame:
        path = Path(path_pattern)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path, **kwargs)
        elif suffix in (".parquet", ".pq"):
            return pd.read_parquet(path, **kwargs)
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")


class UrlAdapter:
    """Download data from a URL."""

    @staticmethod
    def load(
        url: str,
        format: str = "csv",
        auth_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        req = urllib.request.Request(url)
        if auth_headers:
            for key, value in auth_headers.items():
                req.add_header(key, value)

        with urllib.request.urlopen(req) as response:
            data = response.read()
            buf = io.BytesIO(data)
            if format == "csv":
                return pd.read_csv(buf, **kwargs)
            elif format == "json":
                return pd.read_json(buf, **kwargs)
            elif format == "parquet":
                return pd.read_parquet(buf, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format}")


class ApiAdapter:
    """Fetch data from a REST API with pagination and rate limiting."""

    @staticmethod
    def load(
        url: str,
        rate_limit: float = 0.0,
        auth_headers: dict[str, str] | None = None,
        pagination: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        all_data: list[dict] = []
        current_url: str | None = url

        while current_url:
            if rate_limit > 0:
                time.sleep(1.0 / rate_limit)

            req = urllib.request.Request(current_url)
            if auth_headers:
                for key, value in auth_headers.items():
                    req.add_header(key, value)

            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read())

            # Extract records
            if isinstance(data, list):
                all_data.extend(data)
                current_url = None
            elif isinstance(data, dict):
                records_key = (pagination or {}).get("records_key", "data")
                all_data.extend(data.get(records_key, []))
                next_key = (pagination or {}).get("next_key", "next")
                current_url = data.get(next_key)
            else:
                break

        return pd.DataFrame(all_data)


class ComputedAdapter:
    """Derive data by running a Python function on other sources."""

    @staticmethod
    def load(
        fn: Callable[[dict[str, pd.DataFrame]], pd.DataFrame],
        sources: dict[str, pd.DataFrame],
        **kwargs: Any,
    ) -> pd.DataFrame:
        return fn(sources, **kwargs)


# Adapter dispatch
ADAPTERS: dict[str, type] = {
    "file": FileAdapter,
    "url": UrlAdapter,
    "api": ApiAdapter,
    "computed": ComputedAdapter,
}
