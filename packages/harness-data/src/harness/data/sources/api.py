"""API source adapter — fetches JSON data from a REST endpoint."""

from __future__ import annotations

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

from harness.data.sources.protocol import SourceConfig


class ApiSource:
    """Load data from a REST API that returns JSON."""

    def load(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        if not config.url:
            raise ValueError("url is required for API sources")

        # Separate the records_key from actual HTTP query params
        raw_params: dict = dict(config.params)
        records_key: str | None = raw_params.pop("records_key", None)

        resp = requests.get(config.url, params=raw_params)
        resp.raise_for_status()
        data = resp.json()

        if records_key is not None:
            data = data[records_key]

        return pd.DataFrame(data)

    def validate(self, config: SourceConfig) -> list[str]:
        errors: list[str] = []
        if not config.url:
            errors.append("url is required for API sources")
        return errors

    def refresh(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        return self.load(config, base_dir)
