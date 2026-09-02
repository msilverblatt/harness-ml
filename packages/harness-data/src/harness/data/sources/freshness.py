"""Freshness tracking — detect when sources are stale."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from harness.data.io import atomic_write_text

FREQUENCY_DELTAS = {
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "monthly": timedelta(days=30),
    "yearly": timedelta(days=365),
}


class FreshnessTracker:
    def __init__(self, state_file: str | Path):
        self._path = Path(state_file)
        self._state: dict[str, dict] = {}
        self._load()

    def record_fetch(self, source_name: str, row_count: int = 0) -> None:
        self._state[source_name] = {
            "last_fetched": datetime.now(UTC).isoformat(),
            "row_count": row_count,
        }
        self._save()

    def is_stale(self, source_name: str, refresh_frequency: str) -> bool:
        if refresh_frequency == "manual":
            return False
        info = self._state.get(source_name)
        if info is None:
            return True
        delta = FREQUENCY_DELTAS.get(refresh_frequency)
        if delta is None:
            return False
        last = datetime.fromisoformat(info["last_fetched"])
        if last.tzinfo is None:  # Compatibility with pre-v2.1 freshness state.
            last = last.replace(tzinfo=UTC)
        return datetime.now(UTC) - last > delta

    def get_info(self, source_name: str) -> dict | None:
        return self._state.get(source_name)

    def check_all(self, sources: list[tuple[str, str]]) -> list[dict]:
        return [
            {"name": name, "frequency": freq, **self._state.get(name, {})}
            for name, freq in sources
            if self.is_stale(name, freq)
        ]

    def _load(self) -> None:
        if self._path.exists():
            self._state = json.loads(self._path.read_text())

    def _save(self) -> None:
        atomic_write_text(self._path, json.dumps(self._state, indent=2))
