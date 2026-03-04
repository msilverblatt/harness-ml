"""Freshness tracking for data sources."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Map refresh_frequency to timedelta
_FREQUENCY_MAP: dict[str, timedelta | None] = {
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "monthly": timedelta(days=30),
    "yearly": timedelta(days=365),
    "manual": None,  # Never auto-stale
}


class FreshnessTracker:
    """Track last-fetched timestamps and detect stale sources."""

    def __init__(self, state_file: Path):
        self._state_file = Path(state_file)
        self._state: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._state_file.exists():
            self._state = json.loads(self._state_file.read_text())

    def _save(self) -> None:
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_file.write_text(
            json.dumps(self._state, indent=2, default=str)
        )

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_fetch(self, source_name: str, row_count: int = 0) -> None:
        """Record that a source was fetched right now."""
        self._state[source_name] = {
            "last_fetched": datetime.now(timezone.utc).isoformat(),
            "row_count": row_count,
        }
        self._save()

    # ------------------------------------------------------------------
    # Staleness checks
    # ------------------------------------------------------------------

    def is_stale(self, source_name: str, refresh_frequency: str) -> bool:
        """Check if a source is stale based on its refresh frequency."""
        if refresh_frequency == "manual":
            return False
        info = self._state.get(source_name)
        if not info:
            return True  # Never fetched
        max_age = _FREQUENCY_MAP.get(refresh_frequency)
        if max_age is None:
            return False
        last_str = info["last_fetched"]
        last = datetime.fromisoformat(last_str)
        # Ensure timezone-aware comparison
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - last > max_age

    def check_all(self, sources: list) -> list[dict[str, Any]]:
        """Check freshness of all sources. Returns list of stale source info dicts."""
        stale = []
        for src in sources:
            if self.is_stale(src.name, src.refresh_frequency):
                info = self._state.get(src.name, {})
                stale.append({
                    "name": src.name,
                    "refresh_frequency": src.refresh_frequency,
                    "last_fetched": info.get("last_fetched", "never"),
                })
        return stale

    def get_info(self, source_name: str) -> dict[str, Any] | None:
        """Return stored info for a source, or None if never fetched."""
        return self._state.get(source_name)
