"""Fingerprint-based cache invalidation for trained models.

A fingerprint captures the config hash, data hash, and data size so that
a cached model can be skipped when nothing has changed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Fingerprint:
    """Immutable fingerprint for a trained model artifact."""

    config_hash: str
    data_hash: str
    data_size: float

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def compute(
        cls,
        config_dict: dict,
        data_hash: str,
        data_size: float,
    ) -> Fingerprint:
        """Build a fingerprint from a config dict, data hash, and data size.

        The config dict is serialised deterministically (sorted keys) and
        hashed with SHA-256, truncated to 16 hex characters.
        """
        canonical = json.dumps(config_dict, sort_keys=True, default=str)
        cfg_hash = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        return cls(config_hash=cfg_hash, data_hash=data_hash, data_size=data_size)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write fingerprint as JSON to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "data_size": self.data_size,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def matches(self, path: str | Path) -> bool:
        """Return ``True`` if the fingerprint at *path* equals this one.

        Returns ``False`` when the file is missing or any field differs.
        """
        path = Path(path)
        if not path.exists():
            return False
        try:
            stored = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return False

        return (
            stored.get("config_hash") == self.config_hash
            and stored.get("data_hash") == self.data_hash
            and stored.get("data_size") == self.data_size
        )
