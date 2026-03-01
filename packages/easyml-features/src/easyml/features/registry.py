"""Feature registry — decorator-based registration with source hashing."""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable

from easyml.schemas.core import FeatureMeta, TemporalFilter


@dataclass
class _FeatureDefinition:
    """Internal storage for a registered feature."""

    meta: FeatureMeta
    fn: Callable
    source_code: str


class FeatureRegistry:
    """Registry that maps feature names to compute functions + metadata.

    Features are registered via the ``@registry.register(...)`` decorator.
    """

    def __init__(self) -> None:
        self._features: dict[str, _FeatureDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        name: str,
        category: str,
        level: str,
        output_columns: list[str],
        temporal_filter: TemporalFilter | None = None,
        tainted_columns: list[str] | None = None,
        nan_strategy: str = "median",
    ) -> Callable:
        """Decorator factory that registers a feature compute function."""

        def decorator(fn: Callable) -> Callable:
            if name in self._features:
                raise ValueError(
                    f"Feature '{name}' already registered"
                )

            meta = FeatureMeta(
                name=name,
                category=category,
                level=level,
                output_columns=output_columns,
                temporal_filter=temporal_filter,
                tainted_columns=tainted_columns or [],
                nan_strategy=nan_strategy,
            )

            source_code = inspect.getsource(fn)
            self._features[name] = _FeatureDefinition(
                meta=meta, fn=fn, source_code=source_code
            )
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_metadata(self, name: str) -> FeatureMeta:
        """Return metadata for a registered feature, or raise KeyError."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry")
        return self._features[name].meta

    def get_fn(self, name: str) -> Callable:
        """Return the compute function for a registered feature."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry")
        return self._features[name].fn

    def source_hash(self, name: str) -> str:
        """Return the SHA-256 hex digest of the feature's source code."""
        if name not in self._features:
            raise KeyError(f"Feature '{name}' not found in registry")
        return hashlib.sha256(
            self._features[name].source_code.encode()
        ).hexdigest()

    # ------------------------------------------------------------------
    # Listing / iteration
    # ------------------------------------------------------------------

    def list_features(
        self,
        *,
        category: str | None = None,
        level: str | None = None,
    ) -> list[FeatureMeta]:
        """Return metadata for all features, optionally filtered."""
        results = []
        for defn in self._features.values():
            if category is not None and defn.meta.category != category:
                continue
            if level is not None and defn.meta.level != level:
                continue
            results.append(defn.meta)
        return results

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._features

    def __len__(self) -> int:
        return len(self._features)
