"""Feature resolver — maps model config feature names to DataFrame columns."""
from __future__ import annotations

from easyml.features.registry import FeatureRegistry


class FeatureResolver:
    """Resolves feature names / categories into concrete column lists.

    Works against a :class:`FeatureRegistry` and validates that requested
    columns actually exist in the available data.
    """

    def __init__(self, *, registry: FeatureRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Explicit column resolution
    # ------------------------------------------------------------------

    def resolve(
        self,
        feature_names: list[str],
        available_columns: list[str],
    ) -> list[str]:
        """Validate that each name in *feature_names* exists in *available_columns*.

        Returns the validated list (preserving order).  Raises ``ValueError``
        if any requested column is not found.
        """
        available_set = set(available_columns)
        result: list[str] = []
        for name in feature_names:
            if name not in available_set:
                raise ValueError(
                    f"Column '{name}' not found in available columns"
                )
            result.append(name)
        return result

    # ------------------------------------------------------------------
    # Category-based resolution
    # ------------------------------------------------------------------

    def resolve_category(
        self,
        category: str,
        available_columns: list[str],
    ) -> list[str]:
        """Return all output columns for features in *category* that are available."""
        available_set = set(available_columns)
        features = self._registry.list_features(category=category)
        result: list[str] = []
        for meta in features:
            for col in meta.output_columns:
                if col in available_set:
                    result.append(col)
        return result

    # ------------------------------------------------------------------
    # All registered columns
    # ------------------------------------------------------------------

    def resolve_all(
        self,
        available_columns: list[str],
    ) -> list[str]:
        """Return all registered output columns that are present in *available_columns*."""
        available_set = set(available_columns)
        all_features = self._registry.list_features()
        result: list[str] = []
        seen: set[str] = set()
        for meta in all_features:
            for col in meta.output_columns:
                if col in available_set and col not in seen:
                    result.append(col)
                    seen.add(col)
        return result
