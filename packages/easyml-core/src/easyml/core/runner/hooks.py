"""Extension hook registry for domain-specific plugins.

Plugins (e.g., easyml-sports) register implementations via these hooks.
Core provides defaults for each hook.
"""
from __future__ import annotations

from typing import Any, Callable


class HookRegistry:
    """Central registry for plugin hooks."""

    _hooks: dict[str, list[Callable]] = {}

    @classmethod
    def register(cls, hook_name: str, fn: Callable) -> None:
        """Register a hook function under the given name."""
        cls._hooks.setdefault(hook_name, []).append(fn)

    @classmethod
    def get(cls, hook_name: str) -> list[Callable]:
        """Return all registered hooks for the given name."""
        return cls._hooks.get(hook_name, [])

    @classmethod
    def call_first(cls, hook_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call first registered hook, return result. Returns None if no hooks."""
        hooks = cls.get(hook_name)
        if hooks:
            return hooks[0](*args, **kwargs)
        return None

    @classmethod
    def call_all(cls, hook_name: str, *args: Any, **kwargs: Any) -> list:
        """Call all registered hooks, return list of results."""
        return [fn(*args, **kwargs) for fn in cls.get(hook_name)]

    @classmethod
    def clear(cls) -> None:
        """Clear all hooks (for testing)."""
        cls._hooks.clear()


# Hook names
FEATURE_EXPANSION = "feature_expansion"
PROVIDER_INJECTION = "provider_injection"
PRE_TRAINING = "pre_training"
POST_PREDICTION = "post_prediction"
FEATURE_TYPE = "feature_type"
COLUMN_CANDIDATES = "column_candidates"
COLUMN_RENAMES = "column_renames"

# Default column candidates for entity identification (domain-agnostic)
_DEFAULT_A_CANDIDATES = ["entity_a", "group_a"]
_DEFAULT_B_CANDIDATES = ["entity_b", "group_b"]

# Default label/outcome column candidates
_DEFAULT_LABEL_CANDIDATES = ["result"]
_DEFAULT_MARGIN_CANDIDATES = ["margin"]

# Default ID-like column patterns (for data profiler skip logic)
_DEFAULT_ID_PATTERNS = ["entity_a", "entity_b", "game_id", "matchup_id"]


def get_entity_column_candidates() -> tuple[list[str], list[str]]:
    """Get column name candidates for entity A/B, including plugin-registered ones."""
    extra = HookRegistry.call_all(COLUMN_CANDIDATES)
    a_candidates = list(_DEFAULT_A_CANDIDATES)
    b_candidates = list(_DEFAULT_B_CANDIDATES)
    for result in extra:
        if isinstance(result, dict):
            a_candidates.extend(result.get("a", []))
            b_candidates.extend(result.get("b", []))
    return a_candidates, b_candidates


def get_label_candidates() -> list[str]:
    """Get label column candidates, including plugin-registered ones."""
    extra = HookRegistry.call_all(COLUMN_CANDIDATES)
    candidates = list(_DEFAULT_LABEL_CANDIDATES)
    for result in extra:
        if isinstance(result, dict):
            candidates.extend(result.get("label", []))
    return candidates


def get_margin_candidates() -> list[str]:
    """Get margin column candidates, including plugin-registered ones."""
    extra = HookRegistry.call_all(COLUMN_CANDIDATES)
    candidates = list(_DEFAULT_MARGIN_CANDIDATES)
    for result in extra:
        if isinstance(result, dict):
            candidates.extend(result.get("margin", []))
    return candidates


def get_id_patterns() -> list[str]:
    """Get ID-like column patterns, including plugin-registered ones."""
    extra = HookRegistry.call_all(COLUMN_CANDIDATES)
    patterns = list(_DEFAULT_ID_PATTERNS)
    for result in extra:
        if isinstance(result, dict):
            patterns.extend(result.get("id_patterns", []))
    return patterns


def get_column_renames() -> dict[str, str]:
    """Get column rename mappings, including plugin-registered ones.

    Returns a dict of {old_name: new_name} for normalizing column names.
    Core provides generic renames (e.g. Season -> season), plugins add
    domain-specific ones (e.g. TeamAWon -> result).
    """
    renames = {"Season": "season"}  # Default core rename
    extra = HookRegistry.call_all(COLUMN_RENAMES)
    for result in extra:
        if isinstance(result, dict):
            renames.update(result)
    return renames
