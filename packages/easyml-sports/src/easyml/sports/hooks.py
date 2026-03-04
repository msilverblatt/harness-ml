"""Register sports-specific hooks into easyml core."""
from __future__ import annotations

from easyml.core.runner.hooks import (
    COLUMN_CANDIDATES,
    COLUMN_RENAMES,
    HookRegistry,
)


def register() -> None:
    """Register all sports hooks.

    Called automatically when easyml.sports is imported.
    Registers sports-domain column name candidates so that core
    pipeline, reporting, and profiling components can detect
    sports-specific column names (TeamA, TeamB, TeamAWon, etc.).
    """
    HookRegistry.register(COLUMN_CANDIDATES, _sports_column_candidates)
    HookRegistry.register(COLUMN_RENAMES, _sports_column_renames)


def _sports_column_candidates() -> dict[str, list[str]]:
    """Return sports-domain column name candidates."""
    return {
        "a": ["TeamA", "team_a"],
        "b": ["TeamB", "team_b"],
        "label": ["TeamAWon"],
        "margin": ["TeamAMargin"],
        "id_patterns": ["TeamA", "TeamB"],
    }


def _sports_column_renames() -> dict[str, str]:
    """Return sports-domain column renames."""
    return {
        "TeamAWon": "result",
        "TeamAMargin": "margin",
    }
