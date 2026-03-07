"""Tests for sports hook registration including COMPETITION_NARRATIVE."""

from __future__ import annotations

from harnessml.core.runner.hooks import (
    COMPETITION_NARRATIVE,
    HookRegistry,
)


class TestCompetitionNarrativeHook:
    """Verify COMPETITION_NARRATIVE constant and hook registration."""

    def setup_method(self):
        HookRegistry.clear()

    def teardown_method(self):
        HookRegistry.clear()

    def test_constant_exists(self):
        assert COMPETITION_NARRATIVE == "competition_narrative"

    def test_hook_registered_after_import(self):
        """After calling sports register(), the hook should be present."""
        from harnessml.sports.hooks import register

        register()
        hooks = HookRegistry.get(COMPETITION_NARRATIVE)
        assert len(hooks) >= 1

    def test_default_hook_returns_none(self):
        """The default competition narrative hook returns None."""
        from harnessml.sports.hooks import register

        register()
        result = HookRegistry.call_first(COMPETITION_NARRATIVE)
        assert result is None


class TestSportsColumnHooks:
    """Verify existing column hooks still register correctly."""

    def setup_method(self):
        HookRegistry.clear()

    def teardown_method(self):
        HookRegistry.clear()

    def test_column_candidates_registered(self):
        from harnessml.core.runner.hooks import COLUMN_CANDIDATES
        from harnessml.sports.hooks import register

        register()
        hooks = HookRegistry.get(COLUMN_CANDIDATES)
        assert len(hooks) >= 1

    def test_column_renames_registered(self):
        from harnessml.core.runner.hooks import COLUMN_RENAMES
        from harnessml.sports.hooks import register

        register()
        hooks = HookRegistry.get(COLUMN_RENAMES)
        assert len(hooks) >= 1
