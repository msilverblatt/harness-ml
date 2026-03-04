"""Tests for run management."""
from __future__ import annotations

from pathlib import Path

import pytest

from easyml.core.runner.run_manager import RunManager


# -----------------------------------------------------------------------
# Tests: new_run
# -----------------------------------------------------------------------

class TestNewRun:
    """new_run creates timestamped directories."""

    def test_creates_directory(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        run_dir = mgr.new_run()
        assert run_dir.exists()
        assert run_dir.is_dir()

    def test_creates_subdirectories(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        run_dir = mgr.new_run()
        assert (run_dir / "predictions").exists()
        assert (run_dir / "diagnostics").exists()

    def test_custom_run_id(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        run_dir = mgr.new_run(run_id="my_custom_run")
        assert run_dir.name == "my_custom_run"
        assert run_dir.exists()

    def test_timestamp_format(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        run_dir = mgr.new_run()
        # Should be YYYYMMDD_HHMMSS format (14 chars + underscore = 15)
        name = run_dir.name
        assert len(name) == 15
        assert name[8] == "_"

    def test_multiple_runs(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        run_1 = mgr.new_run(run_id="20240101_000001")
        run_2 = mgr.new_run(run_id="20240101_000002")
        assert run_1 != run_2
        assert run_1.exists()
        assert run_2.exists()

    def test_creates_parent_dirs(self, tmp_path):
        mgr = RunManager(tmp_path / "deep" / "nested" / "outputs")
        run_dir = mgr.new_run(run_id="test_run")
        assert run_dir.exists()


# -----------------------------------------------------------------------
# Tests: promote
# -----------------------------------------------------------------------

class TestPromote:
    """promote creates 'current' symlink."""

    def test_creates_symlink(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        mgr.new_run(run_id="20240101_120000")
        mgr.promote("20240101_120000")

        current = tmp_path / "outputs" / "current"
        assert current.is_symlink()

    def test_symlink_resolves_to_run(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        run_dir = mgr.new_run(run_id="20240101_120000")
        mgr.promote("20240101_120000")

        current = mgr.get_current()
        assert current is not None
        assert current.resolve() == run_dir.resolve()

    def test_promote_replaces_existing(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        mgr.new_run(run_id="run_1")
        mgr.new_run(run_id="run_2")

        mgr.promote("run_1")
        mgr.promote("run_2")

        current = mgr.get_current()
        assert current is not None
        assert current.name == "run_2"

    def test_promote_nonexistent_raises(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        (tmp_path / "outputs").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            mgr.promote("nonexistent_run")


# -----------------------------------------------------------------------
# Tests: list_runs
# -----------------------------------------------------------------------

class TestListRuns:
    """list_runs returns sorted run entries."""

    def test_empty_dir(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        assert mgr.list_runs() == []

    def test_nonexistent_dir(self, tmp_path):
        mgr = RunManager(tmp_path / "nonexistent")
        assert mgr.list_runs() == []

    def test_lists_runs_sorted(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        mgr.new_run(run_id="20240101_000001")
        mgr.new_run(run_id="20240102_000002")
        mgr.new_run(run_id="20240103_000003")

        runs = mgr.list_runs()
        assert len(runs) == 3
        # Sorted newest first (reverse)
        assert runs[0]["run_id"] == "20240103_000003"
        assert runs[2]["run_id"] == "20240101_000001"

    def test_marks_current(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        mgr.new_run(run_id="run_a")
        mgr.new_run(run_id="run_b")
        mgr.promote("run_a")

        runs = mgr.list_runs()
        current_runs = [r for r in runs if r["is_current"]]
        assert len(current_runs) == 1
        assert current_runs[0]["run_id"] == "run_a"

    def test_excludes_current_symlink(self, tmp_path):
        """'current' symlink is not listed as a run."""
        mgr = RunManager(tmp_path / "outputs")
        mgr.new_run(run_id="run_a")
        mgr.promote("run_a")

        runs = mgr.list_runs()
        run_ids = [r["run_id"] for r in runs]
        assert "current" not in run_ids

    def test_run_has_path(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        mgr.new_run(run_id="test_run")

        runs = mgr.list_runs()
        assert len(runs) == 1
        assert "path" in runs[0]
        assert Path(runs[0]["path"]).exists()


# -----------------------------------------------------------------------
# Tests: get_current
# -----------------------------------------------------------------------

class TestGetCurrent:
    """get_current returns promoted run path."""

    def test_no_current(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        (tmp_path / "outputs").mkdir(parents=True)
        assert mgr.get_current() is None

    def test_returns_promoted_path(self, tmp_path):
        mgr = RunManager(tmp_path / "outputs")
        run_dir = mgr.new_run(run_id="promoted_run")
        mgr.promote("promoted_run")

        current = mgr.get_current()
        assert current is not None
        assert current.resolve() == run_dir.resolve()

    def test_nonexistent_outputs_dir(self, tmp_path):
        mgr = RunManager(tmp_path / "nonexistent")
        assert mgr.get_current() is None
