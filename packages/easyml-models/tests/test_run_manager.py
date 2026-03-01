"""Tests for RunManager."""
import pytest
from pathlib import Path

from easyml.models.run_manager import RunManager


def test_run_manager_lifecycle(tmp_path):
    rm = RunManager(base_dir=tmp_path)

    run1 = rm.new_run(run_id="20260101_000000")
    assert run1.exists()
    assert rm.get_latest() is None  # not promoted yet

    rm.promote(run1.name)
    assert rm.get_latest() == run1

    run2 = rm.new_run(run_id="20260102_000000")
    rm.promote(run2.name)
    assert rm.get_latest() == run2
    assert len(rm.list_runs()) == 2


def test_run_manager_manifest(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    run = rm.new_run()
    rm.write_manifest(run, stage="train", labels=["test"])
    manifest = rm.read_manifest(run)
    assert manifest["stage"] == "train"
    assert manifest["labels"] == ["test"]


def test_run_manager_explicit_id(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    run = rm.new_run(run_id="20260101_120000")
    assert run.name == "20260101_120000"
    assert run.exists()


def test_run_manager_promote_nonexistent(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        rm.promote("nonexistent_run")


def test_run_manager_get_latest_empty(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    assert rm.get_latest() is None


def test_run_manager_list_empty(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    assert rm.list_runs() == []


def test_run_manager_symlink(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    run = rm.new_run(run_id="20260101_000000")
    rm.promote(run.name)

    symlink = tmp_path / "latest"
    assert symlink.is_symlink()
    assert symlink.resolve() == run.resolve()


def test_run_manager_manifest_update(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    run = rm.new_run()
    rm.write_manifest(run, stage="train")
    rm.write_manifest(run, stage="predict", extra="data")

    manifest = rm.read_manifest(run)
    assert manifest["stage"] == "predict"
    assert manifest["extra"] == "data"


def test_run_manager_list_runs_sorted(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    rm.new_run(run_id="20260103_000000")
    rm.new_run(run_id="20260101_000000")
    rm.new_run(run_id="20260102_000000")

    runs = rm.list_runs()
    run_ids = [r["run_id"] for r in runs]
    assert run_ids == ["20260101_000000", "20260102_000000", "20260103_000000"]


def test_run_manager_list_runs_with_labels(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    run = rm.new_run(run_id="20260101_000000")
    rm.promote(run.name)

    runs = rm.list_runs()
    assert len(runs) == 1
    assert "current" in runs[0]["labels"]
