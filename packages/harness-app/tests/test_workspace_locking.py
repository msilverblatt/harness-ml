import json
import multiprocessing
import shutil

import pandas as pd
import pytest

from harness.app.workspace.locking import (
    WorkspaceBusyError,
    WorkspaceLock,
    atomic_write_json,
    atomic_write_text,
    read_lock_owner,
)
from harness.app.workspace.manager import WorkspaceManager
from harness.app.workspace.versions import VersionMeta
from harness.ml.runners.backtest import BacktestResult


def _hold_workspace_lock(root, ready, release):
    with WorkspaceLock(root, "child_process"):
        ready.set()
        release.wait(timeout=10)


def _create_v001(workspace):
    workspace.versions.create_version(
        VersionMeta(id="v001", hypothesis="baseline"), workspace.config
    )
    workspace.versions.set_current("v001", workspace.config)


def test_concurrent_mutation_fails_with_owner_diagnostics(initialized_workspace):
    workspace = initialized_workspace
    competing = WorkspaceManager(workspace._root, lock_timeout=0)

    with WorkspaceLock(workspace._root, "long_training"):
        owner = read_lock_owner(workspace._root)
        assert owner["operation"] == "long_training"
        with pytest.raises(WorkspaceBusyError, match="long_training"):
            competing.switch_version("v001")

    assert read_lock_owner(workspace._root) is None


def test_lock_is_exclusive_across_processes(initialized_workspace):
    root = initialized_workspace._root
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    process = context.Process(target=_hold_workspace_lock, args=(root, ready, release))
    process.start()
    try:
        assert ready.wait(timeout=10)
        with pytest.raises(WorkspaceBusyError, match="child_process"):
            with initialized_workspace._mutation("parent_process"):
                pass
    finally:
        release.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0


def test_stale_owner_metadata_is_replaced(initialized_workspace):
    root = initialized_workspace._root
    atomic_write_json(
        root / ".harness" / "workspace-lock.json",
        {"pid": 999999, "operation": "stale"},
    )

    with WorkspaceLock(root, "current"):
        assert read_lock_owner(root)["operation"] == "current"

    assert read_lock_owner(root) is None


def test_version_is_invisible_until_run_artifacts_are_complete(
    initialized_workspace, monkeypatch
):
    workspace = initialized_workspace
    root = workspace._root
    pd.DataFrame({"x": [0, 1], "target": [0, 1]}).to_parquet(
        root / "data" / "clean" / "dataset.parquet", index=False
    )
    monkeypatch.setattr(
        "harness.app.workspace.manager.run_backtest",
        lambda **kwargs: BacktestResult(metrics={"accuracy": 1.0}),
    )
    original = workspace._write_run_results
    observed = []

    def inspect_then_write(version_id, result, eval_report=None, run_dir=None):
        observed.append((root / "versions" / version_id).exists())
        return original(version_id, result, eval_report, run_dir)

    monkeypatch.setattr(workspace, "_write_run_results", inspect_then_write)
    workspace.run_experiment(
        "baseline",
        "visibility test",
        {"models": {"lr": {"model_type": "logistic"}}},
    )

    assert observed == [False]
    assert (root / "versions" / "v001" / "run" / "state.json").exists()


def test_mutation_removes_abandoned_staging(initialized_workspace):
    workspace = initialized_workspace
    root = workspace._root
    experiment_staging = root / ".experiment-abandoned"
    version_staging = root / "versions" / ".v999.abandoned.tmp"
    experiment_staging.mkdir()
    version_staging.mkdir()

    with workspace._mutation("test_recovery"):
        assert not experiment_staging.exists()
        assert not version_staging.exists()


def _create_candidate_copy(root, source="v001", candidate="v002"):
    source_dir = root / "versions" / source
    candidate_dir = root / "versions" / candidate
    shutil.copytree(source_dir, candidate_dir)
    metadata = candidate_dir / "meta.yaml"
    metadata.write_text(
        metadata.read_text().replace(f"id: {source}", f"id: {candidate}")
    )
    return candidate_dir


def test_recovery_rolls_back_published_candidate_before_pointer_update(
    initialized_workspace,
):
    workspace = initialized_workspace
    root = workspace._root
    _create_v001(workspace)
    _create_candidate_copy(root)
    workspace.config.snapshot_config(workspace._rollback_config_dir)
    atomic_write_json(
        workspace._transaction_path,
        {"candidate": "v002", "previous": "v001"},
    )
    atomic_write_text(root / "current", "v001")

    with workspace._mutation("recover"):
        assert workspace.versions.get_current() == "v001"
        assert not (root / "versions" / "v002").exists()
        assert not workspace._transaction_path.exists()


def test_recovery_finishes_commit_when_pointer_already_updated(initialized_workspace):
    workspace = initialized_workspace
    root = workspace._root
    _create_v001(workspace)
    _create_candidate_copy(root)
    atomic_write_json(
        workspace._transaction_path,
        {"candidate": "v002", "previous": "v001"},
    )
    atomic_write_text(root / "current", "v002")

    with workspace._mutation("recover"):
        assert workspace.versions.get_current() == "v002"
        assert (root / "versions" / "v002").exists()
        assert not workspace._transaction_path.exists()


def test_invalid_transaction_journal_fails_closed(initialized_workspace):
    workspace = initialized_workspace
    workspace._transaction_path.write_text(json.dumps({"unexpected": True}))

    with pytest.raises(RuntimeError, match="Invalid workspace transaction journal"):
        with workspace._mutation("recover"):
            pass
