"""Tests for stage guards with staleness detection."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from easyml.data.guards import GuardrailViolationError, StageGuard


def test_guard_passes(tmp_path):
    data_file = tmp_path / "features.parquet"
    pd.DataFrame({"x": range(1000)}).to_parquet(data_file)
    guard = StageGuard(
        name="features_exist", requires=[str(data_file)], min_rows=500
    )
    guard.check()  # should not raise


def test_guard_fails_missing_file(tmp_path):
    guard = StageGuard(
        name="test", requires=[str(tmp_path / "nonexistent.parquet")]
    )
    with pytest.raises(GuardrailViolationError):
        guard.check()


def test_guard_fails_min_rows(tmp_path):
    data_file = tmp_path / "features.parquet"
    pd.DataFrame({"x": range(10)}).to_parquet(data_file)
    guard = StageGuard(
        name="test", requires=[str(data_file)], min_rows=500
    )
    with pytest.raises(GuardrailViolationError, match="rows"):
        guard.check()


def test_guard_stale_artifact(tmp_path):
    data_file = tmp_path / "features.parquet"
    data_file.write_bytes(b"old")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("new")
    os.utime(data_file, (1000, 1000))  # make data_file old

    guard = StageGuard(
        name="test",
        requires=[str(data_file)],
        stale_if_older_than=[str(config_file)],
    )
    with pytest.raises(GuardrailViolationError, match="stale"):
        guard.check()
