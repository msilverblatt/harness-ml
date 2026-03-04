"""Tests for pipeline command execution."""

import sys

import pytest

from easyml.core.guardrails.execution import run_pipeline_command


def test_run_pipeline_command_success(tmp_path):
    result = run_pipeline_command(
        ["echo", "hello"],
        tool_name="test",
        log_dir=tmp_path,
    )
    assert result["status"] == "success"
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]
    assert result["log_path"].exists()
    assert result["duration_s"] >= 0


def test_run_pipeline_command_failure(tmp_path):
    result = run_pipeline_command(
        ["false"],
        tool_name="test",
        log_dir=tmp_path,
    )
    assert result["status"] == "error"
    assert result["returncode"] != 0


def test_run_pipeline_command_timeout(tmp_path):
    result = run_pipeline_command(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        tool_name="test",
        timeout=1,
        log_dir=tmp_path,
    )
    assert result["status"] == "timeout"
    assert result["returncode"] == -1
    assert result["log_path"].exists()


def test_run_pipeline_command_stderr(tmp_path):
    result = run_pipeline_command(
        [sys.executable, "-c", "import sys; sys.stderr.write('oops\\n'); sys.exit(1)"],
        tool_name="test",
        log_dir=tmp_path,
    )
    assert result["status"] == "error"
    assert "oops" in result["stderr"]


def test_log_file_contains_command(tmp_path):
    result = run_pipeline_command(
        ["echo", "hello"],
        tool_name="test",
        log_dir=tmp_path,
    )
    content = result["log_path"].read_text()
    assert "echo hello" in content
    assert "Duration:" in content
    assert "Return code: 0" in content


def test_run_pipeline_command_with_cwd(tmp_path):
    result = run_pipeline_command(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        tool_name="test",
        log_dir=tmp_path,
        cwd=tmp_path,
    )
    assert result["status"] == "success"
    assert str(tmp_path) in result["stdout"]


def test_log_dir_created_automatically(tmp_path):
    log_dir = tmp_path / "nested" / "logs"
    result = run_pipeline_command(
        ["echo", "hi"],
        tool_name="test",
        log_dir=log_dir,
    )
    assert result["status"] == "success"
    assert log_dir.exists()
    assert result["log_path"].exists()
