"""Tests for structured JSON audit log."""

import json
from datetime import datetime, timedelta, timezone

from harnessml.core.guardrails.audit import AuditLogger


def test_audit_log_records_invocation(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train_models",
        args={"run_id": None},
        guardrails_passed=True,
        result_status="success",
        duration_s=342.1,
    )
    lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["tool"] == "train_models"
    assert entry["guardrails_passed"] is True
    assert entry["result_status"] == "success"
    assert entry["duration_s"] == 342.1
    assert "timestamp" in entry


def test_audit_log_append_only(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train", args={}, guardrails_passed=True,
        result_status="success", duration_s=1.0,
    )
    logger.log_invocation(
        tool="predict", args={}, guardrails_passed=True,
        result_status="success", duration_s=2.0,
    )
    lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    assert len(lines) == 2


def test_audit_log_with_error(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train", args={}, guardrails_passed=False,
        result_status="error", duration_s=0.5,
        error="Sanity check failed",
    )
    entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
    assert entry["error"] == "Sanity check failed"
    assert entry["guardrails_passed"] is False


def test_audit_log_human_override(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train", args={}, guardrails_passed=True,
        result_status="success", duration_s=1.0,
        human_override=True,
    )
    entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
    assert entry["human_override"] is True


def test_audit_log_no_error_key_when_none(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train", args={}, guardrails_passed=True,
        result_status="success", duration_s=1.0,
    )
    entry = json.loads((tmp_path / "audit.jsonl").read_text().strip())
    assert "error" not in entry


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_audit_log_query_by_tool(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train", args={}, guardrails_passed=True,
        result_status="success", duration_s=1.0,
    )
    logger.log_invocation(
        tool="predict", args={}, guardrails_passed=True,
        result_status="error", duration_s=2.0,
    )
    results = logger.query(tool="train")
    assert len(results) == 1
    assert results[0]["tool"] == "train"


def test_audit_log_query_by_status(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train", args={}, guardrails_passed=True,
        result_status="success", duration_s=1.0,
    )
    logger.log_invocation(
        tool="predict", args={}, guardrails_passed=True,
        result_status="error", duration_s=2.0,
    )
    results = logger.query(status="error")
    assert len(results) == 1
    assert results[0]["tool"] == "predict"


def test_audit_log_query_since(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train", args={}, guardrails_passed=True,
        result_status="success", duration_s=1.0,
    )
    # Query with a time in the past — should return the entry
    past = datetime.now(timezone.utc) - timedelta(hours=1)
    results = logger.query(since=past)
    assert len(results) == 1

    # Query with a time in the future — should return nothing
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    results = logger.query(since=future)
    assert len(results) == 0


def test_audit_log_query_empty(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    results = logger.query()
    assert results == []


def test_audit_log_query_no_filters(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train", args={}, guardrails_passed=True,
        result_status="success", duration_s=1.0,
    )
    logger.log_invocation(
        tool="predict", args={}, guardrails_passed=True,
        result_status="success", duration_s=2.0,
    )
    results = logger.query()
    assert len(results) == 2


def test_audit_log_creates_parent_dirs(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "nested" / "dir" / "audit.jsonl")
    logger.log_invocation(
        tool="train", args={}, guardrails_passed=True,
        result_status="success", duration_s=1.0,
    )
    assert (tmp_path / "nested" / "dir" / "audit.jsonl").exists()
