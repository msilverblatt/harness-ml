"""Tests for the notebook MCP handler."""
from __future__ import annotations

import json

import pytest
import yaml
from harnessml.plugin.handlers.notebook import ACTIONS, dispatch


@pytest.fixture()
def project_dir(tmp_path):
    """Create a minimal project directory with config/models.yaml."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(yaml.dump({"models": {}}))
    return str(tmp_path)


# ── write ────────────────────────────────────────────────────────────


def test_write_action(project_dir):
    result = dispatch("write", type="theory", content="XGBoost should outperform LightGBM on this dataset", project_dir=project_dir)
    assert "nb-001" in result
    assert "theory" in result
    assert "XGBoost" in result


def test_write_requires_content(project_dir):
    result = dispatch("write", type="theory", content=None, project_dir=project_dir)
    assert "**Error**" in result
    assert "content" in result


def test_write_requires_valid_type(project_dir):
    result = dispatch("write", type="garbage", content="something", project_dir=project_dir)
    assert "**Error**" in result
    assert "type" in result


# ── read ─────────────────────────────────────────────────────────────


def test_read_action(project_dir):
    dispatch("write", type="finding", content="Feature X is highly correlated", project_dir=project_dir)
    result = dispatch("read", project_dir=project_dir)
    assert "nb-001" in result
    assert "Feature X" in result


def test_read_filtered_by_type(project_dir):
    dispatch("write", type="finding", content="A finding entry", project_dir=project_dir)
    dispatch("write", type="theory", content="A theory entry", project_dir=project_dir)
    result = dispatch("read", type="theory", project_dir=project_dir)
    assert "theory" in result.lower()
    assert "A theory entry" in result
    # The finding should NOT appear
    assert "A finding entry" not in result


def test_read_filtered_by_tags(project_dir):
    dispatch("write", type="note", content="Tagged note", tags='["model:xgb"]', project_dir=project_dir)
    dispatch("write", type="note", content="Untagged note", project_dir=project_dir)
    result = dispatch("read", tags='["model:xgb"]', project_dir=project_dir)
    assert "Tagged note" in result
    assert "Untagged note" not in result


# ── search ───────────────────────────────────────────────────────────


def test_search_action(project_dir):
    dispatch("write", type="research", content="Investigated rolling averages for momentum features", project_dir=project_dir)
    result = dispatch("search", query="momentum", project_dir=project_dir)
    assert "momentum" in result


def test_search_requires_query(project_dir):
    result = dispatch("search", query=None, project_dir=project_dir)
    assert "**Error**" in result
    assert "query" in result


# ── strike ───────────────────────────────────────────────────────────


def test_strike_action(project_dir):
    dispatch("write", type="theory", content="Theory to invalidate", project_dir=project_dir)
    result = dispatch("strike", entry_id="nb-001", reason="Proven wrong by experiment 5", project_dir=project_dir)
    assert "nb-001" in result
    assert "struck" in result.lower() or "~~" in result
    # Struck entry should not appear in reads
    read_result = dispatch("read", project_dir=project_dir)
    assert "Theory to invalidate" not in read_result


def test_strike_requires_entry_id(project_dir):
    result = dispatch("strike", entry_id=None, reason="some reason", project_dir=project_dir)
    assert "**Error**" in result
    assert "entry_id" in result


def test_strike_requires_reason(project_dir):
    result = dispatch("strike", entry_id="nb-001", reason=None, project_dir=project_dir)
    assert "**Error**" in result
    assert "reason" in result


# ── summary ──────────────────────────────────────────────────────────


def test_summary_action(project_dir):
    dispatch("write", type="theory", content="Main working theory", project_dir=project_dir)
    dispatch("write", type="plan", content="Next steps plan", project_dir=project_dir)
    dispatch("write", type="finding", content="Key finding 1", project_dir=project_dir)
    result = dispatch("summary", project_dir=project_dir)
    assert "Main working theory" in result
    assert "Next steps plan" in result
    assert "Key finding 1" in result


def test_summary_empty_notebook(project_dir):
    result = dispatch("summary", project_dir=project_dir)
    # Should return successfully even with no entries
    assert "**Error**" not in result
