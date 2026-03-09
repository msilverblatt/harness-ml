"""Tests for NotebookStore — JSONL-backed notebook storage."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from harnessml.core.runner.notebook.schema import EntryType, NotebookEntry
from harnessml.core.runner.notebook.store import NotebookStore


@pytest.fixture()
def store(tmp_path: Path) -> NotebookStore:
    return NotebookStore(tmp_path)


# ── write ──────────────────────────────────────────────────────────────

def test_write_creates_file_and_returns_entry(store: NotebookStore) -> None:
    entry = store.write(type="theory", content="XGBoost should dominate")
    assert isinstance(entry, NotebookEntry)
    assert entry.id == "nb-001"
    assert entry.type == EntryType.THEORY
    assert entry.content == "XGBoost should dominate"
    assert store._jsonl_path.exists()


def test_write_auto_increments_id(store: NotebookStore) -> None:
    store.write(type="finding", content="First finding")
    store.write(type="finding", content="Second finding")
    e3 = store.write(type="note", content="Third entry")
    assert e3.id == "nb-003"


def test_write_auto_tags(tmp_path: Path) -> None:
    # Set up a project with a model so auto_tag can find it
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        yaml.dump({"models": {"xgb_main": {"type": "xgboost"}}})
    )
    s = NotebookStore(tmp_path)
    entry = s.write(type="finding", content="xgb_main performs well on holdout")
    assert "model:xgb_main" in entry.auto_tags


def test_write_with_experiment_id(store: NotebookStore) -> None:
    entry = store.write(
        type="finding",
        content="Saw improvement",
        experiment_id="exp-007",
    )
    assert entry.experiment_id == "exp-007"


# ── read / read_all ───────────────────────────────────────────────────

def test_read_all_excludes_struck(store: NotebookStore) -> None:
    store.write(type="note", content="Keep me")
    e2 = store.write(type="note", content="Strike me")
    store.strike(e2.id, reason="wrong")
    entries = store.read_all()
    assert len(entries) == 1
    assert entries[0].content == "Keep me"


def test_read_all_including_struck(store: NotebookStore) -> None:
    store.write(type="note", content="Keep me")
    e2 = store.write(type="note", content="Strike me")
    store.strike(e2.id, reason="wrong")
    entries = store.read_all(include_struck=True)
    assert len(entries) == 2


def test_read_by_type(store: NotebookStore) -> None:
    store.write(type="theory", content="A theory")
    store.write(type="finding", content="A finding")
    store.write(type="theory", content="Another theory")
    entries = store.read(type="theory")
    assert len(entries) == 2
    assert all(e.type == EntryType.THEORY for e in entries)


def test_read_by_tags(store: NotebookStore) -> None:
    store.write(type="note", content="Has tag", tags=["calibration"])
    store.write(type="note", content="No tag")
    entries = store.read(tags=["calibration"])
    assert len(entries) == 1
    assert entries[0].content == "Has tag"


def test_read_by_tags_matches_auto_tags(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.yaml").write_text(
        yaml.dump({"models": {"lgbm": {"type": "lightgbm"}}})
    )
    s = NotebookStore(tmp_path)
    s.write(type="finding", content="lgbm has low variance")
    s.write(type="note", content="Unrelated note")
    entries = s.read(tags=["model:lgbm"])
    assert len(entries) == 1
    assert entries[0].content == "lgbm has low variance"


def test_read_pagination(store: NotebookStore) -> None:
    for i in range(15):
        store.write(type="note", content=f"Entry {i}")
    page1 = store.read(page=1, per_page=10)
    page2 = store.read(page=2, per_page=10)
    assert len(page1) == 10
    assert len(page2) == 5


def test_read_newest_first(store: NotebookStore) -> None:
    store.write(type="note", content="First")
    store.write(type="note", content="Second")
    store.write(type="note", content="Third")
    entries = store.read()
    assert entries[0].content == "Third"
    assert entries[-1].content == "First"


# ── search ─────────────────────────────────────────────────────────────

def test_search_full_text(store: NotebookStore) -> None:
    store.write(type="note", content="The model had overfitting issues")
    store.write(type="note", content="Calibration looked great")
    results = store.search("overfitting")
    assert len(results) == 1
    assert "overfitting" in results[0].content


def test_search_excludes_struck(store: NotebookStore) -> None:
    e = store.write(type="note", content="Will be struck but matches overfitting")
    store.strike(e.id, reason="wrong")
    results = store.search("overfitting")
    assert len(results) == 0


def test_search_case_insensitive(store: NotebookStore) -> None:
    store.write(type="note", content="XGBoost rules")
    results = store.search("xgboost")
    assert len(results) == 1


# ── strike ─────────────────────────────────────────────────────────────

def test_strike_marks_entry(store: NotebookStore) -> None:
    e = store.write(type="theory", content="Bad theory")
    struck = store.strike(e.id, reason="Disproved by exp-003")
    assert struck.struck is True
    assert struck.struck_reason == "Disproved by exp-003"
    assert struck.struck_at is not None


def test_strike_nonexistent_raises(store: NotebookStore) -> None:
    with pytest.raises(ValueError, match="not found"):
        store.strike("nb-999", reason="nope")


def test_strike_already_struck_raises(store: NotebookStore) -> None:
    e = store.write(type="note", content="Once")
    store.strike(e.id, reason="first")
    with pytest.raises(ValueError, match="already struck"):
        store.strike(e.id, reason="second")


# ── summary ────────────────────────────────────────────────────────────

def test_summary(store: NotebookStore) -> None:
    store.write(type="theory", content="Theory one")
    store.write(type="plan", content="Plan one")
    store.write(type="theory", content="Theory two — latest")
    store.write(type="plan", content="Plan two — latest")
    for i in range(6):
        store.write(type="finding", content=f"Finding {i}")
    summary = store.summary()
    assert summary["latest_theory"] == "Theory two — latest"
    assert summary["latest_plan"] == "Plan two — latest"
    assert len(summary["recent_findings"]) == 5
    assert summary["total_entries"] == 10


def test_summary_empty_notebook(store: NotebookStore) -> None:
    summary = store.summary()
    assert summary["latest_theory"] is None
    assert summary["latest_plan"] is None
    assert summary["recent_findings"] == []
    assert summary["total_entries"] == 0
    assert summary["struck_entries"] == 0
    assert summary["entity_index"] == {}


def test_summary_entity_index(store: NotebookStore) -> None:
    store.write(type="note", content="Stuff", tags=["calibration", "xgboost"])
    store.write(type="note", content="More stuff", tags=["calibration"])
    summary = store.summary()
    assert summary["entity_index"]["calibration"] == 2
    assert summary["entity_index"]["xgboost"] == 1


# ── validation ─────────────────────────────────────────────────────────

def test_invalid_type_rejected(store: NotebookStore) -> None:
    with pytest.raises(ValueError):
        store.write(type="invalid_type", content="Nope")
