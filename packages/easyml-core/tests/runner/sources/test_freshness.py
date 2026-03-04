"""Tests for FreshnessTracker -- recording, staleness, persistence."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from easyml.core.runner.sources.freshness import FreshnessTracker
from easyml.core.runner.sources.registry import SourceDef


def test_record_fetch(tmp_path):
    state_file = tmp_path / "state.json"
    tracker = FreshnessTracker(state_file)
    tracker.record_fetch("games", row_count=1000)

    info = tracker.get_info("games")
    assert info is not None
    assert info["row_count"] == 1000
    assert "last_fetched" in info


def test_never_fetched_is_stale(tmp_path):
    tracker = FreshnessTracker(tmp_path / "state.json")
    assert tracker.is_stale("games", "daily") is True


def test_manual_never_stale(tmp_path):
    tracker = FreshnessTracker(tmp_path / "state.json")
    # Even if never fetched, manual sources are never stale
    assert tracker.is_stale("games", "manual") is False


def test_recently_fetched_not_stale(tmp_path):
    state_file = tmp_path / "state.json"
    tracker = FreshnessTracker(state_file)
    tracker.record_fetch("games")
    assert tracker.is_stale("games", "daily") is False
    assert tracker.is_stale("games", "hourly") is False


def test_old_fetch_is_stale(tmp_path):
    state_file = tmp_path / "state.json"
    # Write a state with an old timestamp
    old_time = datetime.now(timezone.utc) - timedelta(hours=25)
    state_file.write_text(json.dumps({
        "games": {
            "last_fetched": old_time.isoformat(),
            "row_count": 500,
        }
    }))
    tracker = FreshnessTracker(state_file)
    assert tracker.is_stale("games", "daily") is True
    assert tracker.is_stale("games", "weekly") is False


def test_check_all(tmp_path):
    state_file = tmp_path / "state.json"
    tracker = FreshnessTracker(state_file)
    tracker.record_fetch("games", row_count=100)

    sources = [
        SourceDef(name="games", source_type="file", refresh_frequency="daily"),
        SourceDef(name="rankings", source_type="api", refresh_frequency="daily"),
    ]
    stale = tracker.check_all(sources)
    # games was just fetched, rankings was never fetched
    assert len(stale) == 1
    assert stale[0]["name"] == "rankings"
    assert stale[0]["last_fetched"] == "never"


def test_persistence_roundtrip(tmp_path):
    state_file = tmp_path / "state.json"
    tracker = FreshnessTracker(state_file)
    tracker.record_fetch("games", row_count=42)

    # Reload
    tracker2 = FreshnessTracker(state_file)
    info = tracker2.get_info("games")
    assert info is not None
    assert info["row_count"] == 42


def test_get_info_missing_returns_none(tmp_path):
    tracker = FreshnessTracker(tmp_path / "state.json")
    assert tracker.get_info("nonexistent") is None


def test_state_file_created_on_first_write(tmp_path):
    state_file = tmp_path / "subdir" / "state.json"
    assert not state_file.exists()
    tracker = FreshnessTracker(state_file)
    tracker.record_fetch("test")
    assert state_file.exists()
