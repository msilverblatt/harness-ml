import pytest
from datetime import datetime, timedelta
from pathlib import Path
from harness.data.sources.freshness import FreshnessTracker

class TestFreshnessTracker:
    def test_record_and_get(self, tmp_path):
        tracker = FreshnessTracker(tmp_path / "freshness.json")
        tracker.record_fetch("games", row_count=500)
        info = tracker.get_info("games")
        assert info is not None
        assert info["row_count"] == 500
        assert "last_fetched" in info

    def test_is_stale_manual(self, tmp_path):
        tracker = FreshnessTracker(tmp_path / "freshness.json")
        tracker.record_fetch("games")
        assert not tracker.is_stale("games", "manual")

    def test_is_stale_daily_fresh(self, tmp_path):
        tracker = FreshnessTracker(tmp_path / "freshness.json")
        tracker.record_fetch("games")
        assert not tracker.is_stale("games", "daily")

    def test_is_stale_unknown_source(self, tmp_path):
        tracker = FreshnessTracker(tmp_path / "freshness.json")
        assert tracker.is_stale("unknown", "daily")

    def test_persistence(self, tmp_path):
        path = tmp_path / "freshness.json"
        tracker1 = FreshnessTracker(path)
        tracker1.record_fetch("games", row_count=100)
        tracker2 = FreshnessTracker(path)
        info = tracker2.get_info("games")
        assert info is not None
        assert info["row_count"] == 100
