"""Tests for the failure-tolerant refresh orchestrator."""
from __future__ import annotations

from easyml.data.sources import SourceRegistry
from easyml.data.refresh import RefreshOrchestrator


def test_refresh_all(tmp_path):
    sources = SourceRegistry()
    results = {"a": False, "b": False}

    @sources.register(
        name="a",
        category="ext",
        outputs=[str(tmp_path / "a")],
        temporal_safety="unknown",
    )
    def scrape_a(output_dir, config):
        results["a"] = True

    @sources.register(
        name="b",
        category="ext",
        outputs=[str(tmp_path / "b")],
        temporal_safety="unknown",
    )
    def scrape_b(output_dir, config):
        results["b"] = True

    orch = RefreshOrchestrator(sources=sources)
    report = orch.refresh_all(config={})
    assert results["a"] is True
    assert results["b"] is True
    assert report["a"]["status"] == "success"


def test_refresh_failure_tolerant(tmp_path):
    sources = SourceRegistry()

    @sources.register(
        name="ok",
        category="ext",
        outputs=[str(tmp_path)],
        temporal_safety="unknown",
    )
    def scrape_ok(output_dir, config):
        pass

    @sources.register(
        name="fail",
        category="ext",
        outputs=[str(tmp_path)],
        temporal_safety="unknown",
    )
    def scrape_fail(output_dir, config):
        raise RuntimeError("scraper down")

    orch = RefreshOrchestrator(sources=sources)
    report = orch.refresh_all(config={})
    assert report["ok"]["status"] == "success"
    assert report["fail"]["status"] == "error"
    assert "scraper down" in report["fail"]["error"]
