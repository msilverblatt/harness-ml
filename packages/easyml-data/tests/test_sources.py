"""Tests for data source registry."""
from __future__ import annotations

from easyml.data.sources import SourceRegistry


def test_register_source():
    sources = SourceRegistry()

    @sources.register(
        name="test_source",
        category="external",
        outputs=["data/external/test/"],
        temporal_safety="pre_tournament",
        leakage_notes="Safe — pre-tournament snapshot",
    )
    def scrape(output_dir, config):
        pass

    assert "test_source" in sources
    meta = sources.get_metadata("test_source")
    assert meta.temporal_safety == "pre_tournament"


def test_list_sources():
    sources = SourceRegistry()
    for name, cat in [("a", "external"), ("b", "external"), ("c", "internal")]:

        @sources.register(
            name=name,
            category=cat,
            outputs=[f"data/{name}/"],
            temporal_safety="unknown",
        )
        def scrape(output_dir, config):
            pass

    assert len(sources.list_sources()) == 3
    assert len(sources.list_sources(category="external")) == 2


def test_freshness_check(tmp_path):
    sources = SourceRegistry()
    data_file = tmp_path / "data.csv"
    data_file.write_text("a,b\n1,2")

    @sources.register(
        name="test",
        category="external",
        outputs=[str(tmp_path)],
        temporal_safety="pre_tournament",
    )
    def scrape(output_dir, config):
        pass

    result = sources.check_freshness("test")
    assert result["exists"] is True
    assert "age_hours" in result


def test_run_source(tmp_path):
    sources = SourceRegistry()
    marker = {"called": False}

    @sources.register(
        name="test",
        category="external",
        outputs=[str(tmp_path)],
        temporal_safety="unknown",
    )
    def scrape(output_dir, config):
        marker["called"] = True
        (output_dir / "result.txt").write_text("done")

    sources.run("test", output_dir=tmp_path, config={})
    assert marker["called"]
    assert (tmp_path / "result.txt").exists()
