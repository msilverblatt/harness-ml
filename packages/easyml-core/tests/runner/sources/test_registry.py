"""Tests for SourceRegistry -- CRUD, persistence, and topological ordering."""
from __future__ import annotations

import yaml

from easyml.core.runner.sources.registry import SourceDef, SourceRegistry


def test_add_and_get(tmp_path):
    reg = SourceRegistry(tmp_path)
    src = SourceDef(name="games", source_type="file", path_pattern="data/games.csv")
    result = reg.add(src)
    assert "Added source" in result
    assert reg.get("games") is not None
    assert reg.get("games").source_type == "file"


def test_add_duplicate_returns_error(tmp_path):
    reg = SourceRegistry(tmp_path)
    src = SourceDef(name="games", source_type="file")
    reg.add(src)
    result = reg.add(src)
    assert "Error" in result
    assert "already exists" in result


def test_remove(tmp_path):
    reg = SourceRegistry(tmp_path)
    reg.add(SourceDef(name="games", source_type="file"))
    result = reg.remove("games")
    assert "Removed" in result
    assert reg.get("games") is None


def test_remove_missing_returns_error(tmp_path):
    reg = SourceRegistry(tmp_path)
    result = reg.remove("nonexistent")
    assert "Error" in result


def test_update(tmp_path):
    reg = SourceRegistry(tmp_path)
    reg.add(SourceDef(name="games", source_type="file", path_pattern="v1.csv"))
    result = reg.update(SourceDef(name="games", source_type="file", path_pattern="v2.csv"))
    assert "Updated" in result
    assert reg.get("games").path_pattern == "v2.csv"


def test_update_missing_returns_error(tmp_path):
    reg = SourceRegistry(tmp_path)
    result = reg.update(SourceDef(name="nope", source_type="file"))
    assert "Error" in result


def test_list_all(tmp_path):
    reg = SourceRegistry(tmp_path)
    reg.add(SourceDef(name="a", source_type="file"))
    reg.add(SourceDef(name="b", source_type="url"))
    all_sources = reg.list_all()
    assert len(all_sources) == 2
    names = {s.name for s in all_sources}
    assert names == {"a", "b"}


def test_persistence_roundtrip(tmp_path):
    """Registry data survives save/reload cycle."""
    reg = SourceRegistry(tmp_path)
    reg.add(SourceDef(
        name="rankings",
        source_type="api",
        path_pattern="https://example.com/api",
        refresh_frequency="daily",
        rate_limit=2.0,
        depends_on=["games"],
        schema={"required_columns": ["team_id", "rank"]},
        description="Team rankings from external API",
    ))

    # Reload from disk
    reg2 = SourceRegistry(tmp_path)
    src = reg2.get("rankings")
    assert src is not None
    assert src.source_type == "api"
    assert src.refresh_frequency == "daily"
    assert src.rate_limit == 2.0
    assert src.depends_on == ["games"]
    assert src.schema == {"required_columns": ["team_id", "rank"]}
    assert src.description == "Team rankings from external API"


def test_persistence_omits_empty_defaults(tmp_path):
    """Empty auth, depends_on, schema, etc. are not written to YAML."""
    reg = SourceRegistry(tmp_path)
    reg.add(SourceDef(name="simple", source_type="file", path_pattern="data.csv"))

    raw = yaml.safe_load((tmp_path / "sources.yaml").read_text())
    src_yaml = raw["sources"]["simple"]
    assert "auth" not in src_yaml
    assert "depends_on" not in src_yaml
    assert "schema" not in src_yaml
    assert "leakage_notes" not in src_yaml
    assert "description" not in src_yaml


def test_topological_order_no_deps(tmp_path):
    reg = SourceRegistry(tmp_path)
    reg.add(SourceDef(name="a", source_type="file"))
    reg.add(SourceDef(name="b", source_type="file"))
    order = reg.topological_order()
    assert set(order) == {"a", "b"}


def test_topological_order_with_deps(tmp_path):
    """Dependencies come before dependents."""
    reg = SourceRegistry(tmp_path)
    reg.add(SourceDef(name="raw_games", source_type="file"))
    reg.add(SourceDef(name="features", source_type="computed", depends_on=["raw_games"]))
    reg.add(SourceDef(name="model_input", source_type="computed", depends_on=["features"]))

    order = reg.topological_order()
    assert order.index("raw_games") < order.index("features")
    assert order.index("features") < order.index("model_input")


def test_topological_order_diamond(tmp_path):
    """Diamond dependency: A -> B, A -> C, B -> D, C -> D."""
    reg = SourceRegistry(tmp_path)
    reg.add(SourceDef(name="A", source_type="file"))
    reg.add(SourceDef(name="B", source_type="computed", depends_on=["A"]))
    reg.add(SourceDef(name="C", source_type="computed", depends_on=["A"]))
    reg.add(SourceDef(name="D", source_type="computed", depends_on=["B", "C"]))

    order = reg.topological_order()
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")
