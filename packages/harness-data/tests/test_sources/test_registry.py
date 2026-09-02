import pytest
import yaml
from pathlib import Path
from harness.data.sources.registry import SourceRegistry
from harness.data.sources.protocol import SourceConfig

class TestSourceRegistry:
    def test_add_and_get(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        config = SourceConfig(name="games", source_type="file", path="data/raw/games.csv")
        registry.add(config)
        result = registry.get("games")
        assert result is not None
        assert result.name == "games"
        assert result.path == "data/raw/games.csv"

    def test_add_duplicate_raises(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        config = SourceConfig(name="games", source_type="file", path="data/raw/games.csv")
        registry.add(config)
        with pytest.raises(ValueError, match="already exists"):
            registry.add(config)

    def test_add_duplicate_with_overwrite(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        config = SourceConfig(name="games", source_type="file", path="data/raw/games.csv")
        registry.add(config)
        config2 = SourceConfig(name="games", source_type="file", path="data/raw/games_v2.csv")
        registry.add(config2, overwrite=True)
        result = registry.get("games")
        assert result.path == "data/raw/games_v2.csv"

    def test_list_all(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        registry.add(SourceConfig(name="a", source_type="file", path="a.csv"))
        registry.add(SourceConfig(name="b", source_type="file", path="b.csv"))
        sources = registry.list_all()
        assert len(sources) == 2
        assert {s.name for s in sources} == {"a", "b"}

    def test_remove(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        registry.add(SourceConfig(name="games", source_type="file", path="a.csv"))
        registry.remove("games")
        assert registry.get("games") is None

    def test_persistence(self, tmp_path):
        registry1 = SourceRegistry(tmp_path)
        registry1.add(SourceConfig(name="games", source_type="file", path="a.csv"))
        registry2 = SourceRegistry(tmp_path)
        result = registry2.get("games")
        assert result is not None
        assert result.name == "games"

    def test_persists_as_yaml(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        registry.add(SourceConfig(name="games", source_type="file", path="a.csv"))
        yaml_path = tmp_path / "sources.yaml"
        assert yaml_path.exists()
        content = yaml.safe_load(yaml_path.read_text())
        assert "games" in content["sources"]
