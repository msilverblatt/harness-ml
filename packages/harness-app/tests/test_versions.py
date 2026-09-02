import pytest
from pathlib import Path

from harness.app.workspace.versions import VersionTree, VersionMeta
from harness.app.workspace.config import ConfigManager
from harness.ml.config.project import ProjectConfig


@pytest.fixture
def version_env(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "versions").mkdir()
    config_mgr = ConfigManager(ws)
    config_mgr.write_project(ProjectConfig(task_type="binary"))
    tree = VersionTree(ws)
    return tree, config_mgr, ws


class TestCreateAndGet:
    def test_create_version(self, version_env):
        tree, config_mgr, ws = version_env
        meta = VersionMeta(id="v001", experiment_type="baseline", hypothesis="test")
        result = tree.create_version(meta, config_mgr)
        assert result == "v001"
        assert (ws / "versions" / "v001" / "meta.yaml").exists()
        assert (ws / "versions" / "v001" / "config").is_dir()
        assert (ws / "versions" / "v001" / "run").is_dir()

    def test_get_version(self, version_env):
        tree, config_mgr, _ = version_env
        meta = VersionMeta(id="v001", experiment_type="baseline", hypothesis="first try")
        tree.create_version(meta, config_mgr)
        loaded = tree.get_version("v001")
        assert loaded is not None
        assert loaded.id == "v001"
        assert loaded.hypothesis == "first try"

    def test_get_nonexistent(self, version_env):
        tree, _, _ = version_env
        assert tree.get_version("v999") is None


class TestList:
    def test_list_versions(self, version_env):
        tree, config_mgr, _ = version_env
        tree.create_version(VersionMeta(id="v001", experiment_type="baseline"), config_mgr)
        tree.create_version(VersionMeta(id="v002", experiment_type="model"), config_mgr)
        versions = tree.list_versions()
        assert len(versions) == 2
        assert versions[0].id == "v001"
        assert versions[1].id == "v002"

    def test_list_empty(self, tmp_path):
        ws = tmp_path / "empty_ws"
        ws.mkdir()
        tree = VersionTree(ws)
        assert tree.list_versions() == []


class TestNextVersionId:
    def test_first_version(self, version_env):
        tree, _, _ = version_env
        assert tree.next_version_id() == "v001"

    def test_increments(self, version_env):
        tree, config_mgr, _ = version_env
        tree.create_version(VersionMeta(id="v001"), config_mgr)
        tree.create_version(VersionMeta(id="v002"), config_mgr)
        assert tree.next_version_id() == "v003"


class TestSetCurrent:
    def test_set_and_get_current(self, version_env):
        tree, config_mgr, ws = version_env
        tree.create_version(VersionMeta(id="v001"), config_mgr)
        tree.set_current("v001", config_mgr)
        assert tree.get_current() == "v001"

    def test_get_current_none(self, version_env):
        tree, _, _ = version_env
        assert tree.get_current() is None

    def test_set_nonexistent_raises(self, version_env):
        tree, config_mgr, _ = version_env
        with pytest.raises(ValueError, match="Version not found"):
            tree.set_current("v999", config_mgr)


class TestCompare:
    def test_compare_metrics(self, version_env):
        tree, config_mgr, _ = version_env
        tree.create_version(
            VersionMeta(id="v001", metrics={"accuracy": 0.80, "brier": 0.20}),
            config_mgr,
        )
        tree.create_version(
            VersionMeta(id="v002", metrics={"accuracy": 0.85, "brier": 0.18}),
            config_mgr,
        )
        deltas = tree.compare("v001", "v002")
        assert deltas["accuracy"]["delta"] == pytest.approx(0.05)
        assert deltas["brier"]["delta"] == pytest.approx(-0.02)

    def test_compare_nonexistent_raises(self, version_env):
        tree, config_mgr, _ = version_env
        tree.create_version(VersionMeta(id="v001"), config_mgr)
        with pytest.raises(ValueError):
            tree.compare("v001", "v999")


class TestAncestry:
    def test_ancestry_chain(self, version_env):
        tree, config_mgr, _ = version_env
        tree.create_version(VersionMeta(id="v001"), config_mgr)
        tree.create_version(VersionMeta(id="v002", parent="v001"), config_mgr)
        tree.create_version(VersionMeta(id="v003", parent="v002"), config_mgr)
        chain = tree.ancestry("v003")
        assert len(chain) == 3
        assert [v.id for v in chain] == ["v001", "v002", "v003"]

    def test_ancestry_single(self, version_env):
        tree, config_mgr, _ = version_env
        tree.create_version(VersionMeta(id="v001"), config_mgr)
        chain = tree.ancestry("v001")
        assert len(chain) == 1
        assert chain[0].id == "v001"


class TestUpdateVersion:
    def test_update_conclusion(self, version_env):
        tree, config_mgr, _ = version_env
        tree.create_version(VersionMeta(id="v001", hypothesis="test"), config_mgr)
        tree.update_version("v001", conclusion="it worked", verdict="keep")
        meta = tree.get_version("v001")
        assert meta.conclusion == "it worked"
        assert meta.verdict == "keep"

    def test_update_nonexistent_raises(self, version_env):
        tree, _, _ = version_env
        with pytest.raises(ValueError, match="Version not found"):
            tree.update_version("v999", conclusion="nope")
