import pytest
from pathlib import Path

from harness.app.workspace.config import ConfigManager
from harness.ml.config.project import ProjectConfig
from harness.ml.config.models import ModelsConfig, SingleModelConfig
from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.features.schema import FeatureSet, FeatureDefinition, FeatureType


@pytest.fixture
def config_mgr(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ConfigManager(ws)


class TestProjectConfig:
    def test_roundtrip(self, config_mgr):
        original = ProjectConfig(task_type="binary", target_column="result")
        config_mgr.write_project(original)
        loaded = config_mgr.read_project()
        assert loaded.task_type == "binary"
        assert loaded.target_column == "result"

    def test_defaults_when_missing(self, config_mgr):
        loaded = config_mgr.read_project()
        assert loaded.task_type == "binary"
        assert loaded.target_column == "target"


class TestModelsConfig:
    def test_roundtrip(self, config_mgr):
        models = ModelsConfig()
        models.models["lr"] = SingleModelConfig(name="lr", model_type="logistic")
        models.models["xgb"] = SingleModelConfig(
            name="xgb", model_type="xgboost",
            params={"max_depth": 6, "n_estimators": 100},
        )
        config_mgr.write_models(models)
        loaded = config_mgr.read_models()
        assert "lr" in loaded.models
        assert "xgb" in loaded.models
        assert loaded.models["xgb"].params["max_depth"] == 6

    def test_defaults_when_missing(self, config_mgr):
        loaded = config_mgr.read_models()
        assert len(loaded.models) == 0


class TestEnsembleConfig:
    def test_roundtrip(self, config_mgr):
        original = EnsembleConfig(method="stacked", calibration="isotonic")
        config_mgr.write_ensemble(original)
        loaded = config_mgr.read_ensemble()
        assert loaded.method == "stacked"
        assert loaded.calibration == "isotonic"

    def test_defaults_when_missing(self, config_mgr):
        loaded = config_mgr.read_ensemble()
        assert loaded.method == "stacked"


class TestFeaturesConfig:
    def test_roundtrip(self, config_mgr):
        fs = FeatureSet()
        fs.features["elo"] = FeatureDefinition(
            name="elo", feature_type=FeatureType.ENTITY, source_column="elo_rating"
        )
        fs.features["win_rate"] = FeatureDefinition(
            name="win_rate", feature_type=FeatureType.INSTANCE, source_column="win_pct"
        )
        config_mgr.write_features(fs)
        loaded = config_mgr.read_features()
        assert "elo" in loaded.features
        assert loaded.features["elo"].feature_type == FeatureType.ENTITY
        assert "win_rate" in loaded.features

    def test_defaults_when_missing(self, config_mgr):
        loaded = config_mgr.read_features()
        assert len(loaded.features) == 0


class TestSnapshotRestore:
    def test_snapshot_and_restore(self, config_mgr, tmp_path):
        # Write some config
        config_mgr.write_project(ProjectConfig(task_type="regression", target_column="score"))
        models = ModelsConfig()
        models.models["lr"] = SingleModelConfig(name="lr", model_type="logistic")
        config_mgr.write_models(models)

        # Snapshot
        snap_dir = tmp_path / "snapshot"
        config_mgr.snapshot_config(snap_dir)
        assert (snap_dir / "project.yaml").exists()
        assert (snap_dir / "models.yaml").exists()

        # Modify config
        config_mgr.write_project(ProjectConfig(task_type="binary", target_column="label"))

        # Restore
        config_mgr.restore_config(snap_dir)
        restored = config_mgr.read_project()
        assert restored.task_type == "regression"
        assert restored.target_column == "score"
