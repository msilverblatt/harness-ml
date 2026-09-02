import pytest
import tempfile
from pathlib import Path

from harness.ml.config.loader import ConfigLoader
from harness.ml.config.project import ProjectConfig, CVConfig
from harness.ml.config.models import ModelsConfig
from harness.ml.config.ensemble import EnsembleConfig


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestConfigLoaderProject:
    def test_load_defaults_from_empty_file(self, tmp_dir):
        path = tmp_dir / "project.yaml"
        path.write_text("{}\n")
        cfg = ConfigLoader.load_project(path)
        assert isinstance(cfg, ProjectConfig)
        assert cfg.task_type == "binary"
        assert cfg.target_column == "target"

    def test_load_project_full(self, tmp_dir):
        path = tmp_dir / "project.yaml"
        path.write_text(
            "task_type: regression\n"
            "target_column: score\n"
            "metrics:\n"
            "  - rmse\n"
            "  - mae\n"
            "cv:\n"
            "  strategy: stratified_kfold\n"
            "  n_folds: 3\n"
        )
        cfg = ConfigLoader.load_project(path)
        assert cfg.task_type == "regression"
        assert cfg.target_column == "score"
        assert cfg.metrics == ["rmse", "mae"]
        assert cfg.cv.strategy == "stratified_kfold"
        assert cfg.cv.n_folds == 3


class TestConfigLoaderModels:
    def test_load_models_from_yaml(self, tmp_dir):
        path = tmp_dir / "models.yaml"
        path.write_text(
            "models:\n"
            "  baseline:\n"
            "    model_type: logistic\n"
            "    features:\n"
            "      - x1\n"
            "      - x2\n"
            "  xgb:\n"
            "    model_type: xgboost\n"
            "    params:\n"
            "      n_estimators: 100\n"
        )
        cfg = ConfigLoader.load_models(path)
        assert isinstance(cfg, ModelsConfig)
        assert "baseline" in cfg.models
        assert cfg.models["baseline"].model_type == "logistic"
        assert cfg.models["baseline"].features == ["x1", "x2"]
        assert cfg.models["xgb"].params == {"n_estimators": 100}

    def test_load_models_empty(self, tmp_dir):
        path = tmp_dir / "models.yaml"
        path.write_text("{}\n")
        cfg = ConfigLoader.load_models(path)
        assert cfg.models == {}


class TestConfigLoaderEnsemble:
    def test_load_ensemble_from_yaml(self, tmp_dir):
        path = tmp_dir / "ensemble.yaml"
        path.write_text(
            "ensemble:\n"
            "  method: stacked\n"
            "  meta_learner_type: ridge\n"
            "  calibration: isotonic\n"
            "  temperature: 0.9\n"
        )
        cfg = ConfigLoader.load_ensemble(path)
        assert isinstance(cfg, EnsembleConfig)
        assert cfg.method == "stacked"
        assert cfg.meta_learner_type == "ridge"
        assert cfg.calibration == "isotonic"
        assert cfg.temperature == 0.9

    def test_load_ensemble_defaults(self, tmp_dir):
        path = tmp_dir / "ensemble.yaml"
        path.write_text("{}\n")
        cfg = ConfigLoader.load_ensemble(path)
        assert cfg.method == "stacked"
        assert cfg.calibration == "none"
