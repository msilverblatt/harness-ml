import yaml
from pathlib import Path

from harness.ml.config.project import ProjectConfig
from harness.ml.config.models import ModelsConfig
from harness.ml.config.ensemble import EnsembleConfig


class ConfigLoader:
    @staticmethod
    def load_project(path: Path) -> ProjectConfig:
        content = yaml.safe_load(Path(path).read_text()) or {}
        return ProjectConfig(**content)

    @staticmethod
    def load_models(path: Path) -> ModelsConfig:
        content = yaml.safe_load(Path(path).read_text()) or {}
        return ModelsConfig.from_yaml_dict(content.get("models", {}))

    @staticmethod
    def load_ensemble(path: Path) -> EnsembleConfig:
        content = yaml.safe_load(Path(path).read_text()) or {}
        return EnsembleConfig(**content.get("ensemble", {}))
