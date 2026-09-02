import shutil
import yaml
from pathlib import Path
from harness.ml.config.project import ProjectConfig
from harness.ml.config.models import ModelsConfig, SingleModelConfig
from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.features.schema import FeatureSet


class ConfigManager:
    def __init__(self, workspace_dir: Path):
        self._root = Path(workspace_dir)
        self._config_dir = self._root / "config"

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    def ensure_dir(self):
        self._config_dir.mkdir(parents=True, exist_ok=True)

    def read_project(self) -> ProjectConfig:
        path = self._config_dir / "project.yaml"
        if not path.exists():
            return ProjectConfig()
        return ProjectConfig(**yaml.safe_load(path.read_text()) or {})

    def write_project(self, config: ProjectConfig):
        self.ensure_dir()
        (self._config_dir / "project.yaml").write_text(
            yaml.dump(config.model_dump(exclude_defaults=False), default_flow_style=False, sort_keys=False)
        )

    def read_models(self) -> ModelsConfig:
        path = self._config_dir / "models.yaml"
        if not path.exists():
            return ModelsConfig()
        content = yaml.safe_load(path.read_text()) or {}
        return ModelsConfig.from_yaml_dict(content.get("models", {}))

    def write_models(self, config: ModelsConfig):
        self.ensure_dir()
        models_dict = {}
        for name, m in config.models.items():
            d = m.model_dump(exclude_defaults=True)
            d.pop("name", None)
            models_dict[name] = d
        (self._config_dir / "models.yaml").write_text(
            yaml.dump({"models": models_dict}, default_flow_style=False, sort_keys=False)
        )

    def read_ensemble(self) -> EnsembleConfig:
        path = self._config_dir / "ensemble.yaml"
        if not path.exists():
            return EnsembleConfig()
        content = yaml.safe_load(path.read_text()) or {}
        return EnsembleConfig(**content.get("ensemble", {}))

    def write_ensemble(self, config: EnsembleConfig):
        self.ensure_dir()
        (self._config_dir / "ensemble.yaml").write_text(
            yaml.dump({"ensemble": config.model_dump(exclude_defaults=False)}, default_flow_style=False, sort_keys=False)
        )

    def read_features(self) -> FeatureSet:
        path = self._config_dir / "features.yaml"
        if not path.exists():
            return FeatureSet()
        content = yaml.safe_load(path.read_text()) or {}
        return FeatureSet.from_yaml_dict(content.get("features", {}))

    def write_features(self, feature_set: FeatureSet):
        self.ensure_dir()
        features_dict = {}
        for name, f in feature_set.features.items():
            d = f.model_dump(exclude_defaults=True, mode="json")
            d.pop("name", None)
            if "feature_type" in d:
                d["type"] = d.pop("feature_type")
            features_dict[name] = d
        (self._config_dir / "features.yaml").write_text(
            yaml.dump({"features": features_dict}, default_flow_style=False, sort_keys=False)
        )

    def read_evals(self) -> dict:
        path = self._config_dir / "evals.yaml"
        if not path.exists():
            return {"evals": {}}
        return yaml.safe_load(path.read_text()) or {"evals": {}}

    def write_evals(self, config: dict):
        self.ensure_dir()
        payload = config if "evals" in config else {"evals": config}
        (self._config_dir / "evals.yaml").write_text(
            yaml.dump(payload, default_flow_style=False, sort_keys=False)
        )

    def snapshot_config(self, dest_dir: Path):
        dest_dir.mkdir(parents=True, exist_ok=True)
        if self._config_dir.exists():
            for f in self._config_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dest_dir / f.name)

    def restore_config(self, source_dir: Path):
        self.ensure_dir()
        for f in self._config_dir.iterdir():
            if f.is_file():
                f.unlink()
        for f in source_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, self._config_dir / f.name)
