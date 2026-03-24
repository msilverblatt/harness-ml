from datetime import datetime
from pathlib import Path
import yaml
import pandas as pd

from harness.app.workspace.config import ConfigManager
from harness.app.workspace.versions import VersionTree, VersionMeta
from harness.data.workspace import DataWorkspace
from harness.ml.runners.backtest import run_backtest, BacktestResult
from harness.ml.config.project import ProjectConfig, CVConfig
from harness.ml.config.models import ModelsConfig, SingleModelConfig
from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.features.schema import FeatureSet


class WorkspaceManager:
    def __init__(self, workspace_dir: Path):
        self._root = Path(workspace_dir)
        self.config = ConfigManager(workspace_dir)
        self.versions = VersionTree(workspace_dir)
        self.data = DataWorkspace(workspace_dir)

    @staticmethod
    def init(workspace_dir: Path, task_type: str = "binary", target_column: str = "target") -> "WorkspaceManager":
        root = Path(workspace_dir)
        root.mkdir(parents=True, exist_ok=True)
        # Create harness.yaml marker
        (root / "harness.yaml").write_text(yaml.dump({
            "name": root.name,
            "created": datetime.utcnow().isoformat(),
        }, default_flow_style=False))
        # Init data workspace
        data_ws = DataWorkspace(root)
        data_ws.init()
        # Create config directory with defaults
        ws = WorkspaceManager(root)
        ws.config.write_project(ProjectConfig(task_type=task_type, target_column=target_column))
        ws.config.write_models(ModelsConfig())
        ws.config.write_ensemble(EnsembleConfig())
        ws.config.write_features(FeatureSet())
        # Create versions + artifacts dirs
        (root / "versions").mkdir(exist_ok=True)
        (root / "artifacts").mkdir(exist_ok=True)
        (root / ".harness").mkdir(exist_ok=True)
        return ws

    def run_experiment(
        self, experiment_type: str, hypothesis: str,
        params: dict, parent: str | None = None,
    ) -> BacktestResult:
        """Run an experiment: apply params to config, run backtest, create version."""
        # Resolve parent config
        if parent:
            self.versions.set_current(parent, self.config)

        # Apply experiment params to config
        self._apply_experiment_params(experiment_type, params)

        # Load current config
        project = self.config.read_project()
        models = self.config.read_models()
        ensemble = self.config.read_ensemble()
        features = self.config.read_features()

        # Load data
        data = self.data.load_clean_data()

        # Run backtest
        result = run_backtest(
            data=data,
            project_config=project,
            models_config=models,
            ensemble_config=ensemble,
            feature_set=features if features.features else None,
            cache_dir=self._root / "artifacts" / "predictions",
        )

        # Create version
        version_id = self.versions.next_version_id()
        current = self.versions.get_current()
        meta = VersionMeta(
            id=version_id,
            parent=parent or current,
            experiment_type=experiment_type,
            hypothesis=hypothesis,
            timestamp=datetime.utcnow().isoformat(),
            metrics=result.metrics,
        )
        self.versions.create_version(meta, self.config)

        # Write run results
        self._write_run_results(version_id, result)

        # Set as current
        (self._root / "current").write_text(version_id)

        return result

    def conclude_experiment(self, version_id: str, conclusion: str, verdict: str) -> None:
        self.versions.update_version(version_id, conclusion=conclusion, verdict=verdict)

    def switch_version(self, version_id: str) -> None:
        self.versions.set_current(version_id, self.config)

    def status(self) -> dict:
        current = self.versions.get_current()
        meta = self.versions.get_version(current) if current else None
        models = self.config.read_models()
        return {
            "workspace": self._root.name,
            "current_version": current,
            "metrics": meta.metrics if meta else {},
            "model_count": len(models.models),
            "version_count": len(self.versions.list_versions()),
        }

    def _apply_experiment_params(self, experiment_type: str, params: dict):
        if experiment_type == "baseline":
            # Set up initial models + features
            if "models" in params:
                models = ModelsConfig()
                for name, cfg in params["models"].items():
                    cfg["name"] = name
                    models.models[name] = SingleModelConfig(**cfg)
                self.config.write_models(models)
            if "features" in params:
                from harness.ml.features.schema import FeatureDefinition, FeatureType
                fs = FeatureSet()
                for name, cfg in params["features"].items():
                    cfg["name"] = name
                    if "type" in cfg:
                        cfg["feature_type"] = cfg.pop("type")
                    fs.features[name] = FeatureDefinition(**cfg)
                self.config.write_features(fs)
        elif experiment_type == "model":
            models = self.config.read_models()
            cfg = dict(params.get("model", params))
            name = cfg.pop("name", cfg.get("model_type", "new_model"))
            cfg["name"] = name
            models.models[name] = SingleModelConfig(**cfg)
            self.config.write_models(models)
        elif experiment_type == "feature":
            from harness.ml.features.schema import FeatureDefinition
            features = self.config.read_features()
            cfg = dict(params.get("feature", params))
            name = cfg.pop("name", "new_feature")
            cfg["name"] = name
            if "type" in cfg:
                cfg["feature_type"] = cfg.pop("type")
            features.features[name] = FeatureDefinition(**cfg)
            self.config.write_features(features)
        elif experiment_type == "hyperparameter":
            models = self.config.read_models()
            model_name = params.get("model_name")
            if model_name and model_name in models.models:
                models.models[model_name].params.update(params.get("params", {}))
                self.config.write_models(models)
        elif experiment_type == "ensemble":
            ensemble = self.config.read_ensemble()
            for k, v in params.items():
                if hasattr(ensemble, k):
                    setattr(ensemble, k, v)
            self.config.write_ensemble(ensemble)

    def _write_run_results(self, version_id: str, result: BacktestResult):
        import json
        run_dir = self._root / "versions" / version_id / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
        if result.predictions is not None:
            result.predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        (run_dir / "diagnostics.json").write_text(json.dumps({
            "per_fold_metrics": result.per_fold_metrics,
            "models_trained": result.models_trained,
            "models_cached": result.models_cached,
            "models_failed": result.models_failed,
            "duration_s": result.duration_s,
            "meta_coefficients": result.meta_coefficients,
        }, indent=2))
