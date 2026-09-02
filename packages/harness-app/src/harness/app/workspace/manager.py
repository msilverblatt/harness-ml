from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml
from harness.app.experiments.types import ExperimentType
from harness.app.workspace.config import ConfigManager
from harness.app.workspace.locking import (
    WorkspaceLock,
    atomic_write_json,
    atomic_write_text,
)
from harness.app.workspace.versions import VersionMeta, VersionTree
from harness.data.workspace import DataWorkspace
from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.config.models import ModelsConfig, SingleModelConfig
from harness.ml.config.project import CVConfig, ProjectConfig
from harness.ml.evals.runner import EvalRunner
from harness.ml.features.schema import FeatureDefinition, FeatureSet
from harness.ml.runners.backtest import BacktestResult, run_backtest


class WorkspaceManager:
    def __init__(self, workspace_dir: Path, lock_timeout: float = 0):
        self._root = Path(workspace_dir)
        self._lock_timeout = lock_timeout
        self.config = ConfigManager(workspace_dir)
        self.versions = VersionTree(workspace_dir)
        self.data = DataWorkspace(workspace_dir)

    @staticmethod
    def init(
        workspace_dir: Path,
        task_type: str = "binary",
        target_column: str = "target",
    ) -> WorkspaceManager:
        root = Path(workspace_dir)
        root.mkdir(parents=True, exist_ok=True)
        ws = WorkspaceManager(root)
        with WorkspaceLock(root, "initialize", ws._lock_timeout):
            atomic_write_text(
                root / "harness.yaml",
                yaml.dump(
                    {"name": root.name, "created": _utc_now()},
                    default_flow_style=False,
                ),
            )
            DataWorkspace(root).init()
            ws.config.write_project(
                ProjectConfig(task_type=task_type, target_column=target_column)
            )
            ws.config.write_models(ModelsConfig())
            ws.config.write_ensemble(EnsembleConfig())
            ws.config.write_features(FeatureSet())
            preset = files("harness.ml.evals").joinpath("presets", f"{task_type}.yaml")
            ws.config.write_evals(yaml.safe_load(preset.read_text()) or {"evals": {}})
            (root / "versions").mkdir(exist_ok=True)
            (root / "artifacts").mkdir(exist_ok=True)
        return ws

    def run_experiment(
        self,
        experiment_type: str,
        hypothesis: str,
        params: dict,
        parent: str | None = None,
    ) -> BacktestResult:
        """Apply and run an experiment without mutating live config until success."""
        with self._mutation("run_experiment"):
            return self._run_experiment_locked(
                experiment_type, hypothesis, params, parent
            )

    def _run_experiment_locked(
        self,
        experiment_type: str,
        hypothesis: str,
        params: dict,
        parent: str | None = None,
    ) -> BacktestResult:
        try:
            exp_type = ExperimentType(experiment_type)
        except ValueError as exc:
            supported = ", ".join(item.value for item in ExperimentType)
            raise ValueError(
                f"Unknown experiment type '{experiment_type}'. Supported: {supported}"
            ) from exc
        if not hypothesis.strip():
            raise ValueError("Experiment hypothesis must not be empty")
        if not isinstance(params, dict):
            raise TypeError("Experiment params must be a dictionary")

        current_before = self.versions.get_current()
        parent_id = parent or current_before
        if parent is not None and self.versions.get_version(parent) is None:
            raise ValueError(f"Parent version not found: {parent}")
        if exp_type is ExperimentType.BASELINE and parent_id is not None:
            raise ValueError("A baseline experiment cannot have a parent")
        if exp_type is not ExperimentType.BASELINE and parent_id is None:
            raise ValueError("Run a baseline experiment before child experiments")

        with tempfile.TemporaryDirectory(prefix=".experiment-", dir=self._root) as tmp:
            staging_root = Path(tmp)
            staging_config = ConfigManager(staging_root)
            source_config = (
                self._root / "versions" / parent_id / "config"
                if parent_id
                else self._root / "config"
            )
            staging_config.restore_config(source_config)
            before = _config_state(staging_config)
            self._apply_experiment_params(exp_type, params, staging_config)
            after = _config_state(staging_config)
            if before == after:
                raise ValueError(
                    f"Experiment '{exp_type.value}' produced no config change"
                )

            project = staging_config.read_project()
            models = staging_config.read_models()
            ensemble = staging_config.read_ensemble()
            features = staging_config.read_features()
            data = self.data.load_clean_data()

            result = run_backtest(
                data=data,
                project_config=project,
                models_config=models,
                ensemble_config=ensemble,
                feature_set=features if features.features else None,
                cache_dir=self._root / "artifacts" / "predictions",
            )
            parent_meta = self.versions.get_version(parent_id) if parent_id else None
            eval_report = EvalRunner.from_yaml(
                staging_config.config_dir / "evals.yaml"
            ).run(
                result.metrics,
                parent_metrics=parent_meta.metrics if parent_meta else None,
            )

            version_id = self.versions.next_version_id()
            meta = VersionMeta(
                id=version_id,
                parent=parent_id,
                experiment_type=exp_type.value,
                hypothesis=hypothesis,
                timestamp=_utc_now(),
                data_hash=_data_hash(self._root / "data" / "clean" / "dataset.parquet"),
                metrics=result.metrics,
            )
            version_staging = self.versions.stage_version(
                meta,
                staging_config,
                diff=_config_diff(before, after),
            )
            try:
                self._write_run_results(
                    version_id,
                    result,
                    eval_report.model_dump(mode="json"),
                    run_dir=version_staging / "run",
                )
                atomic_write_json(
                    version_staging / "run" / "state.json",
                    {"status": "complete", "completed_at": _utc_now()},
                )
                self._begin_commit(version_id, current_before)
                self.versions.publish_version(version_staging, version_id)
                self.config.restore_config(
                    self._root / "versions" / version_id / "config"
                )
                atomic_write_text(self._root / "current", version_id)
                self._finish_commit()
            except BaseException:
                shutil.rmtree(version_staging, ignore_errors=True)
                if self._transaction_path.exists():
                    self._recover_interrupted_mutation()
                else:
                    shutil.rmtree(self._rollback_config_dir, ignore_errors=True)
                raise

            return result

    def conclude_experiment(
        self, version_id: str, conclusion: str, verdict: str
    ) -> None:
        allowed = {"improved", "degraded", "inconclusive", "mixed"}
        if verdict not in allowed:
            raise ValueError(
                f"Invalid verdict '{verdict}'. Expected one of {sorted(allowed)}"
            )
        if not conclusion.strip():
            raise ValueError("Conclusion must not be empty")
        with self._mutation("conclude_experiment"):
            self.versions.update_version(
                version_id, conclusion=conclusion, verdict=verdict
            )

    def switch_version(self, version_id: str) -> None:
        with self._mutation("switch_version"):
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

    @contextmanager
    def _mutation(self, operation: str):
        with WorkspaceLock(self._root, operation, self._lock_timeout):
            self._recover_interrupted_mutation()
            self._remove_abandoned_staging()
            yield

    @property
    def _transaction_path(self) -> Path:
        return self._root / ".harness" / "transaction.json"

    @property
    def _rollback_config_dir(self) -> Path:
        return self._root / ".harness" / "transaction-rollback-config"

    def _begin_commit(self, candidate: str, previous: str | None) -> None:
        shutil.rmtree(self._rollback_config_dir, ignore_errors=True)
        self.config.snapshot_config(self._rollback_config_dir)
        atomic_write_json(
            self._transaction_path,
            {
                "id": uuid.uuid4().hex,
                "operation": "publish_experiment",
                "candidate": candidate,
                "previous": previous,
                "started_at": _utc_now(),
            },
        )

    def _finish_commit(self) -> None:
        self._transaction_path.unlink(missing_ok=True)
        shutil.rmtree(self._rollback_config_dir, ignore_errors=True)

    def _recover_interrupted_mutation(self) -> None:
        if not self._transaction_path.exists():
            return
        try:
            transaction = json.loads(self._transaction_path.read_text())
            candidate = transaction["candidate"]
            previous = transaction.get("previous")
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(
                f"Invalid workspace transaction journal: {self._transaction_path}"
            ) from error

        current = self.versions.get_current()
        candidate_dir = self._root / "versions" / candidate
        if current == candidate:
            if not candidate_dir.exists():
                raise RuntimeError(
                    f"Current version {candidate} is missing during transaction recovery"
                )
            self.config.restore_config(candidate_dir / "config")
            atomic_write_text(self._root / "current", candidate)
            self._finish_commit()
            return

        self.versions.delete_version(candidate)
        if previous:
            previous_dir = self._root / "versions" / previous
            if not previous_dir.exists():
                raise RuntimeError(
                    f"Previous version {previous} is missing during transaction recovery"
                )
            self.config.restore_config(previous_dir / "config")
            atomic_write_text(self._root / "current", previous)
        elif self._rollback_config_dir.exists():
            self.config.restore_config(self._rollback_config_dir)
            (self._root / "current").unlink(missing_ok=True)
        self._finish_commit()

    def _remove_abandoned_staging(self) -> None:
        for path in self._root.glob(".experiment-*"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        versions_dir = self._root / "versions"
        if versions_dir.exists():
            for path in versions_dir.glob(".*.tmp"):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)

    def _apply_experiment_params(
        self,
        experiment_type: ExperimentType,
        params: dict,
        config: ConfigManager,
    ) -> None:
        # Copy through JSON so caller-owned nested dictionaries cannot be mutated.
        params = json.loads(json.dumps(params))

        if experiment_type is ExperimentType.BASELINE:
            if not params.get("models"):
                raise ValueError("Baseline requires at least one model")
            models = ModelsConfig()
            for name, raw in params["models"].items():
                definition = dict(raw)
                definition["name"] = name
                models.models[name] = SingleModelConfig(**definition)
            config.write_models(models)
            if "features" in params:
                config.write_features(_feature_set_from_params(params["features"]))
            return

        if experiment_type is ExperimentType.MODEL:
            models = config.read_models()
            definition = dict(params.get("model", params))
            name = definition.pop("name", None)
            if not name:
                raise ValueError("Model experiment requires a model name")
            definition["name"] = name
            models.models[name] = SingleModelConfig(**definition)
            config.write_models(models)
            return

        if experiment_type is ExperimentType.FEATURE:
            features = config.read_features()
            definition = dict(params.get("feature", params))
            name = definition.pop("name", None)
            if not name:
                raise ValueError("Feature experiment requires a feature name")
            definition["name"] = name
            if "type" in definition:
                definition["feature_type"] = definition.pop("type")
            features.features[name] = FeatureDefinition(**definition)
            config.write_features(features)
            return

        if experiment_type is ExperimentType.HYPERPARAMETER:
            models = config.read_models()
            model_name = params.get("model_name")
            changes = params.get("params")
            if not model_name or model_name not in models.models:
                raise ValueError(f"Model not found: {model_name}")
            if not isinstance(changes, dict) or not changes:
                raise ValueError("Hyperparameter experiment requires non-empty params")
            models.models[model_name].params.update(changes)
            config.write_models(models)
            return

        if experiment_type is ExperimentType.FEATURE_SELECTION:
            models = config.read_models()
            model_name = params.get("model_name")
            selected = params.get("features")
            if not model_name or model_name not in models.models:
                raise ValueError(f"Model not found: {model_name}")
            if not isinstance(selected, list) or not selected:
                raise ValueError("Feature selection requires a non-empty features list")
            models.models[model_name].features = selected
            config.write_models(models)
            return

        if experiment_type is ExperimentType.CV_STRATEGY:
            project = config.read_project()
            changes = dict(params.get("cv", params))
            unknown = set(changes) - set(CVConfig.model_fields)
            if unknown:
                raise ValueError(f"Unknown CV fields: {sorted(unknown)}")
            project.cv = CVConfig(**{**project.cv.model_dump(), **changes})
            config.write_project(project)
            return

        if experiment_type is ExperimentType.CALIBRATION:
            ensemble = config.read_ensemble()
            method = params.get("method", params.get("calibration"))
            if method not in {"none", "isotonic", "platt"}:
                raise ValueError("Calibration method must be none, isotonic, or platt")
            ensemble.calibration = method
            config.write_ensemble(ensemble)
            return

        if experiment_type is ExperimentType.ENSEMBLE:
            ensemble = config.read_ensemble()
            changes = dict(params.get("ensemble", params))
            unknown = set(changes) - set(EnsembleConfig.model_fields)
            if unknown:
                raise ValueError(f"Unknown ensemble fields: {sorted(unknown)}")
            config.write_ensemble(
                EnsembleConfig(**{**ensemble.model_dump(), **changes})
            )
            return

        raise AssertionError(f"Unhandled experiment type: {experiment_type}")

    def _write_run_results(
        self,
        version_id: str,
        result: BacktestResult,
        eval_report: dict | None = None,
        run_dir: Path | None = None,
    ) -> None:
        run_dir = run_dir or self._root / "versions" / version_id / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
        if result.predictions is not None:
            result.predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        if result.production_bundle is not None:
            result.production_bundle.save(run_dir / "model.bundle")
            (run_dir / "explainability.json").write_text(
                json.dumps(result.production_bundle.explain(), indent=2)
            )
        (run_dir / "diagnostics.json").write_text(
            json.dumps(
                {
                    "per_fold_metrics": result.per_fold_metrics,
                    "models_trained": result.models_trained,
                    "models_cached": result.models_cached,
                    "models_failed": result.models_failed,
                    "duration_s": result.duration_s,
                    "meta_coefficients": result.meta_coefficients,
                },
                indent=2,
            )
        )
        if eval_report is not None:
            (run_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2))


def _feature_set_from_params(definitions: dict[str, dict]) -> FeatureSet:
    features = FeatureSet()
    for name, raw in definitions.items():
        definition: dict[str, Any] = dict(raw)
        definition["name"] = name
        if "type" in definition:
            definition["feature_type"] = definition.pop("type")
        features.features[name] = FeatureDefinition(**definition)
    return features


def _config_state(config: ConfigManager) -> dict:
    return {
        "project": config.read_project().model_dump(mode="json"),
        "models": config.read_models().model_dump(mode="json"),
        "ensemble": config.read_ensemble().model_dump(mode="json"),
        "features": config.read_features().model_dump(mode="json"),
        "evals": config.read_evals(),
    }


def _config_diff(before: dict, after: dict) -> dict:
    return {
        key: {"before": before[key], "after": after[key]}
        for key in after
        if before.get(key) != after.get(key)
    }


def _data_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
