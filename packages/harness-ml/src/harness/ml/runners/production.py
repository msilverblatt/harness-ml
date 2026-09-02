from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd
from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.config.models import ModelsConfig, SingleModelConfig
from harness.ml.config.project import ProjectConfig
from harness.ml.features.resolver import FeatureResolver
from harness.ml.features.schema import FeatureSet
from harness.ml.models.registry import ModelRegistry
from harness.ml.runners.dag import ModelDAG
from harness.ml.runners.postprocessing import apply_postprocessing
from harness.ml.runners.provider_context import ProviderContext
from harness.ml.runners.training import train_single_model


@dataclass
class ProductionModel:
    config: SingleModelConfig
    models: list[Any]
    medians: dict[str, float] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class ProductionBundle:
    project_config: ProjectConfig
    models_config: ModelsConfig
    ensemble_config: EnsembleConfig
    feature_set: FeatureSet | None
    models: dict[str, ProductionModel]
    ensemble_model: Any = None
    ensemble_columns: list[str] = field(default_factory=list)
    ensemble_method: str = "average"
    calibrator: Any = None
    conformal_radius: float | None = None
    class_labels: list[Any] = field(default_factory=list)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions from raw, target-optional tabular input."""
        predictors = self._prepare_predictors(data)
        context = ProviderContext()
        prediction_columns: dict[str, np.ndarray] = {}
        active = {
            name: config
            for name, config in self.models_config.models.items()
            if config.active and name in self.models
        }
        for wave in ModelDAG(active).topological_waves():
            for model_name in wave:
                entry = self.models[model_name]
                frame = predictors.copy()
                if entry.config.depends_on:
                    missing = [
                        dependency
                        for dependency in entry.config.depends_on
                        if dependency not in context.available_providers()
                    ]
                    if missing:
                        raise ValueError(
                            f"Missing production providers for {model_name}: {missing}"
                        )
                    frame = context.inject_features(
                        frame, "test", entry.config.depends_on
                    )
                feature_columns = entry.config.features or list(frame.columns)
                frame = frame[feature_columns].copy()
                for column in entry.config.zero_fill_features:
                    if column in frame:
                        frame[column] = frame[column].fillna(0)
                for column, median in entry.medians.items():
                    if column in frame:
                        frame[column] = frame[column].fillna(median)
                if frame.isna().any().any():
                    missing = frame.columns[frame.isna().any()].tolist()
                    raise ValueError(
                        f"Missing values remain in production features: {missing}"
                    )

                wrapper = ModelRegistry.get(entry.config.model_type)
                if wrapper is None:
                    raise ValueError(f"Unknown model type: {entry.config.model_type}")
                predictions = np.mean(
                    [wrapper.predict(model, frame) for model in entry.models], axis=0
                )
                if entry.config.provides:
                    context.store_instance(model_name, predictions, predictions)
                if entry.config.include_in_ensemble:
                    if predictions.ndim == 2:
                        for class_index in range(predictions.shape[1]):
                            prediction_columns[
                                f"prob_{model_name}__class_{class_index}"
                            ] = predictions[:, class_index]
                    else:
                        prediction_columns[f"prob_{model_name}"] = predictions

        missing_columns = set(self.ensemble_columns) - set(prediction_columns)
        if missing_columns:
            raise ValueError(
                f"Production ensemble is missing model outputs: {sorted(missing_columns)}"
            )
        matrix = pd.DataFrame(
            {column: prediction_columns[column] for column in self.ensemble_columns}
        )
        if self.ensemble_method == "average":
            predictions = _average_predictions(matrix)
        elif self.ensemble_model is not None:
            if hasattr(self.ensemble_model, "predict_proba"):
                predictions = self.ensemble_model.predict_proba(matrix.values)
                if predictions.ndim == 2 and predictions.shape[1] == 2:
                    predictions = predictions[:, 1]
            else:
                predictions = self.ensemble_model.predict(matrix.values)
        else:
            predictions = _average_predictions(matrix)

        if self.calibrator is not None:
            from harness.ml.runners.calibration import Calibrator

            predictions = Calibrator.transform(predictions, self.calibrator)
        return apply_postprocessing(
            np.asarray(predictions), self.ensemble_config, self.project_config.task_type
        )

    def predict_interval(self, data: pd.DataFrame) -> pd.DataFrame:
        if (
            self.project_config.task_type != "regression"
            or self.conformal_radius is None
        ):
            raise ValueError("Conformal intervals are not configured for this bundle")
        predictions = self.predict(data)
        return pd.DataFrame(
            {
                "prediction": predictions,
                "lower": predictions - self.conformal_radius,
                "upper": predictions + self.conformal_radius,
            },
            index=data.index,
        )

    def explain(self, data: pd.DataFrame | None = None) -> dict[str, Any]:
        """Return native importance and, when requested, SHAP attributions."""
        by_model = {
            name: entry.feature_importance
            for name, entry in self.models.items()
            if entry.feature_importance
        }
        aggregate: dict[str, float] = {}
        for importance in by_model.values():
            scale = sum(abs(value) for value in importance.values()) or 1.0
            for feature, value in importance.items():
                aggregate[feature] = aggregate.get(feature, 0.0) + abs(value) / scale
        if by_model:
            aggregate = {
                feature: value / len(by_model) for feature, value in aggregate.items()
            }
        explanation: dict[str, Any] = {
            "method": "native_feature_importance",
            "aggregate": dict(
                sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
            ),
            "by_model": by_model,
        }
        if data is not None:
            explanation["shap"] = self._shap_explain(data)
        return explanation

    def _shap_explain(self, data: pd.DataFrame) -> dict[str, dict[str, float]]:
        try:
            import shap
        except ImportError as error:
            raise ImportError(
                "SHAP explainability requires `pip install harness-ml[explain]`"
            ) from error
        predictors = self._prepare_predictors(data)
        results: dict[str, dict[str, float]] = {}
        for name, entry in self.models.items():
            if entry.config.depends_on:
                continue
            columns = entry.config.features or list(predictors.columns)
            frame = predictors[columns].copy()
            for column in entry.config.zero_fill_features:
                if column in frame:
                    frame[column] = frame[column].fillna(0)
            frame = frame.fillna(entry.medians)
            values = shap.Explainer(entry.models[0], frame)(frame).values
            importance = np.abs(values)
            while importance.ndim > 2:
                importance = importance.mean(axis=-1)
            mean_importance = importance.mean(axis=0)
            results[name] = {
                column: float(value)
                for column, value in zip(columns, mean_importance, strict=True)
            }
        return results

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("wb") as handle:
            cloudpickle.dump(self, handle)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> ProductionBundle:
        with Path(path).open("rb") as handle:
            bundle = cloudpickle.load(handle)
        if not isinstance(bundle, cls):
            raise TypeError("Artifact is not a Harness ProductionBundle")
        return bundle

    def _prepare_predictors(self, data: pd.DataFrame) -> pd.DataFrame:
        forbidden = {
            self.project_config.target_column,
            *self.project_config.exclude_columns,
        }
        if self.project_config.cv.fold_column:
            forbidden.add(self.project_config.cv.fold_column)
        predictors = data.drop(
            columns=[column for column in forbidden if column in data],
            errors="ignore",
        ).copy()
        if self.feature_set is not None:
            predictors = FeatureResolver().resolve(predictors, self.feature_set)
        return predictors


def train_production_bundle(
    predictors: pd.DataFrame,
    target: pd.Series,
    project_config: ProjectConfig,
    models_config: ModelsConfig,
    ensemble_config: EnsembleConfig,
    feature_set: FeatureSet | None,
    ensemble_model: Any,
    ensemble_columns: list[str],
    ensemble_method: str,
    calibrator: Any = None,
    conformal_radius: float | None = None,
    class_labels: list[Any] | None = None,
) -> ProductionBundle:
    """Fit serializable base models on all available rows after OOF evaluation."""
    entries: dict[str, ProductionModel] = {}
    context = ProviderContext()
    active = {
        name: config for name, config in models_config.models.items() if config.active
    }
    for wave in ModelDAG(active).topological_waves():
        for model_name in wave:
            config = active[model_name]
            frame = predictors.copy()
            if config.depends_on:
                available = context.available_providers()
                if any(dependency not in available for dependency in config.depends_on):
                    continue
                frame = context.inject_features(frame, "test", config.depends_on)
            result = train_single_model(
                config,
                frame,
                target,
                frame,
                task_type=project_config.task_type,
            )
            if result.error or not result.models:
                continue
            importance = (
                result.fit_result.feature_importance if result.fit_result else {}
            )
            entries[model_name] = ProductionModel(
                config=config.model_copy(deep=True),
                models=result.models,
                medians=result.feature_medians,
                feature_importance=importance,
            )
            if config.provides:
                context.store_instance(
                    model_name, result.test_predictions, result.test_predictions
                )

    required_models = {
        column.removeprefix("prob_").split("__class_", 1)[0]
        for column in ensemble_columns
    }
    missing = required_models - set(entries)
    if missing:
        raise RuntimeError(
            f"Could not fit production models required by ensemble: {sorted(missing)}"
        )
    return ProductionBundle(
        project_config=project_config.model_copy(deep=True),
        models_config=models_config.model_copy(deep=True),
        ensemble_config=ensemble_config.model_copy(deep=True),
        feature_set=feature_set.model_copy(deep=True) if feature_set else None,
        models=entries,
        ensemble_model=ensemble_model,
        ensemble_columns=ensemble_columns,
        ensemble_method=ensemble_method,
        calibrator=calibrator,
        conformal_radius=conformal_radius,
        class_labels=[_python_scalar(value) for value in (class_labels or [])],
    )


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _average_predictions(frame: pd.DataFrame) -> np.ndarray:
    class_suffixes = sorted(
        {
            column.split("__class_", 1)[1]
            for column in frame.columns
            if "__class_" in column
        },
        key=int,
    )
    if class_suffixes:
        return np.column_stack(
            [
                frame[
                    [column for column in frame if column.endswith(f"__class_{suffix}")]
                ].mean(axis=1)
                for suffix in class_suffixes
            ]
        )
    return frame.mean(axis=1).values
