import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.config.models import ModelsConfig
from harness.ml.config.project import ProjectConfig
from harness.ml.features.resolver import FeatureResolver
from harness.ml.features.schema import FeatureSet
from harness.ml.runners.calibration import Calibrator
from harness.ml.runners.cross_validation import generate_folds
from harness.ml.runners.dag import ModelDAG
from harness.ml.runners.meta_learner import MetaLearner
from harness.ml.runners.postprocessing import apply_postprocessing
from harness.ml.runners.prediction_cache import PredictionCache
from harness.ml.runners.production import ProductionBundle, train_production_bundle
from harness.ml.runners.progress import NoOpProgress
from harness.ml.runners.provider_context import ProviderContext
from harness.ml.runners.training import train_single_model
from harness.ml.tasks.registry import TaskRegistry


@dataclass
class BacktestResult:
    metrics: dict[str, float] = field(default_factory=dict)
    per_fold_metrics: list[dict[str, float]] = field(default_factory=list)
    predictions: pd.DataFrame | None = None
    models_trained: int = 0
    models_cached: int = 0
    models_failed: list[dict] = field(default_factory=list)
    duration_s: float = 0.0
    meta_coefficients: dict[str, float] = field(default_factory=dict)
    production_bundle: ProductionBundle | None = None


def _hash_pandas(value: pd.DataFrame | pd.Series) -> str:
    hashed = pd.util.hash_pandas_object(value, index=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def run_backtest(
    data: pd.DataFrame,
    project_config: ProjectConfig,
    models_config: ModelsConfig,
    ensemble_config: EnsembleConfig,
    feature_set: FeatureSet | None = None,
    cache_dir: Path | None = None,
    progress: Any = None,
) -> BacktestResult:
    """Execute the complete backtest pipeline (4 phases).

    Phase 1: Base model training per fold (DAG-ordered)
    Phase 2: Meta-learner + post-processing (nested LOSO)
    Phase 3: Metrics + diagnostics
    Phase 4: (Production artifacts -- deferred to workspace layer)
    """
    start_time = time.time()
    if progress is None:
        progress = NoOpProgress()

    task = TaskRegistry.get(project_config.task_type)
    supported_metrics = {metric.name for metric in task.metrics()}
    unknown_metrics = sorted(set(project_config.metrics) - supported_metrics)
    if unknown_metrics:
        raise ValueError(
            f"Metrics {unknown_metrics} are not supported for task "
            f"'{project_config.task_type}'. Supported: {sorted(supported_metrics)}"
        )

    target_col = project_config.target_column
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    y_full = data[target_col].copy()
    target_validation = task.validate_target(y_full)
    if not target_validation.is_valid:
        raise ValueError(f"Invalid target: {target_validation.messages}")
    class_labels: list[Any] = []
    if project_config.task_type == "multiclass":
        class_labels = sorted(y_full.unique().tolist())
        label_to_index = {label: index for index, label in enumerate(class_labels)}
        y_full = y_full.map(label_to_index).astype(int)

    # Generate folds from the complete frame because grouped/temporal strategies
    # need metadata columns that must never be passed to a model.
    folds = generate_folds(data, project_config.cv, y_full)
    if not folds:
        raise ValueError("No folds generated -- check CV configuration")

    # --- Step 1: Build a predictor-only frame and resolve features ---
    forbidden_columns = {target_col, *project_config.exclude_columns}
    if project_config.cv.fold_column:
        forbidden_columns.add(project_config.cv.fold_column)
    predictor_data = data.drop(
        columns=[column for column in forbidden_columns if column in data.columns]
    ).copy()

    if feature_set is not None:
        resolver = FeatureResolver()
        # Resolving against predictor_data also prevents formulas and aliases from
        # reading the target or CV metadata.
        predictor_data = resolver.resolve(predictor_data, feature_set)

    if predictor_data.empty or not len(predictor_data.columns):
        raise ValueError("No eligible feature columns remain after exclusions")

    # --- Step 2: Filter to active models ---
    active_models = {
        name: cfg for name, cfg in models_config.models.items() if cfg.active
    }
    if not active_models:
        raise ValueError("No active models configured")

    # --- Step 3: Build and validate DAG ---
    dag = ModelDAG(active_models)
    errors = dag.validate()
    if errors:
        raise ValueError(f"DAG validation failed: {errors}")
    waves = dag.topological_waves()

    for model_name, model_config in active_models.items():
        forbidden_requested = sorted(set(model_config.features) & forbidden_columns)
        if forbidden_requested:
            raise ValueError(
                f"Model '{model_name}' requests forbidden feature columns: "
                f"{forbidden_requested}"
            )
        missing = sorted(set(model_config.features) - set(predictor_data.columns))
        if missing:
            raise ValueError(
                f"Model '{model_name}' requests missing features: {missing}"
            )

    # Dataset and target hashes deliberately include values, ordering, and index.
    data_fingerprint = _hash_pandas(predictor_data)
    target_fingerprint = _hash_pandas(y_full)
    feature_schema = str(
        [(column, str(dtype)) for column, dtype in predictor_data.dtypes.items()]
    )

    # --- Phase 1: Base model training ---
    cache = PredictionCache(cache_dir)
    models_trained = 0
    models_cached = 0
    models_failed: list[dict] = []

    # fold_predictions: {fold_id: DataFrame with prob_modelname columns + target}
    fold_predictions: dict[str, pd.DataFrame] = {}

    for fold_num, (train_idx, test_idx) in enumerate(folds):
        fold_id = str(fold_num)
        progress.on_fold_start(fold_id, fold_num, len(folds))

        X_train = predictor_data.iloc[train_idx].copy()
        X_test = predictor_data.iloc[test_idx].copy()
        y_train = y_full.iloc[train_idx]
        y_test = y_full.iloc[test_idx]

        provider_ctx = ProviderContext()
        fold_preds_df = pd.DataFrame(index=X_test.index)
        fold_preds_df[target_col] = y_test.values

        for wave_num, wave in enumerate(waves):
            for model_name in wave:
                model_config = active_models[model_name]

                # Inject provider features if needed
                X_tr = X_train.copy()
                X_te = X_test.copy()
                if model_config.depends_on:
                    X_tr = provider_ctx.inject_features(
                        X_tr, "train", model_config.depends_on
                    )
                    X_te = provider_ctx.inject_features(
                        X_te, "test", model_config.depends_on
                    )

                # Check fingerprint cache
                fold_fingerprint = hashlib.sha256(
                    train_idx.tobytes() + b":" + test_idx.tobytes()
                ).hexdigest()
                selected_features = model_config.features or list(X_tr.columns)
                selected_schema = str(
                    [(column, str(X_tr[column].dtype)) for column in selected_features]
                )
                fp = cache.compute_fingerprint(
                    model_config.model_dump(),
                    selected_schema or feature_schema,
                    data_fingerprint=data_fingerprint,
                    target_fingerprint=target_fingerprint,
                    fold_fingerprint=fold_fingerprint,
                    task_type=project_config.task_type,
                )
                cached_preds = cache.get(
                    model_name, fold_id, fp, expected_length=len(test_idx)
                )

                if cached_preds is not None:
                    test_preds = cached_preds
                    models_cached += 1
                else:
                    # Train the model
                    try:
                        result = train_single_model(
                            model_config,
                            X_tr,
                            y_train,
                            X_te,
                            task_type=project_config.task_type,
                        )
                        if result.error:
                            models_failed.append(
                                {"name": model_name, "error": result.error}
                            )
                            continue

                        test_preds = result.test_predictions

                        # Store in provider context if this is a provider
                        if model_config.provides:
                            provider_ctx.store_instance(
                                model_name, result.train_predictions, test_preds
                            )

                        # Cache (providers are NOT cached)
                        if not model_config.provides:
                            cache.put(model_name, fold_id, fp, test_preds)

                        models_trained += 1
                        progress.on_model_trained(
                            model_name, fold_id, result.duration_s
                        )

                    except Exception as e:
                        models_failed.append({"name": model_name, "error": str(e)})
                        continue

                # Add predictions to fold DataFrame. Multiclass models expose
                # one probability column per class so stacking remains tabular.
                if model_config.include_in_ensemble:
                    if np.asarray(test_preds).ndim == 2:
                        for class_index in range(test_preds.shape[1]):
                            fold_preds_df[f"prob_{model_name}__class_{class_index}"] = (
                                test_preds[:, class_index]
                            )
                    else:
                        fold_preds_df[f"prob_{model_name}"] = test_preds

            progress.on_wave_complete(wave_num, len(waves))

        fold_predictions[fold_id] = fold_preds_df

    # A model that fails in any fold cannot participate in a coherent OOF
    # ensemble. Keep only prediction columns present in every fold.
    common_prediction_columns = set.intersection(
        *[
            {column for column in frame.columns if column.startswith("prob_")}
            for frame in fold_predictions.values()
        ]
    )
    if not common_prediction_columns:
        unique_failures = list(
            dict.fromkeys(
                f"{failure['name']}: {failure['error']}"
                for failure in models_failed
            )
        )
        detail = f" Failures: {'; '.join(unique_failures)}" if unique_failures else ""
        raise RuntimeError(
            "No model produced predictions for every fold; cannot build ensemble."
            + detail
        )
    for fold_id, frame in fold_predictions.items():
        fold_predictions[fold_id] = frame[
            [target_col, *sorted(common_prediction_columns)]
        ]

    # --- Phase 2: Meta-learner ---
    meta_learner = MetaLearner()
    meta_result = meta_learner.train(
        fold_predictions, ensemble_config, target_col, project_config.task_type
    )

    uncalibrated_fold_predictions = {
        fold_id: predictions.copy()
        for fold_id, predictions in meta_result.fold_predictions.items()
    }
    production_calibrator = None

    # Calibrate each holdout using only out-of-fold predictions from the other
    # folds. Fitting a calibrator on its own holdout would leak outcomes.
    if ensemble_config.calibration != "none":
        if project_config.task_type != "binary":
            raise ValueError("Calibration is currently supported only for binary tasks")
        for holdout_id in sorted(meta_result.fold_predictions):
            calibration_ids = [
                fold_id
                for fold_id in sorted(meta_result.fold_predictions)
                if fold_id != holdout_id
            ]
            if not calibration_ids:
                continue
            calibration_y = np.concatenate(
                [
                    fold_predictions[fold_id][target_col].values
                    for fold_id in calibration_ids
                ]
            )
            calibration_pred = np.concatenate(
                [uncalibrated_fold_predictions[fold_id] for fold_id in calibration_ids]
            )
            calibrator = Calibrator.fit(
                calibration_y, calibration_pred, ensemble_config.calibration
            )
            meta_result.fold_predictions[holdout_id] = Calibrator.transform(
                uncalibrated_fold_predictions[holdout_id], calibrator
            )
        production_calibrator = Calibrator.fit(
            np.concatenate(
                [
                    fold_predictions[fold_id][target_col].values
                    for fold_id in sorted(fold_predictions, key=int)
                ]
            ),
            np.concatenate(
                [
                    uncalibrated_fold_predictions[fold_id]
                    for fold_id in sorted(fold_predictions, key=int)
                ]
            ),
            ensemble_config.calibration,
        )

    # Apply post-processing to each fold's ensemble predictions
    for fold_id, preds in meta_result.fold_predictions.items():
        meta_result.fold_predictions[fold_id] = apply_postprocessing(
            preds, ensemble_config, project_config.task_type
        )

    # --- Phase 3: Metrics ---
    all_y_true = []
    all_y_pred = []
    all_row_positions = []
    all_row_indices = []
    all_fold_ids = []
    per_fold_metrics = []

    for fold_id in sorted(fold_predictions.keys(), key=int):
        fold_df = fold_predictions[fold_id]
        y_true = fold_df[target_col].values
        y_pred = meta_result.fold_predictions[fold_id]

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        test_idx = folds[int(fold_id)][1]
        all_row_positions.extend(test_idx.tolist())
        all_row_indices.extend(data.index[test_idx].tolist())
        all_fold_ids.extend([fold_id] * len(test_idx))

        prediction_validation = task.validate_predictions(y_pred)
        if not prediction_validation.is_valid:
            raise ValueError(
                f"Invalid predictions in fold {fold_id}: "
                f"{prediction_validation.messages}"
            )
        fold_m = task.compute_metrics(y_true, y_pred, project_config.metrics)
        per_fold_metrics.append(fold_m)

    # Pooled metrics
    pooled_y_true = np.concatenate(all_y_true)
    pooled_y_pred = np.concatenate(all_y_pred)
    pooled_metrics = task.compute_metrics(
        pooled_y_true, pooled_y_pred, project_config.metrics
    )

    # Build predictions with stable source-row and fold identity.
    prediction_data: dict[str, Any] = {
        "row_position": all_row_positions,
        "row_index": all_row_indices,
        "fold_id": all_fold_ids,
        "y_true": (
            [class_labels[int(value)] for value in pooled_y_true]
            if class_labels
            else pooled_y_true
        ),
    }
    conformal_radius = None
    if pooled_y_pred.ndim == 2:
        for class_index in range(pooled_y_pred.shape[1]):
            prediction_data[f"y_pred_class_{class_index}"] = pooled_y_pred[
                :, class_index
            ]
    else:
        prediction_data["y_pred"] = pooled_y_pred
        if (
            project_config.task_type == "regression"
            and ensemble_config.conformal_alpha is not None
        ):
            alpha = ensemble_config.conformal_alpha
            if not 0 < alpha < 1:
                raise ValueError("conformal_alpha must be between 0 and 1")
            residuals = np.abs(pooled_y_true - pooled_y_pred)
            quantile = min(
                1.0, np.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals)
            )
            conformal_radius = float(np.quantile(residuals, quantile, method="higher"))
            prediction_data["y_pred_lower"] = pooled_y_pred - conformal_radius
            prediction_data["y_pred_upper"] = pooled_y_pred + conformal_radius
    all_predictions = pd.DataFrame(prediction_data)

    production_bundle = train_production_bundle(
        predictor_data,
        y_full,
        project_config,
        models_config,
        ensemble_config,
        feature_set,
        meta_result.meta_model,
        meta_result.model_columns,
        meta_result.method,
        calibrator=production_calibrator,
        conformal_radius=conformal_radius,
        class_labels=class_labels,
    )

    duration = time.time() - start_time
    progress.on_backtest_complete(pooled_metrics)

    return BacktestResult(
        metrics=pooled_metrics,
        per_fold_metrics=per_fold_metrics,
        predictions=all_predictions,
        models_trained=models_trained,
        models_cached=models_cached,
        models_failed=models_failed,
        duration_s=duration,
        meta_coefficients=meta_result.meta_coefficients,
        production_bundle=production_bundle,
    )
