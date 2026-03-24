from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import time
import numpy as np
import pandas as pd

from harness.ml.config.project import ProjectConfig
from harness.ml.config.models import ModelsConfig, SingleModelConfig
from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.features.schema import FeatureSet
from harness.ml.features.resolver import FeatureResolver
from harness.ml.tasks.registry import TaskRegistry
from harness.ml.runners.cross_validation import generate_folds
from harness.ml.runners.dag import ModelDAG
from harness.ml.runners.provider_context import ProviderContext
from harness.ml.runners.prediction_cache import PredictionCache
from harness.ml.runners.training import train_single_model
from harness.ml.runners.meta_learner import MetaLearner
from harness.ml.runners.postprocessing import apply_postprocessing
from harness.ml.runners.progress import NoOpProgress


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

    target_col = project_config.target_column
    y_full = data[target_col]

    # --- Step 1: Resolve features ---
    if feature_set is not None:
        resolver = FeatureResolver()
        data = resolver.resolve(data, feature_set)

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

    # --- Step 4: Generate CV folds ---
    folds = generate_folds(data, project_config.cv, y_full)
    if not folds:
        raise ValueError("No folds generated -- check CV configuration")

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

        X_train = data.iloc[train_idx].copy()
        X_test = data.iloc[test_idx].copy()
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
                    X_tr = provider_ctx.inject_features(X_tr, "train", model_config.depends_on)
                    X_te = provider_ctx.inject_features(X_te, "test", model_config.depends_on)

                # Check fingerprint cache
                fp = cache.compute_fingerprint(
                    model_config.model_dump(),
                    str(sorted(model_config.features or list(X_tr.columns))),
                )
                cached_preds = cache.get(model_name, fold_id, fp)

                if cached_preds is not None:
                    test_preds = cached_preds
                    models_cached += 1
                else:
                    # Train the model
                    try:
                        result = train_single_model(
                            model_config, X_tr, y_train, X_te,
                            task_type=project_config.task_type,
                        )
                        if result.error:
                            models_failed.append({"name": model_name, "error": result.error})
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
                        progress.on_model_trained(model_name, fold_id, result.duration_s)

                    except Exception as e:
                        models_failed.append({"name": model_name, "error": str(e)})
                        continue

                # Add predictions to fold DataFrame
                if model_config.include_in_ensemble:
                    fold_preds_df[f"prob_{model_name}"] = test_preds

            progress.on_wave_complete(wave_num, len(waves))

        fold_predictions[fold_id] = fold_preds_df

    # --- Phase 2: Meta-learner ---
    meta_learner = MetaLearner()
    meta_result = meta_learner.train(fold_predictions, ensemble_config, target_col)

    # Apply post-processing to each fold's ensemble predictions
    for fold_id, preds in meta_result.fold_predictions.items():
        meta_result.fold_predictions[fold_id] = apply_postprocessing(preds, ensemble_config)

    # --- Phase 3: Metrics ---
    all_y_true = []
    all_y_pred = []
    per_fold_metrics = []

    for fold_id in sorted(fold_predictions.keys()):
        fold_df = fold_predictions[fold_id]
        y_true = fold_df[target_col].values
        y_pred = meta_result.fold_predictions[fold_id]

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)

        fold_m = task.compute_metrics(y_true, y_pred, project_config.metrics)
        per_fold_metrics.append(fold_m)

    # Pooled metrics
    pooled_y_true = np.concatenate(all_y_true)
    pooled_y_pred = np.concatenate(all_y_pred)
    pooled_metrics = task.compute_metrics(pooled_y_true, pooled_y_pred, project_config.metrics)

    # Build predictions DataFrame
    all_predictions = pd.DataFrame({
        "y_true": pooled_y_true,
        "y_pred": pooled_y_pred,
    })

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
    )
