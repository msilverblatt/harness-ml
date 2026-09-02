from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from harness.ml.config.models import SingleModelConfig
from harness.ml.models.registry import ModelRegistry
from harness.ml.models.protocol import FitResult
from harness.ml.features.augmentation import augment_symmetric


@dataclass
class TrainingResult:
    model_name: str
    train_predictions: np.ndarray
    test_predictions: np.ndarray
    fit_result: FitResult
    models: list[Any] = field(default_factory=list)
    feature_medians: dict[str, float] = field(default_factory=dict)
    fingerprint: str = ""
    from_cache: bool = False
    duration_s: float = 0.0
    error: str | None = None


def train_single_model(
    model_config: SingleModelConfig,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    task_type: str = "binary",
    random_state: int = 42,
) -> TrainingResult:
    """Train one model on one fold.

    Inner loop:
    1. Apply training_filter (if configured)
    2. Zero-fill specified features
    3. Drop NaN rows in feature columns
    4. Compute class weights (if configured)
    5. Apply symmetric augmentation (if configured)
    6. Train with n_seeds, average predictions
    7. Predict on both train and test
    """
    import time

    start = time.time()

    model_wrapper = ModelRegistry.get(model_config.model_type)
    if model_wrapper is None:
        return TrainingResult(
            model_name=model_config.name,
            train_predictions=np.array([]),
            test_predictions=np.array([]),
            fit_result=FitResult(model=None),
            error=f"Unknown model type: {model_config.model_type}",
        )

    # Use configured features or all columns
    feature_cols = (
        model_config.features if model_config.features else list(X_train.columns)
    )

    # Work with copies
    X_tr = X_train[feature_cols].copy()
    y_tr = y_train.copy()
    X_te = X_test[feature_cols].copy()

    # 1. Training filter
    if model_config.training_filter:
        combined = X_tr.copy()
        combined["__target__"] = y_tr.values
        mask = combined.eval(model_config.training_filter)
        X_tr = X_tr[mask]
        y_tr = y_tr[mask]

    # 2. Zero-fill
    for col in model_config.zero_fill_features:
        if col in X_tr.columns:
            X_tr[col] = X_tr[col].fillna(0)
            X_te[col] = X_te[col].fillna(0)

    # 3. Drop NaN rows
    valid_mask = X_tr.notna().all(axis=1)
    X_tr = X_tr[valid_mask]
    y_tr = y_tr[valid_mask]

    # Fill test NaN with the exact training medians retained for production.
    feature_medians = {}
    for col in X_tr.columns:
        if pd.api.types.is_numeric_dtype(X_tr[col]):
            median = X_tr[col].median()
            if pd.notna(median):
                feature_medians[col] = float(median)
                X_te[col] = X_te[col].fillna(median)

    # 4. Class weights (sklearn format)
    sample_weight = None
    if model_config.class_weight and task_type != "regression":
        from sklearn.utils.class_weight import compute_sample_weight

        sample_weight = compute_sample_weight(model_config.class_weight, y_tr)

    # 5. Symmetric augmentation
    if model_config.augment_symmetry:
        aug_df = X_tr.copy()
        aug_df["_target"] = y_tr.values
        augmented = augment_symmetric(aug_df, "_target", task_type)
        X_tr = augmented.drop(columns=["_target"])
        y_tr = augmented["_target"]
        if sample_weight is not None:
            sample_weight = np.concatenate([sample_weight, sample_weight])

    # 6. Multi-seed training
    params = dict(model_config.params)
    default_p = model_wrapper.default_params(task_type)
    merged_params = {}
    merged_params.update(default_p)
    merged_params.update(params)  # user params override defaults

    n_seeds = model_config.n_seeds
    all_train_preds = []
    all_test_preds = []
    last_fit_result = None
    fitted_models = []

    for seed in range(n_seeds):
        seed_params = dict(merged_params)
        if "random_state" in seed_params or model_wrapper.supports_multi_seed():
            seed_params["random_state"] = random_state + seed

        fit_result = model_wrapper.fit(X_tr, y_tr, None, None, seed_params)
        last_fit_result = fit_result
        fitted_models.append(fit_result.model)

        train_preds = model_wrapper.predict(fit_result.model, X_tr)
        test_preds = model_wrapper.predict(fit_result.model, X_te)

        all_train_preds.append(train_preds)
        all_test_preds.append(test_preds)

    avg_train = np.mean(all_train_preds, axis=0) if all_train_preds else np.array([])
    avg_test = np.mean(all_test_preds, axis=0) if all_test_preds else np.array([])

    duration = time.time() - start

    return TrainingResult(
        model_name=model_config.name,
        train_predictions=avg_train,
        test_predictions=avg_test,
        fit_result=last_fit_result,
        models=fitted_models,
        feature_medians=feature_medians,
        duration_s=duration,
    )
