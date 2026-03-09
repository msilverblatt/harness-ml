"""Model training internals for the pipeline runner.

Handles per-model training with feature extraction, train fold filtering,
regressor CDF conversion, multi-seed averaging, validation set splitting,
and matchup symmetry augmentation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from harnessml.core.logging import get_logger
from harnessml.core.models.registry import ModelRegistry
from harnessml.core.runner.data_utils import get_feature_columns
from harnessml.core.runner.schema import DataConfig, ModelDef
from scipy.optimize import minimize_scalar

logger = get_logger(__name__)


def train_single_model(
    model_name: str,
    model_def: ModelDef,
    train_df: pd.DataFrame,
    registry: ModelRegistry,
    target_fold: int | None = None,
    fold_column: str | None = None,
    augment_symmetry: bool = False,
    target_column: str = "result",
    data_config: DataConfig | None = None,
    task_type: str = "binary",
) -> tuple[Any, list[str], dict[str, Any]]:
    """Train a single model from its ModelDef.

    Handles: feature extraction, train period filtering,
    data augmentation (matchup symmetry), regressor target (margin) vs
    classifier (result), multi-seed training and averaging, CDF scale
    fitting for regressors.

    Parameters
    ----------
    model_name : str
        Name of the model (for logging).
    model_def : ModelDef
        Model definition with type, features, params, mode, n_seeds, etc.
    train_df : pd.DataFrame
        Training data with columns for features, target column,
        fold column, and optionally 'margin'.
    registry : ModelRegistry
        Model registry for creating model instances.
    target_fold : int | None
        If set, filter training data to fold values < target_fold.
    fold_column : str | None
        Column name used for fold/period splitting (e.g. 'season', 'year').
        Required when target_fold is set or train_folds != 'all'.
    augment_symmetry : bool
        If True, double data by negating diff_* features and flipping labels.
    target_column : str
        Name of the target column for classifiers (default: 'result').

    Returns
    -------
    tuple of (fitted_model_or_list, feature_columns, metrics_dict)
        fitted_model_or_list: single model if n_seeds==1, list otherwise
        feature_columns: list of feature column names used
        metrics_dict: dict with training metadata (e.g. cdf_scale)
    """
    # Filter by train_folds setting
    df = _filter_train_folds(train_df, model_def.train_folds, target_fold, fold_column)

    # Apply training_filter (non-destructive row exclusion)
    if model_def.training_filter:
        exclude_expr = model_def.training_filter.get("exclude")
        if exclude_expr:
            mask = ~df.eval(exclude_expr)
            df = df.loc[mask]
            logger.info(
                "training_filter_applied",
                excluded=int((~mask).sum()),
                model=model_name,
                kept=len(df),
            )

    if len(df) == 0:
        raise ValueError(
            f"No training data for model {model_name} after filtering "
            f"(train_folds={model_def.train_folds}, target_fold={target_fold})"
        )

    # Get feature columns
    if model_def.features:
        feature_cols = [f for f in model_def.features if f in df.columns]
    else:
        # Default to all available numeric features
        if data_config is not None:
            feature_cols = get_feature_columns(df, data_config, fold_column=fold_column)
        else:
            # Fallback: all numeric columns except target and fold
            feature_cols = [c for c in df.select_dtypes(include=['number']).columns
                           if c != target_column and c != fold_column]

    # Auto-exclude fold column from features
    if fold_column:
        feature_cols = [f for f in feature_cols if f != fold_column]

    if not feature_cols:
        raise ValueError(
            f"No features found in data for model {model_name}. "
            f"Requested: {model_def.features}"
        )

    # Zero-fill specified features before dropping NaN rows
    if model_def.zero_fill_features:
        fills = {
            col: df[col].fillna(0.0)
            for col in model_def.zero_fill_features
            if col in df.columns
        }
        if fills:
            df = df.assign(**fills)

    # Drop rows with NaN in any feature column
    before_drop = len(df)
    df = df.dropna(subset=feature_cols)
    if len(df) < before_drop:
        logger.info("nan_rows_dropped", before=before_drop, after=len(df), columns=feature_cols)

    if len(df) == 0:
        raise ValueError(
            f"No training data for model {model_name} after dropping NaN rows"
        )

    is_regressor = _is_regressor(model_def)

    # Fold-based early stopping validation split.
    # When a model has early_stopping_rounds in its params, carve the most
    # recent fold as a validation set for early stopping.
    fit_kwargs: dict[str, Any] = {}
    if "early_stopping_rounds" in model_def.params and fold_column and fold_column in df.columns and len(df) > 0:
        max_fold = df[fold_column].max()
        val_mask = df[fold_column] == max_fold
        remaining_train = int((~val_mask).sum())
        if int(val_mask.sum()) > 10 and remaining_train > 10:
            val_df = df[val_mask]
            df = df[~val_mask]
            X_val = val_df[feature_cols].values.astype(np.float64)
            y_val = (
                val_df[_regressor_target_col(df, target_column)].values.astype(np.float64)
                if is_regressor
                else val_df[target_column].values.astype(np.float64)
            )
            fit_kwargs["eval_set"] = [(X_val, y_val)]

    # Extract X, y
    X = df[feature_cols].values.astype(np.float64)
    if is_regressor:
        reg_col = _regressor_target_col(df, target_column)
        y = df[reg_col].values.astype(np.float64)
    else:
        y = df[target_column].values.astype(np.float64)

    # Compute sample weights for class imbalance handling
    sample_weight: np.ndarray | None = None
    if model_def.class_weight is not None:
        if is_regressor:
            logger.warning(
                "class_weight ignored for regressor model",
                model=model_name,
            )
        else:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight(model_def.class_weight, y)

    # Augment symmetry if requested
    if augment_symmetry:
        original_len = len(X)
        X, y = _augment_matchup_symmetry(X, y, feature_cols)
        logger.info("symmetry_augmented", original=original_len, augmented=len(X))
        # Re-compute sample weights for augmented data
        if sample_weight is not None:
            sample_weight = compute_sample_weight(model_def.class_weight, y)

    # Use preset CDF scale if provided
    cdf_scale = model_def.cdf_scale

    # Determine the model type to use with registry
    model_type = _resolve_model_type(model_def)

    metrics: dict[str, Any] = {}
    metrics["n_train_rows"] = len(df)

    # Multi-seed training
    n_seeds = model_def.n_seeds
    if n_seeds > 1:
        models = []
        for seed_idx in range(n_seeds):
            params = dict(model_def.params)
            params["random_state"] = seed_idx
            model = _create_model(registry, model_type, params, model_def, cdf_scale)
            model.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
            models.append(model)
        metrics["n_seeds"] = n_seeds
        # Fit CDF scale post-training from first model's predictions
        # Skip for pure regression tasks — CDF conversion is only for
        # sports-style regressor→probability pipelines
        if is_regressor and cdf_scale is None and task_type != "regression":
            train_margins = models[0].predict_margin(X)
            y_binary = (
                df[target_column].values.astype(np.float64)
                if not augment_symmetry
                else np.concatenate([
                    df[target_column].values.astype(np.float64),
                    1.0 - df[target_column].values.astype(np.float64),
                ])
            )
            cdf_scale = _fit_cdf_scale_after_training(train_margins, y_binary)
        if cdf_scale is not None:
            metrics["cdf_scale"] = cdf_scale
        return models, feature_cols, metrics
    else:
        params = dict(model_def.params)
        model = _create_model(registry, model_type, params, model_def, cdf_scale)
        model.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        # Fit CDF scale post-training from model's predictions
        # Skip for pure regression tasks
        if is_regressor and cdf_scale is None and task_type != "regression":
            train_margins = model.predict_margin(X)
            y_binary = (
                df[target_column].values.astype(np.float64)
                if not augment_symmetry
                else np.concatenate([
                    df[target_column].values.astype(np.float64),
                    1.0 - df[target_column].values.astype(np.float64),
                ])
            )
            cdf_scale = _fit_cdf_scale_after_training(train_margins, y_binary)
        if cdf_scale is not None:
            metrics["cdf_scale"] = cdf_scale
        return model, feature_cols, metrics


def predict_single_model(
    model: Any,
    model_def: ModelDef,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    cdf_scale: float | None = None,
    feature_medians: dict[str, float] | None = None,
    task_type: str = "binary",
) -> np.ndarray:
    """Generate predictions from a fitted model.

    Handles: regressor -> CDF probability conversion, NaN handling.
    For multi-seed: model is a list, predictions are averaged.
    For pure regression tasks (task_type="regression"), returns raw
    predictions without CDF conversion.

    Parameters
    ----------
    model : fitted model or list of fitted models
        The trained model(s).
    model_def : ModelDef
        Model definition.
    test_df : pd.DataFrame
        Test data with feature columns.
    feature_columns : list[str]
        Feature column names to extract from test_df.
    cdf_scale : float | None
        CDF scale for regressor conversion. Required for regressors
        without a pre-set cdf_scale (not used for pure regression).
    feature_medians : dict[str, float] | None
        Pre-computed feature medians from training data for NaN imputation.
        If None, falls back to computing medians from test data.
    task_type : str
        Task type — "regression" skips CDF conversion.

    Returns
    -------
    np.ndarray
        Probability array (binary/multiclass) or raw predictions (regression).
    """
    X_test = test_df[feature_columns].values.astype(np.float64)

    # Handle NaN in test features
    nan_mask = np.isnan(X_test)
    if nan_mask.any():
        if feature_medians is not None:
            # Use pre-computed training-data medians
            for col_idx, col_name in enumerate(feature_columns):
                row_mask = nan_mask[:, col_idx]
                if row_mask.any():
                    fill_val = feature_medians.get(col_name, 0.0)
                    X_test[row_mask, col_idx] = fill_val
        else:
            # Fallback: compute medians from test data
            col_medians = np.nanmedian(X_test, axis=0)
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            for col_idx in range(X_test.shape[1]):
                row_mask = nan_mask[:, col_idx]
                X_test[row_mask, col_idx] = col_medians[col_idx]

    is_regressor = _is_regressor(model_def)
    pure_regression = task_type == "regression"

    if isinstance(model, list):
        # Multi-seed: average predictions
        all_preds = []
        for m in model:
            if is_regressor:
                margins = m.predict_margin(X_test)
                if pure_regression:
                    preds = margins
                else:
                    scale = cdf_scale if cdf_scale is not None else model_def.cdf_scale
                    if scale is None:
                        raise ValueError("cdf_scale required for regressor predictions")
                    preds = _margin_to_prob(margins, scale)
            else:
                preds = m.predict_proba(X_test)
            all_preds.append(preds)
        return np.mean(all_preds, axis=0)
    else:
        if is_regressor:
            if pure_regression:
                return model.predict_margin(X_test)
            scale = cdf_scale if cdf_scale is not None else model_def.cdf_scale
            if scale is None:
                raise ValueError("cdf_scale required for regressor predictions")
            # If model supports set_cdf_scale, let it handle per-seed CDF
            # conversion internally (avoids Jensen's inequality bias)
            if hasattr(model, "set_cdf_scale"):
                model.set_cdf_scale(scale)
                return model.predict_proba(X_test)
            margins = model.predict_margin(X_test)
            return _margin_to_prob(margins, scale)
        else:
            return model.predict_proba(X_test)


def _sigmoid(x: np.ndarray, scale: float) -> np.ndarray:
    """Logistic sigmoid CDF: maps values to (0, 1) probabilities."""
    z = np.asarray(x / scale, dtype=np.float64)
    # Numerically stable sigmoid: use exp(-z) for z>=0, exp(z) for z<0
    out = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    return np.clip(out, 1e-7, 1 - 1e-7)


def _margin_to_prob(margins: np.ndarray, cdf_scale: float) -> np.ndarray:
    """Convert predicted margins to probabilities via logistic sigmoid."""
    return _sigmoid(margins, cdf_scale)


def _fit_cdf_scale(margins: np.ndarray, y_binary: np.ndarray) -> float:
    """Fit CDF scale from margin predictions and binary outcomes.

    Find scale where sigmoid-converted margin predictions best match
    binary outcomes (minimize Brier score).
    """
    def brier_at_scale(log_scale: float) -> float:
        scale = np.exp(log_scale)
        probs = _sigmoid(margins, scale)
        return float(np.mean((probs - y_binary) ** 2))

    margin_std = max(float(np.std(margins)), 1.0)
    lo = np.log(margin_std / 10.0)
    hi = np.log(margin_std * 10.0)
    result = minimize_scalar(brier_at_scale, bounds=(lo, hi), method="bounded")
    return float(np.exp(result.x))


def _fit_cdf_scale_after_training(
    margins: np.ndarray,
    y_binary: np.ndarray,
) -> float:
    """Fit CDF scale from model's predicted margins and binary labels.

    Called after training to find the sigmoid scale that minimizes
    Brier score on the training set predictions.
    """
    return _fit_cdf_scale(margins, y_binary)


def _fit_cdf_scale_from_data(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_margin: np.ndarray,
) -> float:
    """Fit CDF scale from training margins and binary labels.

    Uses the raw training margins directly (no model needed).
    Kept for backward compatibility.
    """
    return _fit_cdf_scale(y_margin, y_binary)


def compute_feature_medians(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    """Compute per-feature medians from training data for NaN imputation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    feature_columns : list[str]
        Feature column names.

    Returns
    -------
    dict[str, float]
        Mapping from feature name to median value. Falls back to 0.0
        if a feature is entirely NaN.
    """
    medians = {}
    for col in feature_columns:
        if col in train_df.columns:
            med = train_df[col].median()
            medians[col] = 0.0 if pd.isna(med) else float(med)
        else:
            medians[col] = 0.0
    return medians


def _augment_matchup_symmetry(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Double data by negating diff_* features and flipping result/margin labels.

    For diff_* features, the sign is negated. For other features, values
    are kept as-is (they should be symmetric features like seed_num).
    The label y is flipped: for classifiers (binary) -> 1-y,
    for regressors (margin) -> -y.
    """
    X_aug = X.copy()
    for i, col in enumerate(feature_cols):
        if col.startswith("diff_"):
            X_aug[:, i] = -X_aug[:, i]

    # Determine if y is binary (classifier) or continuous (regressor)
    unique_vals = np.unique(y)
    if len(unique_vals) <= 2 and set(unique_vals).issubset({0.0, 1.0}):
        y_aug = 1.0 - y
    else:
        y_aug = -y

    return np.vstack([X, X_aug]), np.concatenate([y, y_aug])


def _filter_train_folds(
    df: pd.DataFrame,
    train_folds: str,
    target_fold: int | None = None,
    fold_column: str | None = None,
) -> pd.DataFrame:
    """Filter training data by fold constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Data with the fold column.
    train_folds : str
        One of 'all' or 'last_N' (e.g. 'last_5').
    target_fold : int | None
        If set, filter to fold values < target_fold (for backtesting).
    fold_column : str | None
        Column name used for fold/period splitting.
        Required when target_fold is set or train_folds != 'all'.
    """
    result = df
    if target_fold is not None and fold_column and fold_column in result.columns:
        result = result[result[fold_column] < target_fold]
    if train_folds == "all":
        return result
    elif train_folds.startswith("last_"):
        n = int(train_folds.split("_")[1])
        if len(result) == 0 or not fold_column or fold_column not in result.columns:
            return result
        max_fold = result[fold_column].max()
        return result[result[fold_column] > max_fold - n]
    return result


def _regressor_target_col(df: pd.DataFrame, target_column: str) -> str:
    """Return the column to use as regression target.

    Uses 'margin' if present (backward compat for sports), otherwise
    falls back to the configured target_column.
    """
    if "margin" in df.columns:
        return "margin"
    if target_column in df.columns:
        return target_column
    raise ValueError(
        f"Regressor requires either 'margin' or target column "
        f"'{target_column}' in the data"
    )


def _is_regressor(model_def: ModelDef) -> bool:
    """Determine if a model definition represents a regressor."""
    if model_def.mode == "regressor":
        return True
    if model_def.prediction_type == "margin":
        return True
    if model_def.type == "xgboost_regression":
        return True
    return False


def _resolve_model_type(model_def: ModelDef) -> str:
    """Resolve the model type string for the registry.

    Maps compound types like 'xgboost_regression' to their base registry
    type ('xgboost').
    """
    model_type = model_def.type
    # xgboost_regression -> xgboost (the XGBoostModel handles both modes)
    if model_type == "xgboost_regression":
        return "xgboost"
    return model_type


def _create_model(
    registry: ModelRegistry,
    model_type: str,
    params: dict,
    model_def: ModelDef,
    cdf_scale: float | None = None,
) -> Any:
    """Create a model instance from the registry, forwarding kwargs generically.

    The registry.create() method inspects each model's constructor and
    forwards only the kwargs it accepts (mode, cdf_scale, n_seeds, etc.).
    Params from model_def.params that match constructor kwargs (e.g.
    normalize, batch_norm, weight_decay for MLP/TabNet) are forwarded
    as keyword arguments rather than staying in the params dict.
    """
    is_regressor = _is_regressor(model_def)
    mode = "regressor" if is_regressor else model_def.mode
    # Build kwargs from params first, then overlay model_def-level fields.
    # This lets params like n_seeds, normalize, etc. flow through to
    # model constructors that accept them as explicit keyword arguments.
    extra_kwargs: dict[str, Any] = {}
    extra_kwargs.update(params)
    extra_kwargs["mode"] = mode
    extra_kwargs["cdf_scale"] = cdf_scale
    # Only set n_seeds from model_def if not already in params
    if "n_seeds" not in extra_kwargs:
        extra_kwargs["n_seeds"] = model_def.n_seeds
    return registry.create(
        model_type,
        params=params,
        **extra_kwargs,
    )
