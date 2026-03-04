"""Model training internals for the pipeline runner.

Handles per-model training with feature extraction, train season filtering,
regressor CDF conversion, multi-seed averaging, validation set splitting,
and matchup symmetry augmentation.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from easyml.core.models.registry import ModelRegistry
from easyml.core.runner.schema import ModelDef

logger = logging.getLogger(__name__)


def train_single_model(
    model_name: str,
    model_def: ModelDef,
    train_df: pd.DataFrame,
    registry: ModelRegistry,
    target_season: int | None = None,
    augment_symmetry: bool = False,
) -> tuple[Any, list[str], dict[str, Any]]:
    """Train a single model from its ModelDef.

    Handles: feature extraction, train_seasons filtering,
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
        Training data with columns for features, 'result', 'season',
        and optionally 'margin'.
    registry : ModelRegistry
        Model registry for creating model instances.
    target_season : int | None
        If set, filter training data to seasons < target_season.
    augment_symmetry : bool
        If True, double data by negating diff_* features and flipping labels.

    Returns
    -------
    tuple of (fitted_model_or_list, feature_columns, metrics_dict)
        fitted_model_or_list: single model if n_seeds==1, list otherwise
        feature_columns: list of feature column names used
        metrics_dict: dict with training metadata (e.g. cdf_scale)
    """
    # Filter by train_seasons
    df = _filter_train_seasons(train_df, model_def.train_seasons, target_season)

    if len(df) == 0:
        raise ValueError(
            f"No training data for model {model_name} after filtering "
            f"(train_seasons={model_def.train_seasons}, target_season={target_season})"
        )

    # Get feature columns that exist in the data
    feature_cols = [f for f in model_def.features if f in df.columns]
    if not feature_cols:
        raise ValueError(
            f"No features found in data for model {model_name}. "
            f"Requested: {model_def.features}"
        )

    is_regressor = _is_regressor(model_def)

    # Extract X, y
    X = df[feature_cols].values.astype(np.float64)
    if is_regressor:
        if "margin" not in df.columns:
            raise ValueError(
                f"Model {model_name} is a regressor but 'margin' column not found"
            )
        y = df["margin"].values.astype(np.float64)
    else:
        y = df["result"].values.astype(np.float64)

    # Augment symmetry if requested
    if augment_symmetry:
        X, y = _augment_matchup_symmetry(X, y, feature_cols)

    # Fit CDF scale for regressors
    cdf_scale = model_def.cdf_scale
    if is_regressor and cdf_scale is None:
        # Need binary labels for CDF fitting
        if augment_symmetry:
            # Use the original (non-augmented) data for fitting
            orig_X = df[feature_cols].values.astype(np.float64)
            orig_y_binary = df["result"].values.astype(np.float64)
            cdf_scale = _fit_cdf_scale_from_data(orig_X, orig_y_binary, y[:len(df)])
        else:
            y_binary = df["result"].values.astype(np.float64)
            cdf_scale = _fit_cdf_scale_from_data(X, y_binary, y)

    # Determine the model type to use with registry
    model_type = _resolve_model_type(model_def)

    metrics: dict[str, Any] = {}
    if cdf_scale is not None:
        metrics["cdf_scale"] = cdf_scale

    # Multi-seed training
    n_seeds = model_def.n_seeds
    if n_seeds > 1:
        models = []
        for seed_idx in range(n_seeds):
            params = dict(model_def.params)
            params["random_state"] = seed_idx
            model = _create_model(registry, model_type, params, model_def)
            model.fit(X, y)
            models.append(model)
        metrics["n_seeds"] = n_seeds
        return models, feature_cols, metrics
    else:
        params = dict(model_def.params)
        model = _create_model(registry, model_type, params, model_def)
        model.fit(X, y)
        return model, feature_cols, metrics


def predict_single_model(
    model: Any,
    model_def: ModelDef,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    cdf_scale: float | None = None,
) -> np.ndarray:
    """Generate predictions from a fitted model.

    Handles: regressor -> CDF probability conversion, NaN handling.
    For multi-seed: model is a list, predictions are averaged.

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
        without a pre-set cdf_scale.

    Returns
    -------
    np.ndarray
        Probability array.
    """
    X_test = test_df[feature_columns].values.astype(np.float64)

    # Handle NaN in test features
    nan_mask = np.isnan(X_test)
    if nan_mask.any():
        col_medians = np.nanmedian(X_test, axis=0)
        for col_idx in range(X_test.shape[1]):
            row_mask = nan_mask[:, col_idx]
            X_test[row_mask, col_idx] = col_medians[col_idx]

    is_regressor = _is_regressor(model_def)

    if isinstance(model, list):
        # Multi-seed: average predictions
        all_preds = []
        for m in model:
            if is_regressor:
                margins = m.predict_margin(X_test)
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
            margins = model.predict_margin(X_test)
            scale = cdf_scale if cdf_scale is not None else model_def.cdf_scale
            if scale is None:
                raise ValueError("cdf_scale required for regressor predictions")
            return _margin_to_prob(margins, scale)
        else:
            return model.predict_proba(X_test)


def _margin_to_prob(margins: np.ndarray, cdf_scale: float) -> np.ndarray:
    """Convert predicted margins to win probabilities via normal CDF."""
    return norm.cdf(margins / cdf_scale)


def _fit_cdf_scale(margins: np.ndarray, y_binary: np.ndarray) -> float:
    """Fit CDF scale from margin predictions and binary outcomes.

    Find scale where CDF-converted margin predictions best match
    binary outcomes (minimize Brier score).
    """
    def brier_at_scale(scale: float) -> float:
        probs = norm.cdf(margins / scale)
        return float(np.mean((probs - y_binary) ** 2))

    result = minimize_scalar(brier_at_scale, bounds=(0.1, 50.0), method="bounded")
    return float(result.x)


def _fit_cdf_scale_from_data(
    X: np.ndarray,
    y_binary: np.ndarray,
    y_margin: np.ndarray,
) -> float:
    """Fit CDF scale from training margins and binary labels.

    Uses the raw training margins directly (no model needed).
    """
    return _fit_cdf_scale(y_margin, y_binary)


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


def _filter_train_seasons(
    df: pd.DataFrame,
    train_seasons: str,
    target_season: int | None = None,
) -> pd.DataFrame:
    """Filter training data by season constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'season' column.
    train_seasons : str
        One of 'all' or 'last_N' (e.g. 'last_5').
    target_season : int | None
        If set, filter to seasons < target_season (for backtesting).
    """
    result = df
    if target_season is not None:
        result = result[result["season"] < target_season]
    if train_seasons == "all":
        return result
    elif train_seasons.startswith("last_"):
        n = int(train_seasons.split("_")[1])
        if len(result) == 0:
            return result
        max_season = result["season"].max()
        return result[result["season"] > max_season - n]
    return result


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
) -> Any:
    """Create a model instance from the registry, handling mode/cdf_scale kwargs.

    XGBoost models accept mode and cdf_scale as constructor kwargs.
    Other models just get params.
    """
    is_regressor = _is_regressor(model_def)

    if model_type == "xgboost" and is_regressor:
        # XGBoostModel takes mode and cdf_scale as keyword arguments
        from easyml.core.models.wrappers.xgboost import XGBoostModel
        return XGBoostModel(
            params=params,
            mode="regressor",
            cdf_scale=model_def.cdf_scale,
        )
    else:
        return registry.create(model_type, params=params)
