"""Matchup prediction utilities.

Provides generic matchup prediction (predict_all_matchups, compute_interactions).
Sports-specific matchup *generation* (generate_pairwise_matchups) has moved to
the ``harness-sports`` plugin.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from harnessml.core.runner.schema import InteractionDef, ModelDef
from harnessml.core.runner.training import _is_regressor, _margin_to_prob

logger = logging.getLogger(__name__)


def compute_interactions(
    df: pd.DataFrame,
    interactions: dict[str, InteractionDef],
) -> pd.DataFrame:
    """Add interaction feature columns to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with existing columns.
    interactions : dict[str, InteractionDef]
        Mapping of output column name -> InteractionDef.

    Returns
    -------
    pd.DataFrame
        Copy of df with new interaction columns appended.

    Raises
    ------
    KeyError
        If left or right column is not present in df.
    """
    result = df.copy()

    for name, interaction_def in interactions.items():
        if interaction_def.left not in result.columns:
            raise KeyError(
                f"Left column {interaction_def.left!r} not found in DataFrame "
                f"for interaction {name!r}"
            )
        if interaction_def.right not in result.columns:
            raise KeyError(
                f"Right column {interaction_def.right!r} not found in DataFrame "
                f"for interaction {name!r}"
            )

        left = result[interaction_def.left]
        right = result[interaction_def.right]

        if interaction_def.op == "multiply":
            result[name] = left * right
        elif interaction_def.op == "add":
            result[name] = left + right
        elif interaction_def.op == "subtract":
            result[name] = left - right
        elif interaction_def.op == "divide":
            result[name] = (left / right.replace(0, np.nan)).fillna(0)
        elif interaction_def.op == "abs_diff":
            result[name] = (left - right).abs()

    return result


def generate_pairwise_matchups(*args, **kwargs):
    """Generate all pairwise features for entities.

    .. deprecated::
        This function has moved to the ``harness-sports`` plugin.
        Install ``harness-sports`` and import from ``harnessml.sports.matchups``.
    """
    raise ImportError(
        "generate_pairwise_matchups() has moved to harness-sports. "
        "Install harness-sports and import from harnessml.sports.matchups."
    )


def predict_all_matchups(
    matchups: pd.DataFrame,
    models: dict[str, Any],
    model_defs: dict[str, ModelDef],
    feature_medians: dict | None = None,
) -> pd.DataFrame:
    """Run all models on all matchups, return DataFrame with prob_* columns.

    Handles regressor models (CDF conversion) and multi-seed models (averaging).

    Parameters
    ----------
    matchups : pd.DataFrame
        Matchup features from generate_pairwise_matchups().
    models : dict
        Mapping of model_name -> (fitted_model, feature_columns, metrics).
        Where fitted_model is either a single model or list (multi-seed).
    model_defs : dict[str, ModelDef]
        Model definitions keyed by model name.
    feature_medians : dict | None
        Optional medians for NaN imputation in test features.

    Returns
    -------
    pd.DataFrame
        Copy of matchups with additional prob_{model_name} columns.
    """
    result = matchups.copy()

    for model_name, (model, feature_cols, metrics) in models.items():
        model_def = model_defs.get(model_name)
        if model_def is None:
            logger.warning("No ModelDef found for %s, skipping", model_name)
            continue

        try:
            # Extract features from matchups
            available_cols = [c for c in feature_cols if c in result.columns]
            if not available_cols:
                logger.warning(
                    "No features available for model %s, skipping", model_name
                )
                continue

            X = result[available_cols].values.astype(np.float64)

            # Impute NaN
            nan_mask = np.isnan(X)
            if nan_mask.any():
                if feature_medians is not None:
                    for col_idx, col_name in enumerate(available_cols):
                        row_mask = nan_mask[:, col_idx]
                        if row_mask.any():
                            median_val = feature_medians.get(col_name, 0.0)
                            X[row_mask, col_idx] = median_val
                else:
                    # Fall back to column median
                    col_medians = np.nanmedian(X, axis=0)
                    for col_idx in range(X.shape[1]):
                        row_mask = nan_mask[:, col_idx]
                        X[row_mask, col_idx] = col_medians[col_idx]

            is_regressor = _is_regressor(model_def)
            cdf_scale = metrics.get("cdf_scale") if metrics else None

            if isinstance(model, list):
                # Multi-seed: average predictions
                all_preds = []
                for m in model:
                    preds = _predict_one(m, X, is_regressor, cdf_scale, model_def)
                    all_preds.append(preds)
                probs = np.mean(all_preds, axis=0)
            else:
                probs = _predict_one(model, X, is_regressor, cdf_scale, model_def)

            result[f"prob_{model_name}"] = probs

        except Exception:
            logger.exception("Failed to predict with model %s", model_name)
            continue

    return result


def _predict_one(
    model: Any,
    X: np.ndarray,
    is_regressor: bool,
    cdf_scale: float | None,
    model_def: ModelDef,
) -> np.ndarray:
    """Generate predictions from a single fitted model."""
    if is_regressor:
        margins = model.predict_margin(X)
        scale = cdf_scale if cdf_scale is not None else model_def.cdf_scale
        if scale is None:
            raise ValueError("cdf_scale required for regressor predictions")
        return _margin_to_prob(margins, scale)
    else:
        return model.predict_proba(X)
