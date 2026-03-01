"""Ensemble post-processing pipeline.

Applies the full chain: model filtering, pre-calibration, meta-learner,
post-calibration, temperature scaling, clipping, availability adjustment,
and seed compression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from easyml.runner.calibration import temperature_scale
from easyml.runner.meta_learner import StackedEnsemble


def apply_ensemble_postprocessing(
    preds: pd.DataFrame,
    meta_learner: StackedEnsemble,
    calibrator,
    ensemble_config: dict,
    team_features: pd.DataFrame | None = None,
    pre_calibrators: dict | None = None,
) -> pd.DataFrame:
    """Apply full 9-step ensemble post-processing pipeline.

    Steps:
    1. Extract base model prob columns (prob_*), excluding prob_logreg_seed
    2. Filter out excluded models (ensemble_config["exclude_models"])
    3. Apply pre-calibration to specified models
    4. Run meta-learner prediction (using seed_diffs from diff_seed_num column)
    5. Apply post-calibration
    6. Temperature scaling
    7. Probability clipping
    8. Availability adjustment (if strength > 0)
    9. Seed-proximity compression (if configured and > 0)

    Adds 'prob_ensemble' column to preds DataFrame and returns it.
    """
    # Step 1: Extract base model prob columns
    prob_cols = [
        c for c in preds.columns
        if c.startswith("prob_") and c != "prob_ensemble"
    ]

    # Step 2: Filter out excluded models
    exclude = set(ensemble_config.get("exclude_models", []))
    active_cols = [
        c for c in prob_cols
        if c.replace("prob_", "", 1) not in exclude
    ]

    # Build model_preds dict for meta-learner
    active_model_names = [c.replace("prob_", "", 1) for c in active_cols]
    model_preds = {}
    for col, name in zip(active_cols, active_model_names):
        model_preds[name] = preds[col].values.copy()

    # Step 3: Apply pre-calibration to specified models
    if pre_calibrators:
        for model_name, cal in pre_calibrators.items():
            if model_name in model_preds and hasattr(cal, "transform"):
                model_preds[model_name] = cal.transform(model_preds[model_name])

    # Extract seed_diffs
    seed_diffs = preds["diff_seed_num"].values if "diff_seed_num" in preds.columns else np.zeros(len(preds))

    # Extract extra features (meta_features from config)
    extra_features = None
    meta_feature_names = ensemble_config.get("meta_features", [])
    if meta_feature_names:
        extra_features = {}
        for feat_name in meta_feature_names:
            if feat_name in preds.columns:
                extra_features[feat_name] = preds[feat_name].values

    # Step 4: Run meta-learner prediction
    probs = meta_learner.predict(
        model_preds,
        seed_diffs,
        extra_features=extra_features if extra_features else None,
    )

    # Step 5: Apply post-calibration
    if calibrator is not None and hasattr(calibrator, "transform"):
        probs = calibrator.transform(probs)

    # Step 6: Temperature scaling
    T = ensemble_config.get("temperature", 1.0)
    if T != 1.0:
        probs = temperature_scale(probs, T)

    # Step 7: Probability clipping
    clip_floor = ensemble_config.get("clip_floor", 0.0)
    if clip_floor > 0:
        probs = np.clip(probs, clip_floor, 1.0 - clip_floor)

    # Step 8: Availability adjustment
    av_strength = ensemble_config.get("availability_adjustment", 0.0)
    if av_strength > 0:
        probs = apply_availability_adjustment(probs, preds, ensemble_config, av_strength)

    # Step 9: Seed-proximity compression
    compression = ensemble_config.get("seed_compression", 0.0)
    threshold = ensemble_config.get("seed_compression_threshold", 4)
    if compression > 0:
        probs = apply_seed_compression(probs, preds, compression, threshold)

    preds = preds.copy()
    preds["prob_ensemble"] = probs
    return preds


def apply_availability_adjustment(
    probs: np.ndarray,
    preds: pd.DataFrame,
    ensemble_config: dict,
    strength: float,
) -> np.ndarray:
    """Adjust ensemble probabilities based on model availability.

    When some models have no prediction for certain games (NaN in prob_*
    columns), pulls ensemble predictions toward 0.5 proportional to the
    fraction of missing models and the strength parameter.

    Parameters
    ----------
    probs : np.ndarray
        Ensemble probabilities (from meta-learner).
    preds : pd.DataFrame
        Predictions DataFrame with prob_* columns for each model.
    ensemble_config : dict
        Ensemble configuration (needs exclude_models list).
    strength : float
        Adjustment strength (0.0 = no adjustment, 1.0 = full pull to 0.5).

    Returns
    -------
    np.ndarray
        Adjusted probabilities.
    """
    # Identify all prob_* columns, excluding prob_ensemble
    prob_cols = [
        c for c in preds.columns
        if c.startswith("prob_") and c != "prob_ensemble"
    ]

    # Filter out excluded models
    exclude = set(ensemble_config.get("exclude_models", []))
    active_cols = [
        c for c in prob_cols
        if c.replace("prob_", "", 1) not in exclude
    ]

    n_total_active = len(active_cols)
    if n_total_active == 0:
        return probs.copy()

    # Count non-NaN values per row across active model columns
    active_data = preds[active_cols]
    n_available = active_data.notna().sum(axis=1).values

    # Compute coverage and pull factor
    coverage_ratio = n_available / n_total_active
    pull_factor = strength * (1.0 - coverage_ratio)

    # Apply: adjusted = probs * (1 - pull_factor) + 0.5 * pull_factor
    adjusted = probs * (1.0 - pull_factor) + 0.5 * pull_factor
    return adjusted


def apply_seed_compression(
    probs: np.ndarray,
    preds: pd.DataFrame,
    compression: float,
    threshold: int,
) -> np.ndarray:
    """Compress toward 0.5 for close seed matchups.

    For matchups where abs(seed_diff) <= threshold, compresses
    the probability toward 0.5 by the compression factor:
    p_new = p * (1 - compression) + 0.5 * compression
    """
    probs = probs.copy()
    if "diff_seed_num" not in preds.columns:
        return probs

    seed_diffs = preds["diff_seed_num"].values
    close_mask = np.abs(seed_diffs) <= threshold

    probs[close_mask] = (
        probs[close_mask] * (1.0 - compression) + 0.5 * compression
    )
    return probs
