"""Ensemble post-processing pipeline.

Applies the full chain: model filtering, pre-calibration, meta-learner,
post-calibration, temperature scaling, clipping, availability adjustment,
and prior compression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from harnessml.core.runner.training.calibration import temperature_scale
from harnessml.core.runner.training.meta_learner import StackedEnsemble


def apply_ensemble_postprocessing(
    preds: pd.DataFrame,
    meta_learner: StackedEnsemble,
    calibrator,
    ensemble_config: dict,
    entity_features: pd.DataFrame | None = None,
    pre_calibrators: dict | None = None,
) -> pd.DataFrame:
    """Apply full 9-step ensemble post-processing pipeline.

    Steps:
    1. Extract base model prob columns (prob_*), excluding prob_logreg_seed
    2. Filter out excluded models (ensemble_config["exclude_models"])
    3. Apply pre-calibration to specified models
    4. Run meta-learner prediction (using prior_diffs from diff_prior column)
    5. Apply post-calibration
    6. Temperature scaling
    7. Probability clipping
    8. Availability adjustment (if strength > 0)
    9. Prior-proximity compression (if configured and > 0)

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

    # Extract prior_diffs
    prior_diffs = preds["diff_prior"].values if "diff_prior" in preds.columns else np.zeros(len(preds))

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
        prior_diffs,
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

    # Step 8: Logit adjustments
    logit_adjustments = ensemble_config.get("logit_adjustments", [])
    if logit_adjustments:
        probs = apply_logit_adjustments(probs, preds, logit_adjustments)

    # Step 9: Prior-proximity compression
    compression = ensemble_config.get("prior_compression", 0.0)
    threshold = ensemble_config.get("prior_compression_threshold", 4)
    if compression > 0:
        probs = apply_prior_compression(probs, preds, compression, threshold)

    preds = preds.copy()
    preds["prob_ensemble"] = probs
    return preds


def apply_logit_adjustments(
    probs: np.ndarray,
    preds: pd.DataFrame,
    adjustments: list[dict],
) -> np.ndarray:
    """Apply one or more logit-space adjustments to ensemble probabilities.

    Each adjustment is a dict with keys: columns, strength, default, mode.

    Modes:
      - paired: 2 columns (entity A value, entity B value), each 0-1.
        Penalty = strength * (1 - value). Applied as
        logit -= penalty_a, logit += penalty_b.
      - diff: 1 column (signed difference). Applied as
        logit += strength * value.

    Parameters
    ----------
    probs : np.ndarray
        Ensemble probabilities.
    preds : pd.DataFrame
        Predictions DataFrame with the referenced columns.
    adjustments : list[dict]
        Each entry has: columns (list[str]), strength (float),
        default (float), mode ("paired" | "diff").

    Returns
    -------
    np.ndarray
        Adjusted probabilities.
    """
    eps = 1e-7
    clipped = np.clip(probs, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped))

    for adj in adjustments:
        columns = adj.get("columns", [])
        strength = adj.get("strength", 0.0)
        default = adj.get("default", 1.0)
        mode = adj.get("mode", "paired")

        if strength == 0.0:
            continue

        if mode == "paired" and len(columns) == 2:
            a_col, b_col = columns
            if a_col not in preds.columns or b_col not in preds.columns:
                continue
            val_a = preds[a_col].values.astype(np.float64)
            val_b = preds[b_col].values.astype(np.float64)
            val_a = np.where(np.isnan(val_a), default, val_a)
            val_b = np.where(np.isnan(val_b), default, val_b)
            logits -= strength * (1.0 - val_a)
            logits += strength * (1.0 - val_b)

        elif mode == "diff" and len(columns) == 1:
            col = columns[0]
            if col not in preds.columns:
                continue
            val = preds[col].values.astype(np.float64)
            val = np.where(np.isnan(val), default, val)
            logits += strength * val

    return 1.0 / (1.0 + np.exp(-logits))


def apply_prior_compression(
    probs: np.ndarray,
    preds: pd.DataFrame,
    compression: float,
    threshold: int,
) -> np.ndarray:
    """Compress toward 0.5 for close-prior matchups.

    For matchups where abs(prior_diff) <= threshold, compresses
    the probability toward 0.5 by the compression factor:
    p_new = p * (1 - compression) + 0.5 * compression
    """
    probs = probs.copy()
    if "diff_prior" not in preds.columns:
        return probs

    prior_diffs = preds["diff_prior"].values
    close_mask = np.abs(prior_diffs) <= threshold

    probs[close_mask] = (
        probs[close_mask] * (1.0 - compression) + 0.5 * compression
    )
    return probs
