"""Matchup generation and prediction for pairwise tournament matchups.

Generates all pairwise diff-feature matchups from team-season features
and runs trained models across them to produce probability predictions.
"""
from __future__ import annotations

import logging
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from easyml.core.runner.schema import InteractionDef, ModelDef
from easyml.core.runner.training import _is_regressor, _margin_to_prob

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


def generate_pairwise_matchups(
    team_features: pd.DataFrame,
    seeds: pd.DataFrame,
    season: int,
    feature_medians: dict | None = None,
    interactions: dict | None = None,
) -> pd.DataFrame:
    """Generate all pairwise matchup features for seeded teams.

    For each pair of seeded teams (A < B by seed) in a season:
    - Look up team-season features for both teams
    - Compute diff_* features (TeamA value - TeamB value)
    - Impute NaN with feature medians
    - Include TeamA, TeamB, season, diff_seed_num columns

    Parameters
    ----------
    team_features : pd.DataFrame
        Team-level features with 'team_id' and 'season' columns.
        All other numeric columns are treated as features.
    seeds : pd.DataFrame
        Seeding data with 'team_id', 'season', and 'seed_num' columns.
    season : int
        Season to generate matchups for.
    feature_medians : dict | None
        Optional mapping of feature_name -> median value for NaN imputation.
        If None, medians are computed from team_features.

    Returns
    -------
    pd.DataFrame
        Matchup DataFrame with N*(N-1)/2 rows for N seeded teams.
        Columns include TeamA, TeamB, season, diff_seed_num, and
        diff_{feature} for each numeric feature.
    """
    # Filter seeds for this season
    season_seeds = seeds[seeds["season"] == season].copy()
    if len(season_seeds) == 0:
        return pd.DataFrame()

    # Filter team features for this season
    season_feats = team_features[team_features["season"] == season].copy()
    if len(season_feats) == 0:
        return pd.DataFrame()

    # Merge seeds with team features
    merged = season_seeds.merge(season_feats, on=["team_id", "season"], how="left")

    # Identify numeric feature columns (exclude identifiers)
    exclude_cols = {"team_id", "season", "seed_num"}
    feature_cols = [
        c for c in merged.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(merged[c])
    ]

    # Compute medians for imputation if not provided
    if feature_medians is None:
        feature_medians = {}
        for col in feature_cols:
            median_val = season_feats[col].median() if col in season_feats.columns else 0.0
            feature_medians[col] = float(median_val) if not np.isnan(median_val) else 0.0

    # Generate all pairs
    team_ids = sorted(merged["team_id"].unique())
    rows = []

    # Index merged by team_id for fast lookup
    team_data = merged.set_index("team_id")

    for team_a, team_b in combinations(team_ids, 2):
        if team_a not in team_data.index or team_b not in team_data.index:
            continue

        row_a = team_data.loc[team_a]
        row_b = team_data.loc[team_b]

        # Handle case where loc returns a DataFrame (duplicate team_ids)
        if isinstance(row_a, pd.DataFrame):
            row_a = row_a.iloc[0]
        if isinstance(row_b, pd.DataFrame):
            row_b = row_b.iloc[0]

        matchup = {
            "TeamA": team_a,
            "TeamB": team_b,
            "season": season,
            "diff_seed_num": float(row_a.get("seed_num", 0)) - float(row_b.get("seed_num", 0)),
        }

        # Compute diff features
        for col in feature_cols:
            val_a = row_a.get(col, np.nan)
            val_b = row_b.get(col, np.nan)

            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                diff = float(val_a) - float(val_b)
            else:
                diff = np.nan

            # Impute NaN
            if np.isnan(diff):
                diff = feature_medians.get(col, 0.0)

            matchup[f"diff_{col}"] = diff

        rows.append(matchup)

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    if interactions:
        result_df = compute_interactions(result_df, interactions)

    return result_df


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
