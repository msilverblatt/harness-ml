"""Sports-domain matchup generation for pairwise tournament matchups.

Generates all pairwise diff-feature matchups from team-season features
and seed data.  This is a sports-specific function that was extracted
from harness-core to the harness-sports plugin.
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd

from harnessml.core.runner.matchups import compute_interactions

logger = logging.getLogger(__name__)


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
    - Include TeamA, TeamB, season, diff_prior columns

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
        Columns include TeamA, TeamB, season, diff_prior, and
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
            "diff_prior": float(row_a.get("seed_num", 0)) - float(row_b.get("seed_num", 0)),
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
