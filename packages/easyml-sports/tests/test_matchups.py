"""Tests for sports matchup generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.sports.matchups import generate_pairwise_matchups
from easyml.core.runner.schema import InteractionDef


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_team_features(n_teams: int = 4, season: int = 2024) -> pd.DataFrame:
    """Create synthetic team features."""
    rng = np.random.default_rng(42)
    data = {
        "team_id": list(range(1, n_teams + 1)),
        "season": [season] * n_teams,
        "adj_oe": rng.standard_normal(n_teams) * 5 + 100,
        "adj_de": rng.standard_normal(n_teams) * 5 + 95,
        "win_pct": rng.uniform(0.4, 0.9, n_teams),
    }
    return pd.DataFrame(data)


def _make_seeds(team_ids: list[int], season: int = 2024) -> pd.DataFrame:
    """Create seeding data."""
    return pd.DataFrame({
        "team_id": team_ids,
        "season": [season] * len(team_ids),
        "seed_num": list(range(1, len(team_ids) + 1)),
    })


# -----------------------------------------------------------------------
# Tests: generate_pairwise_matchups (sports plugin version)
# -----------------------------------------------------------------------

class TestGeneratePairwiseMatchups:
    """Test matchup generation produces correct pairwise structure."""

    def test_n_choose_2_rows(self):
        """N teams produce N*(N-1)/2 matchup rows."""
        n_teams = 4
        team_feats = _make_team_features(n_teams=n_teams)
        seeds = _make_seeds(list(range(1, n_teams + 1)))
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        expected = n_teams * (n_teams - 1) // 2
        assert len(matchups) == expected

    def test_diff_features_computed(self):
        """Diff features are computed as TeamA - TeamB values."""
        team_feats = pd.DataFrame({
            "team_id": [1, 2],
            "season": [2024, 2024],
            "adj_oe": [110.0, 100.0],
            "win_pct": [0.8, 0.6],
        })
        seeds = _make_seeds([1, 2])
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)

        assert len(matchups) == 1
        assert "diff_adj_oe" in matchups.columns
        assert "diff_win_pct" in matchups.columns

        row = matchups.iloc[0]
        assert row["diff_adj_oe"] == pytest.approx(10.0)
        assert row["diff_win_pct"] == pytest.approx(0.2)

    def test_diff_seed_num_included(self):
        """diff_seed_num column is included."""
        team_feats = _make_team_features(n_teams=3)
        seeds = _make_seeds([1, 2, 3])
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        assert "diff_seed_num" in matchups.columns

    def test_empty_season_returns_empty(self):
        """Empty team features for the season returns empty DataFrame."""
        team_feats = _make_team_features(n_teams=4, season=2023)
        seeds = _make_seeds([1, 2, 3, 4], season=2024)
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        assert len(matchups) == 0

    def test_interactions_applied(self):
        team_feats = pd.DataFrame({
            "team_id": [1, 2],
            "season": [2024, 2024],
            "adj_oe": [110.0, 100.0],
            "adj_de": [90.0, 95.0],
        })
        seeds = _make_seeds([1, 2])
        interactions = {
            "net_eff": InteractionDef(left="diff_adj_oe", right="diff_adj_de", op="subtract"),
        }
        matchups = generate_pairwise_matchups(
            team_feats, seeds, season=2024, interactions=interactions,
        )
        assert "net_eff" in matchups.columns
        assert matchups.iloc[0]["net_eff"] == pytest.approx(15.0)
