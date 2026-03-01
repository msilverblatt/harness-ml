"""Tests for matchup generation and prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.runner.matchups import generate_pairwise_matchups, predict_all_matchups
from easyml.runner.schema import ModelDef


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


class _MockClassifier:
    """Mock classifier that returns probabilities."""

    def predict_proba(self, X):
        n = X.shape[0]
        return np.full(n, 0.6)


class _MockRegressor:
    """Mock regressor that returns margins."""

    def predict_margin(self, X):
        n = X.shape[0]
        return np.full(n, 3.0)


# -----------------------------------------------------------------------
# Tests: generate_pairwise_matchups
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

    def test_six_teams(self):
        """6 teams produce 15 matchup rows."""
        n_teams = 6
        team_feats = _make_team_features(n_teams=n_teams)
        seeds = _make_seeds(list(range(1, n_teams + 1)))
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        expected = n_teams * (n_teams - 1) // 2
        assert len(matchups) == expected
        assert expected == 15

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

        # Team 1 (seed 1) vs Team 2 (seed 2): diff = Team1 - Team2
        row = matchups.iloc[0]
        assert row["diff_adj_oe"] == pytest.approx(10.0)
        assert row["diff_win_pct"] == pytest.approx(0.2)

    def test_diff_seed_num_included(self):
        """diff_seed_num column is included."""
        team_feats = _make_team_features(n_teams=3)
        seeds = _make_seeds([1, 2, 3])
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        assert "diff_seed_num" in matchups.columns

    def test_season_column_included(self):
        """season column is present."""
        team_feats = _make_team_features(n_teams=2)
        seeds = _make_seeds([1, 2])
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        assert "season" in matchups.columns
        assert all(matchups["season"] == 2024)

    def test_nan_imputation_with_medians(self):
        """NaN values are imputed with provided medians."""
        team_feats = pd.DataFrame({
            "team_id": [1, 2],
            "season": [2024, 2024],
            "adj_oe": [110.0, np.nan],
        })
        seeds = _make_seeds([1, 2])
        medians = {"adj_oe": 100.0}
        matchups = generate_pairwise_matchups(
            team_feats, seeds, season=2024, feature_medians=medians
        )

        # NaN diff should be imputed with the median value
        assert len(matchups) == 1
        assert not np.isnan(matchups.iloc[0]["diff_adj_oe"])

    def test_nan_imputation_auto_medians(self):
        """NaN values are imputed with auto-computed medians when none provided."""
        team_feats = pd.DataFrame({
            "team_id": [1, 2, 3],
            "season": [2024, 2024, 2024],
            "adj_oe": [110.0, np.nan, 100.0],
        })
        seeds = _make_seeds([1, 2, 3])
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)

        # All diff values should be non-NaN
        assert not matchups["diff_adj_oe"].isna().any()

    def test_empty_season_returns_empty(self):
        """Empty team features for the season returns empty DataFrame."""
        team_feats = _make_team_features(n_teams=4, season=2023)
        seeds = _make_seeds([1, 2, 3, 4], season=2024)
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        assert len(matchups) == 0

    def test_empty_seeds_returns_empty(self):
        """No seeds for the season returns empty DataFrame."""
        team_feats = _make_team_features(n_teams=4, season=2024)
        seeds = _make_seeds([1, 2], season=2023)
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        assert len(matchups) == 0

    def test_team_a_team_b_columns(self):
        """TeamA and TeamB columns are present."""
        team_feats = _make_team_features(n_teams=3)
        seeds = _make_seeds([1, 2, 3])
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        assert "TeamA" in matchups.columns
        assert "TeamB" in matchups.columns


# -----------------------------------------------------------------------
# Tests: predict_all_matchups
# -----------------------------------------------------------------------

class TestPredictAllMatchups:
    """Test prediction across all matchups."""

    def test_classifier_predictions(self):
        """Classifier model adds prob_{name} column."""
        matchups = pd.DataFrame({
            "TeamA": [1, 2],
            "TeamB": [3, 4],
            "season": [2024, 2024],
            "diff_seed_num": [-1.0, -3.0],
            "diff_x": [0.5, -0.5],
        })
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
        )
        models = {
            "logreg": (_MockClassifier(), ["diff_x"], {}),
        }
        model_defs = {"logreg": model_def}

        result = predict_all_matchups(matchups, models, model_defs)
        assert "prob_logreg" in result.columns
        assert len(result) == 2
        assert np.all(result["prob_logreg"].values >= 0)
        assert np.all(result["prob_logreg"].values <= 1)

    def test_regressor_cdf_conversion(self):
        """Regressor model predictions are converted via CDF."""
        matchups = pd.DataFrame({
            "TeamA": [1, 2],
            "TeamB": [3, 4],
            "season": [2024, 2024],
            "diff_seed_num": [-1.0, -3.0],
            "diff_x": [0.5, -0.5],
        })
        model_def = ModelDef(
            type="xgboost",
            features=["diff_x"],
            mode="regressor",
            cdf_scale=5.0,
        )
        models = {
            "xgb_reg": (_MockRegressor(), ["diff_x"], {"cdf_scale": 5.0}),
        }
        model_defs = {"xgb_reg": model_def}

        result = predict_all_matchups(matchups, models, model_defs)
        assert "prob_xgb_reg" in result.columns
        # Margin=3.0, scale=5.0 -> CDF(0.6) > 0.5
        assert np.all(result["prob_xgb_reg"].values > 0.5)

    def test_multi_seed_averaging(self):
        """Multi-seed model predictions are averaged."""
        matchups = pd.DataFrame({
            "TeamA": [1],
            "TeamB": [2],
            "season": [2024],
            "diff_seed_num": [-1.0],
            "diff_x": [0.5],
        })
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
            n_seeds=3,
        )
        # List of models = multi-seed
        models = {
            "logreg_multi": (
                [_MockClassifier(), _MockClassifier(), _MockClassifier()],
                ["diff_x"],
                {},
            ),
        }
        model_defs = {"logreg_multi": model_def}

        result = predict_all_matchups(matchups, models, model_defs)
        assert "prob_logreg_multi" in result.columns
        # All return 0.6, so average should be 0.6
        assert result.iloc[0]["prob_logreg_multi"] == pytest.approx(0.6)

    def test_nan_imputation(self):
        """NaN in test features are imputed."""
        matchups = pd.DataFrame({
            "TeamA": [1],
            "TeamB": [2],
            "season": [2024],
            "diff_seed_num": [-1.0],
            "diff_x": [np.nan],
        })
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
        )
        models = {
            "logreg": (_MockClassifier(), ["diff_x"], {}),
        }
        model_defs = {"logreg": model_def}
        medians = {"diff_x": 0.0}

        result = predict_all_matchups(
            matchups, models, model_defs, feature_medians=medians,
        )
        assert "prob_logreg" in result.columns

    def test_missing_model_def_skipped(self):
        """Model without ModelDef is skipped."""
        matchups = pd.DataFrame({
            "TeamA": [1],
            "TeamB": [2],
            "season": [2024],
            "diff_seed_num": [-1.0],
            "diff_x": [0.5],
        })
        models = {
            "unknown_model": (_MockClassifier(), ["diff_x"], {}),
        }
        model_defs = {}  # No defs

        result = predict_all_matchups(matchups, models, model_defs)
        assert "prob_unknown_model" not in result.columns

    def test_preserves_original_columns(self):
        """Result preserves original matchup columns."""
        matchups = pd.DataFrame({
            "TeamA": [1],
            "TeamB": [2],
            "season": [2024],
            "diff_seed_num": [-1.0],
            "diff_x": [0.5],
            "diff_y": [1.0],
        })
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
        )
        models = {"logreg": (_MockClassifier(), ["diff_x"], {})}
        model_defs = {"logreg": model_def}

        result = predict_all_matchups(matchups, models, model_defs)
        assert "TeamA" in result.columns
        assert "diff_y" in result.columns
