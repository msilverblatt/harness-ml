"""Tests for matchup generation and prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harnessml.core.runner.matchups import compute_interactions, predict_all_matchups
from harnessml.sports.matchups import generate_pairwise_matchups
from harnessml.core.runner.schema import InteractionDef, ModelDef


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

    def test_diff_prior_included(self):
        """diff_prior column is included."""
        team_feats = _make_team_features(n_teams=3)
        seeds = _make_seeds([1, 2, 3])
        matchups = generate_pairwise_matchups(team_feats, seeds, season=2024)
        assert "diff_prior" in matchups.columns

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
            "diff_prior": [-1.0, -3.0],
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
            "diff_prior": [-1.0, -3.0],
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
            "diff_prior": [-1.0],
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
            "diff_prior": [-1.0],
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
            "diff_prior": [-1.0],
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
            "diff_prior": [-1.0],
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


# -----------------------------------------------------------------------
# Tests: compute_interactions
# -----------------------------------------------------------------------

class TestComputeInteractions:
    """Test interaction feature computation."""

    def test_multiply(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        interactions = {"a_times_b": InteractionDef(left="a", right="b", op="multiply")}
        result = compute_interactions(df, interactions)
        assert "a_times_b" in result.columns
        np.testing.assert_array_almost_equal(result["a_times_b"].values, [4.0, 10.0, 18.0])

    def test_add(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        interactions = {"a_plus_b": InteractionDef(left="a", right="b", op="add")}
        result = compute_interactions(df, interactions)
        assert "a_plus_b" in result.columns
        np.testing.assert_array_almost_equal(result["a_plus_b"].values, [5.0, 7.0, 9.0])

    def test_subtract(self):
        df = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [3.0, 5.0, 7.0]})
        interactions = {"a_minus_b": InteractionDef(left="a", right="b", op="subtract")}
        result = compute_interactions(df, interactions)
        assert "a_minus_b" in result.columns
        np.testing.assert_array_almost_equal(result["a_minus_b"].values, [7.0, 15.0, 23.0])

    def test_divide_with_zero(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 4.0]})
        interactions = {"a_div_b": InteractionDef(left="a", right="b", op="divide")}
        result = compute_interactions(df, interactions)
        assert result["a_div_b"].iloc[0] == 0.0  # zero-div handled
        assert result["a_div_b"].iloc[1] == 0.5

    def test_abs_diff(self):
        df = pd.DataFrame({"a": [1.0, 5.0, 3.0], "b": [4.0, 2.0, 3.0]})
        interactions = {"a_abs_b": InteractionDef(left="a", right="b", op="abs_diff")}
        result = compute_interactions(df, interactions)
        assert "a_abs_b" in result.columns
        np.testing.assert_array_almost_equal(result["a_abs_b"].values, [3.0, 3.0, 0.0])

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1.0]})
        interactions = {"x": InteractionDef(left="a", right="missing", op="multiply")}
        with pytest.raises(KeyError):
            compute_interactions(df, interactions)

    def test_missing_left_column_raises(self):
        df = pd.DataFrame({"b": [1.0]})
        interactions = {"x": InteractionDef(left="missing", right="b", op="multiply")}
        with pytest.raises(KeyError):
            compute_interactions(df, interactions)

    def test_multiple_interactions(self):
        df = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})
        interactions = {
            "prod": InteractionDef(left="a", right="b", op="multiply"),
            "total": InteractionDef(left="a", right="b", op="add"),
        }
        result = compute_interactions(df, interactions)
        assert "prod" in result.columns
        assert "total" in result.columns
        np.testing.assert_array_almost_equal(result["prod"].values, [8.0, 15.0])
        np.testing.assert_array_almost_equal(result["total"].values, [6.0, 8.0])

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        original_cols = list(df.columns)
        interactions = {"c": InteractionDef(left="a", right="b", op="multiply")}
        result = compute_interactions(df, interactions)
        # Original df unchanged
        assert list(df.columns) == original_cols
        assert "c" not in df.columns
        # Result has new column
        assert "c" in result.columns


class TestGenerateMatchupsWithInteractions:
    """Test generate_pairwise_matchups with interactions parameter."""

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
        # diff_adj_oe = 110-100 = 10, diff_adj_de = 90-95 = -5
        # net_eff = 10 - (-5) = 15
        assert matchups.iloc[0]["net_eff"] == pytest.approx(15.0)

    def test_no_interactions_unchanged(self):
        team_feats = _make_team_features(n_teams=3)
        seeds = _make_seeds([1, 2, 3])
        result_none = generate_pairwise_matchups(team_feats, seeds, season=2024)
        result_empty = generate_pairwise_matchups(
            team_feats, seeds, season=2024, interactions={},
        )
        assert list(result_none.columns) == list(result_empty.columns)
