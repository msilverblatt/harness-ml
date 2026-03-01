"""Tests for PairwiseFeatureBuilder — diff, ratio, and combined methods."""
import pandas as pd
from easyml.features.pairwise import PairwiseFeatureBuilder


def test_pairwise_diff():
    entity_df = pd.DataFrame(
        {
            "entity_id": [1, 2, 3],
            "period_id": [2025, 2025, 2025],
            "scoring_margin": [10.0, 5.0, -3.0],
            "win_pct": [0.8, 0.6, 0.3],
        }
    )
    matchups = pd.DataFrame(
        {
            "entity_a_id": [1, 2],
            "entity_b_id": [2, 3],
            "period_id": [2025, 2025],
        }
    )
    builder = PairwiseFeatureBuilder(methods=["diff"])
    result = builder.build(
        entity_df, matchups, feature_columns=["scoring_margin", "win_pct"]
    )
    assert "diff_scoring_margin" in result.columns
    assert result.iloc[0]["diff_scoring_margin"] == 5.0
    assert "diff_win_pct" in result.columns


def test_pairwise_ratio():
    entity_df = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "period_id": [2025, 2025],
            "ppg": [80.0, 60.0],
        }
    )
    matchups = pd.DataFrame(
        {
            "entity_a_id": [1],
            "entity_b_id": [2],
            "period_id": [2025],
        }
    )
    builder = PairwiseFeatureBuilder(methods=["ratio"])
    result = builder.build(entity_df, matchups, feature_columns=["ppg"])
    assert "ratio_ppg" in result.columns
    assert abs(result.iloc[0]["ratio_ppg"] - (80.0 / 60.0)) < 1e-10


def test_pairwise_both():
    entity_df = pd.DataFrame(
        {
            "entity_id": [1, 2],
            "period_id": [2025, 2025],
            "score": [10.0, 5.0],
        }
    )
    matchups = pd.DataFrame(
        {
            "entity_a_id": [1],
            "entity_b_id": [2],
            "period_id": [2025],
        }
    )
    builder = PairwiseFeatureBuilder(methods=["diff", "ratio"])
    result = builder.build(entity_df, matchups, feature_columns=["score"])
    assert "diff_score" in result.columns
    assert "ratio_score" in result.columns
