"""End-to-end integration tests.

These tests exercise the full chain of harness-data functionality
using realistic data and workflows. They catch integration issues
that unit tests miss — wrong interfaces, broken wiring, half-assed
implementations that pass isolated tests but fail in combination.

Run after every few tasks to verify nothing is fake.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harness.data.sources.protocol import SourceConfig, SourceMetadata
from harness.data.sources.file import FileSource
from harness.data.sources.registry import SourceRegistry
from harness.data.sources.freshness import FreshnessTracker
from harness.data.expressions.engine import ExpressionEngine
from harness.data.expressions.registry import FunctionRegistry
from harness.data.expressions.validator import ExpressionValidator
from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig


# ---------------------------------------------------------------------------
# Realistic test data
# ---------------------------------------------------------------------------

@pytest.fixture
def sports_data(tmp_path):
    """Create a realistic sports dataset with multiple CSVs."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    # Team stats per season
    teams = pd.DataFrame({
        "team_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "season": [2023] * 5 + [2024] * 5,
        "wins": [20, 15, 25, 10, 18, 22, 18, 23, 12, 20],
        "losses": [10, 15, 5, 20, 12, 8, 12, 7, 18, 10],
        "points_per_game": [78.5, 72.3, 85.1, 65.0, 74.2, 80.1, 75.0, 83.5, 67.2, 76.8],
        "opp_points_per_game": [72.1, 74.5, 68.3, 78.9, 71.0, 70.5, 73.2, 69.1, 76.5, 72.3],
        "rating": [85.5, 72.3, 91.0, 58.2, 76.1, 88.2, 75.1, 89.5, 60.5, 78.3],
    })
    teams_path = raw_dir / "teams.csv"
    teams.to_csv(teams_path, index=False)

    # Game results
    games = pd.DataFrame({
        "game_id": range(1, 21),
        "season": [2023] * 10 + [2024] * 10,
        "team_a_id": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "team_b_id": [2, 3, 4, 5, 1, 3, 4, 5, 1, 2, 2, 3, 4, 5, 1, 3, 4, 5, 1, 2],
        "score_a": [80, 65, 90, 55, 75, 85, 70, 88, 60, 72, 82, 68, 92, 58, 78, 87, 73, 86, 62, 74],
        "score_b": [72, 78, 62, 80, 70, 68, 82, 65, 85, 68, 75, 80, 60, 82, 72, 65, 85, 68, 88, 70],
        "neutral_site": [False] * 15 + [True] * 5,
    })
    games_path = raw_dir / "games.csv"
    games.to_csv(games_path, index=False)

    return {
        "tmp_path": tmp_path,
        "teams_path": str(teams_path),
        "games_path": str(games_path),
        "teams_df": teams,
        "games_df": games,
    }


# ---------------------------------------------------------------------------
# E2E: Source loading pipeline
# ---------------------------------------------------------------------------

class TestE2ESourcePipeline:
    """Test loading data from files, registering sources, tracking freshness."""

    def test_load_register_and_retrieve(self, sports_data):
        """Full flow: load CSV → register source → retrieve from registry → load again."""
        tmp_path = sports_data["tmp_path"]

        # Load the file
        source = FileSource()
        config = SourceConfig(name="teams", path=sports_data["teams_path"])
        df = source.load(config)
        assert len(df) == 10
        assert "rating" in df.columns

        # Register it
        registry = SourceRegistry(tmp_path)
        registry.add(config)

        # Retrieve and reload
        retrieved = registry.get("teams")
        assert retrieved is not None
        df2 = source.load(retrieved)
        assert df.equals(df2), "Reloaded data should be identical"

    def test_multiple_sources_with_freshness(self, sports_data):
        """Register multiple sources and track freshness."""
        tmp_path = sports_data["tmp_path"]
        registry = SourceRegistry(tmp_path)
        tracker = FreshnessTracker(tmp_path / ".freshness.json")

        # Register both sources
        registry.add(SourceConfig(name="teams", source_type="file", path=sports_data["teams_path"]))
        registry.add(SourceConfig(name="games", source_type="file", path=sports_data["games_path"]))

        assert len(registry.list_all()) == 2

        # Record fetch for teams only
        tracker.record_fetch("teams", row_count=10)

        # teams is fresh, games is stale (never fetched)
        assert not tracker.is_stale("teams", "daily")
        assert tracker.is_stale("games", "daily")

        # Verify freshness info
        info = tracker.get_info("teams")
        assert info["row_count"] == 10

    def test_source_metadata_from_loaded_data(self, sports_data):
        """Verify SourceMetadata.from_dataframe captures real column info."""
        source = FileSource()
        config = SourceConfig(name="teams", path=sports_data["teams_path"])
        df = source.load(config)

        meta = SourceMetadata.from_dataframe("teams", "file", df)
        assert meta.row_count == 10
        assert "team_id" in meta.columns
        assert "rating" in meta.columns
        assert meta.column_types["rating"] == "float64"
        assert meta.column_types["wins"] == "int64"

    def test_registry_persistence_across_instances(self, sports_data):
        """Registry survives process restart (new instance reads from YAML)."""
        tmp_path = sports_data["tmp_path"]

        reg1 = SourceRegistry(tmp_path)
        reg1.add(SourceConfig(name="teams", source_type="file", path=sports_data["teams_path"]))
        reg1.add(SourceConfig(name="games", source_type="file", path=sports_data["games_path"]))

        # Simulate process restart
        reg2 = SourceRegistry(tmp_path)
        assert len(reg2.list_all()) == 2
        teams = reg2.get("teams")
        assert teams.path == sports_data["teams_path"]


# ---------------------------------------------------------------------------
# E2E: Expression engine with real data
# ---------------------------------------------------------------------------

class TestE2EExpressionEngine:
    """Test expression evaluation against realistic DataFrames."""

    def test_derived_columns_from_sports_data(self, sports_data):
        """Compute realistic derived features from sports data."""
        engine = ExpressionEngine()
        df = sports_data["games_df"]

        # Score differential
        result = engine.evaluate(df, "score_a - score_b")
        assert len(result) == 20
        assert result.iloc[0] == 8  # 80 - 72

        # Total score
        result = engine.evaluate(df, "score_a + score_b")
        assert result.iloc[0] == 152  # 80 + 72

        # Home win indicator
        result = engine.evaluate(df, "where(score_a > score_b, 1, 0)")
        assert result.dtype in [np.int64, np.float64, int, float]
        home_wins = int(result.sum())
        assert 0 < home_wins < 20, f"Expected some but not all home wins, got {home_wins}"

    def test_statistical_functions_on_real_data(self, sports_data):
        """Z-score, rank_pct on actual team ratings."""
        engine = ExpressionEngine()
        df = sports_data["teams_df"]

        # Z-score of ratings
        zscores = engine.evaluate(df, "zscore(rating)")
        assert abs(zscores.mean()) < 1e-10, "Z-score mean should be ~0"
        assert abs(zscores.std() - 1.0) < 0.15, "Z-score std should be ~1"

        # Rank percentile
        ranks = engine.evaluate(df, "rank_pct(rating)")
        assert ranks.min() > 0
        assert ranks.max() <= 1.0
        # Best rated team should have high rank
        best_idx = df["rating"].idxmax()
        assert ranks.iloc[best_idx] >= 0.8

    def test_safe_div_prevents_real_division_errors(self, sports_data):
        """safe_div handles zero denominators in real data."""
        engine = ExpressionEngine()
        df = sports_data["teams_df"].copy()
        # Simulate a team with 0 losses
        df.loc[df["team_id"] == 3, "losses"] = 0

        result = engine.evaluate(df, "safe_div(wins, losses)")
        # Team 3 with 0 losses should get 0, not inf/nan
        team3_rows = df[df["team_id"] == 3].index
        for idx in team3_rows:
            assert result.iloc[idx] == 0.0, f"Expected 0.0 for 0 losses, got {result.iloc[idx]}"

        # Other teams should get actual win/loss ratio
        team1_2023 = df[(df["team_id"] == 1) & (df["season"] == 2023)].index[0]
        assert result.iloc[team1_2023] == 2.0  # 20/10

    def test_chained_expression_produces_correct_results(self, sports_data):
        """Multi-function expression on real data."""
        engine = ExpressionEngine()
        df = sports_data["teams_df"]

        # Win percentage
        result = engine.evaluate(df, "safe_div(wins, wins + losses)")
        assert all(0 <= r <= 1 for r in result), "Win pct should be 0-1"
        # Team 3 season 2023: 25/(25+5) = 0.833...
        team3_2023 = df[(df["team_id"] == 3) & (df["season"] == 2023)].index[0]
        assert abs(result.iloc[team3_2023] - 0.8333) < 0.01

    def test_expression_with_comparison_operators(self, sports_data):
        """Expressions using >, <, == on real data."""
        engine = ExpressionEngine()
        df = sports_data["games_df"]

        # Games where team A won by 10+
        blowouts = engine.evaluate(df, "where(score_a - score_b > 10, 1, 0)")
        assert isinstance(blowouts, pd.Series)
        assert blowouts.sum() > 0  # Some blowouts should exist


# ---------------------------------------------------------------------------
# E2E: Expression validation catches real mistakes
# ---------------------------------------------------------------------------

class TestE2EExpressionValidation:
    """Test that validation catches the kinds of mistakes agents actually make."""

    def test_catches_misspelled_column(self):
        """Agent types 'raitng' instead of 'rating'."""
        validator = ExpressionValidator()
        schema = {
            "columns": ["team_id", "season", "wins", "losses", "rating"],
            "column_types": {"rating": "float64", "wins": "int64"},
        }
        result = validator.validate("raitng * 2", schema)
        assert not result.is_valid
        assert "raitng" in result.errors[0]

    def test_catches_column_from_wrong_table(self):
        """Agent uses a column from games table in a teams expression."""
        validator = ExpressionValidator()
        schema = {
            "columns": ["team_id", "season", "wins", "losses", "rating"],
            "column_types": {},
        }
        result = validator.validate("score_a - score_b", schema)
        assert not result.is_valid

    def test_catches_nonexistent_function(self):
        """Agent invents a function that doesn't exist."""
        validator = ExpressionValidator()
        schema = {"columns": ["rating"], "column_types": {}}
        result = validator.validate("normalize(rating)", schema)
        assert not result.is_valid
        assert "normalize" in result.errors[0]

    def test_approves_valid_complex_expression(self):
        """Realistic multi-function expression should validate."""
        validator = ExpressionValidator()
        schema = {
            "columns": ["wins", "losses", "rating"],
            "column_types": {"wins": "int64", "losses": "int64", "rating": "float64"},
        }
        result = validator.validate("abs(zscore(rating))", schema)
        assert result.is_valid, f"Should be valid, errors: {result.errors}"

    def test_suggestion_contains_available_columns(self):
        """When a column is wrong, suggestion should list available ones."""
        validator = ExpressionValidator()
        schema = {
            "columns": ["wins", "losses", "rating"],
            "column_types": {},
        }
        result = validator.validate("momentum * 2", schema)
        assert not result.is_valid
        # Suggestion should help the agent find the right column
        assert result.suggestion is not None
        assert len(result.suggestion) > 0


# ---------------------------------------------------------------------------
# E2E: All transform steps on real data
# ---------------------------------------------------------------------------

@pytest.fixture
def team_history():
    """Multi-season team data for windowed operation testing."""
    return pd.DataFrame({
        "team_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "season": [2022, 2023, 2024, 2022, 2023, 2024, 2022, 2023, 2024],
        "wins": [18, 20, 22, 12, 15, 18, 25, 23, 21],
        "rating": [78.0, 82.0, 85.5, 65.0, 70.0, 75.0, 90.0, 88.0, 86.0],
    })


class TestE2ETransformSteps:
    """Verify every transform step produces correct values on real data."""

    def test_derive_with_expression_engine(self, team_history):
        """Derive uses the expression engine — not just string eval."""
        engine = TransformEngine()
        result = engine.apply_step(team_history, StepConfig(op="derive", params={
            "columns": {"win_over_15": "wins - 15", "rating_z": "zscore(rating)"}
        }))
        assert result["win_over_15"].iloc[0] == 3  # 18 - 15
        assert abs(result["rating_z"].mean()) < 1e-10

    def test_rolling_mean_correct_values(self, team_history):
        """Rolling mean partitioned by team produces correct numbers."""
        engine = TransformEngine()
        result = engine.apply_step(team_history, StepConfig(op="rolling", params={
            "keys": ["team_id"], "order_by": "season", "window": 2,
            "aggs": {"rating_ma2": "rating:mean"},
        }))
        t1 = list(result[result["team_id"] == 1]["rating_ma2"])
        assert abs(t1[1] - 80.0) < 0.01, f"(78+82)/2 = 80, got {t1[1]}"
        assert abs(t1[2] - 83.75) < 0.01, f"(82+85.5)/2 = 83.75, got {t1[2]}"

    def test_lag_correct_values(self, team_history):
        """Lag produces previous season's value per team."""
        engine = TransformEngine()
        result = engine.apply_step(team_history, StepConfig(op="lag", params={
            "keys": ["team_id"], "order_by": "season",
            "columns": {"prev_wins": "wins:1"},
        }))
        t1 = list(result[result["team_id"] == 1]["prev_wins"])
        assert pd.isna(t1[0]), "First season has no previous"
        assert t1[1] == 18, f"2023 prev should be 18, got {t1[1]}"
        assert t1[2] == 20, f"2024 prev should be 20, got {t1[2]}"

    def test_diff_correct_values(self, team_history):
        """Diff computes year-over-year change."""
        engine = TransformEngine()
        result = engine.apply_step(team_history, StepConfig(op="diff", params={
            "keys": ["team_id"], "order_by": "season",
            "columns": {"win_change": "wins:1"},
        }))
        t1 = list(result[result["team_id"] == 1]["win_change"])
        assert pd.isna(t1[0])
        assert t1[1] == 2, f"20-18=2, got {t1[1]}"
        assert t1[2] == 2, f"22-20=2, got {t1[2]}"

    def test_trend_direction_correct(self, team_history):
        """Trend detects increasing vs decreasing ratings."""
        engine = TransformEngine()
        result = engine.apply_step(team_history, StepConfig(op="trend", params={
            "keys": ["team_id"], "order_by": "season", "window": 3,
            "columns": {"rating_trend": "rating"},
        }))
        # Team 1: 78→82→85.5 (increasing)
        t1_trend = result[result["team_id"] == 1]["rating_trend"].iloc[2]
        assert t1_trend > 0, f"Team 1 trend should be positive, got {t1_trend}"
        # Team 3: 90→88→86 (decreasing)
        t3_trend = result[result["team_id"] == 3]["rating_trend"].iloc[2]
        assert t3_trend < 0, f"Team 3 trend should be negative, got {t3_trend}"

    def test_fill_median_correct(self):
        """Fill with median produces the actual median value."""
        engine = TransformEngine()
        df = pd.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        result = engine.apply_step(df, StepConfig(op="fill", params={"strategy": "median"}))
        assert result["a"].iloc[1] == 3.0, "Median of [1,3,5] is 3"
        assert result["a"].isna().sum() == 0

    def test_rank_ordering_correct(self, team_history):
        """Best rated team gets rank 1 (descending)."""
        engine = TransformEngine()
        result = engine.apply_step(team_history, StepConfig(op="rank", params={
            "columns": {"r": "rating"}, "ascending": False,
        }))
        best_idx = team_history["rating"].idxmax()
        assert result.loc[best_idx, "r"] == 1.0

    def test_ewm_produces_smooth_values(self, team_history):
        """EWM mean should be between min and max of the series."""
        engine = TransformEngine()
        result = engine.apply_step(team_history, StepConfig(op="ewm", params={
            "keys": ["team_id"], "order_by": "season", "span": 2,
            "aggs": {"ewm_rating": "rating:mean"},
        }))
        for tid in [1, 2, 3]:
            vals = result[result["team_id"] == tid]["ewm_rating"]
            orig = team_history[team_history["team_id"] == tid]["rating"]
            assert vals.min() >= orig.min() - 1, "EWM should be within range"
            assert vals.max() <= orig.max() + 1, "EWM should be within range"

    def test_complex_pipeline_realistic_workflow(self, team_history):
        """Realistic multi-step pipeline: sort → filter → derive → head."""
        engine = TransformEngine()
        result = engine.run_pipeline(team_history, [
            StepConfig(op="filter", params={"expr": "season == 2024"}),
            StepConfig(op="derive", params={"columns": {"win_pct": "safe_div(wins, 30)"}}),
            StepConfig(op="sort", params={"by": "rating", "ascending": False}),
            StepConfig(op="head", params={"n": 2}),
        ])
        assert len(result) == 2
        assert result["rating"].iloc[0] > result["rating"].iloc[1]
        assert all(0 <= wp <= 1 for wp in result["win_pct"])


class TestE2ETransformEngine:
    """Test transform engine behavior on realistic DataFrames."""

    def test_filter_then_select_real_data(self, sports_data):
        """Filter to one season, select specific columns."""
        engine = TransformEngine()
        df = sports_data["teams_df"]

        result = engine.run_pipeline(df, [
            StepConfig(op="filter", params={"expr": "season == 2024"}),
            StepConfig(op="select", params={"columns": ["team_id", "wins", "rating"]}),
        ])

        assert len(result) == 5, f"Expected 5 teams in 2024, got {len(result)}"
        assert list(result.columns) == ["team_id", "wins", "rating"]
        assert "losses" not in result.columns
        assert "season" not in result.columns

    def test_filter_produces_correct_subset(self, sports_data):
        """Verify filter actually filters correctly, not just returns some rows."""
        engine = TransformEngine()
        df = sports_data["teams_df"]

        result = engine.apply_step(df, StepConfig(
            op="filter",
            params={"expr": "rating > 80"},
        ))

        # Manually verify: teams with rating > 80
        expected = df[df["rating"] > 80]
        assert len(result) == len(expected), f"Expected {len(expected)}, got {len(result)}"
        assert set(result["team_id"].tolist()) == set(expected["team_id"].tolist())

    def test_select_rename_preserves_data(self, sports_data):
        """Select with rename dict preserves actual values."""
        engine = TransformEngine()
        df = sports_data["teams_df"]

        result = engine.apply_step(df, StepConfig(
            op="select",
            params={"columns": {"id": "team_id", "w": "wins", "l": "losses"}},
        ))

        assert list(result.columns) == ["id", "w", "l"]
        assert result["w"].iloc[0] == df["wins"].iloc[0]
        assert result["l"].iloc[0] == df["losses"].iloc[0]

    def test_keyword_arg_style_works_identically(self, sports_data):
        """apply_step with step_type= kwarg produces same result as StepConfig."""
        engine = TransformEngine()
        df = sports_data["teams_df"]

        result_config = engine.apply_step(df, StepConfig(
            op="filter", params={"expr": "wins > 20"}
        ))
        result_kwargs = engine.apply_step(df, step_type="filter", params={"expr": "wins > 20"})

        assert result_config.equals(result_kwargs), "Both calling styles should produce identical results"


# ---------------------------------------------------------------------------
# E2E: Full chain — source → transform → expression → validate
# ---------------------------------------------------------------------------

class TestE2EFullChain:
    """Test the complete pipeline: load source → apply transforms → derive features → validate."""

    def test_complete_workflow(self, sports_data):
        """Realistic workflow: load → filter → derive → verify."""
        tmp_path = sports_data["tmp_path"]

        # 1. Load source
        file_source = FileSource()
        teams_df = file_source.load(SourceConfig(name="teams", path=sports_data["teams_path"]))
        assert len(teams_df) == 10

        # 2. Apply transforms
        transform_engine = TransformEngine()
        transformed = transform_engine.run_pipeline(teams_df, [
            StepConfig(op="filter", params={"expr": "season == 2024"}),
        ])
        assert len(transformed) == 5

        # 3. Derive features using expression engine
        expr_engine = ExpressionEngine()
        transformed = transformed.copy()
        transformed["win_pct"] = expr_engine.evaluate(transformed, "safe_div(wins, wins + losses)")
        transformed["point_diff"] = expr_engine.evaluate(transformed, "points_per_game - opp_points_per_game")
        transformed["rating_zscore"] = expr_engine.evaluate(transformed, "zscore(rating)")

        # 4. Verify derived features are correct
        assert "win_pct" in transformed.columns
        assert "point_diff" in transformed.columns
        assert "rating_zscore" in transformed.columns

        # Win pct should be between 0 and 1
        assert all(0 <= wp <= 1 for wp in transformed["win_pct"])

        # Point diff should be positive for good teams, negative for bad
        best_team = transformed.loc[transformed["rating"].idxmax()]
        worst_team = transformed.loc[transformed["rating"].idxmin()]
        assert best_team["point_diff"] > 0, "Best team should have positive point diff"
        assert worst_team["point_diff"] < 0, "Worst team should have negative point diff"

        # Z-score should have mean ~0
        assert abs(transformed["rating_zscore"].mean()) < 1e-10

        # 5. Validate the schema
        validator = ExpressionValidator()
        schema = {
            "columns": list(transformed.columns),
            "column_types": {col: str(dtype) for col, dtype in transformed.dtypes.items()},
        }

        # These should pass
        assert validator.validate("win_pct * 100", schema).is_valid
        assert validator.validate("abs(point_diff)", schema).is_valid
        assert validator.validate("where(win_pct > 0.6, 1, 0)", schema).is_valid

        # These should fail
        assert not validator.validate("nonexistent_col * 2", schema).is_valid
        assert not validator.validate("bad_func(rating)", schema).is_valid

    def test_games_data_pipeline(self, sports_data):
        """Process games data: load → filter → derive → verify correctness."""
        file_source = FileSource()
        games_df = file_source.load(SourceConfig(name="games", path=sports_data["games_path"]))

        expr_engine = ExpressionEngine()
        transform_engine = TransformEngine()

        # Filter to non-neutral-site games in 2024
        processed = transform_engine.run_pipeline(games_df, [
            StepConfig(op="filter", params={"expr": "season == 2024 and neutral_site == False"}),
        ])
        assert len(processed) < len(games_df), "Filter should reduce rows"
        assert all(processed["season"] == 2024)
        assert all(processed["neutral_site"] == False)

        # Derive features
        processed = processed.copy()
        processed["margin"] = expr_engine.evaluate(processed, "score_a - score_b")
        processed["total"] = expr_engine.evaluate(processed, "score_a + score_b")
        processed["home_win"] = expr_engine.evaluate(processed, "where(score_a > score_b, 1, 0)")

        # Verify margin is correct for each row
        for _, row in processed.iterrows():
            expected_margin = row["score_a"] - row["score_b"]
            assert row["margin"] == expected_margin, f"Margin wrong: {row['margin']} != {expected_margin}"

        # Verify home_win matches margin sign
        for _, row in processed.iterrows():
            if row["margin"] > 0:
                assert row["home_win"] == 1, f"home_win should be 1 when margin > 0"
            else:
                assert row["home_win"] == 0, f"home_win should be 0 when margin <= 0"
