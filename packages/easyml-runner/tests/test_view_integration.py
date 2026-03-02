"""Integration tests for the declarative ETL view system.

Tests the full workflow: scaffold -> add sources -> add views -> preview -> set features_view -> get_features_df.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.runner import config_writer
from easyml.runner.scaffold import scaffold_project
from easyml.runner.data_utils import get_features_df, load_data_config
from easyml.runner.view_resolver import ViewResolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scaffold(tmp_path: Path) -> Path:
    """Scaffold a project into tmp_path and return it."""
    scaffold_project(tmp_path, "test-project")
    return tmp_path


def _make_csv(project_dir: Path, rel_path: str, df: pd.DataFrame) -> Path:
    """Write a DataFrame as CSV at a relative path inside the project."""
    full = project_dir / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full, index=False)
    return full


def _load_pipeline_yaml(project_dir: Path) -> dict:
    """Load the pipeline.yaml as a raw dict."""
    path = project_dir / "config" / "pipeline.yaml"
    return yaml.safe_load(path.read_text()) or {}


# ---------------------------------------------------------------------------
# TestViewConfigWriter — config_writer backend functions
# ---------------------------------------------------------------------------


class TestViewConfigWriter:
    """Test config_writer view functions in sequence."""

    def test_add_source(self, tmp_path: Path):
        """Scaffold project, create CSV, add_source, verify YAML and markdown."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Season": rng.integers(2020, 2025, size=30),
            "TeamID": rng.integers(1000, 1020, size=30),
            "Score": rng.integers(40, 100, size=30),
            "FGM": rng.integers(10, 40, size=30),
            "FGA": rng.integers(30, 80, size=30),
        })
        _make_csv(proj, "data/raw/stats.csv", df)

        result = config_writer.add_source(proj, "stats", "data/raw/stats.csv")
        assert "**Added source**" in result
        assert "Season" in result
        assert "TeamID" in result

        # Verify pipeline.yaml has the source entry
        pipeline = _load_pipeline_yaml(proj)
        sources = pipeline["data"]["sources"]
        assert "stats" in sources
        assert sources["stats"]["path"] == "data/raw/stats.csv"

    def test_add_view_filter(self, tmp_path: Path):
        """Add a source then add a view with a filter step."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Season": rng.integers(2020, 2025, size=30),
            "TeamID": rng.integers(1000, 1020, size=30),
            "Score": rng.integers(40, 100, size=30),
        })
        _make_csv(proj, "data/raw/scores.csv", df)
        config_writer.add_source(proj, "scores", "data/raw/scores.csv")

        result = config_writer.add_view(
            proj, "high_scores", "scores",
            steps=[{"op": "filter", "expr": "Score > 50"}],
        )
        assert "**Added view**" in result
        assert "high_scores" in result

    def test_add_view_validates_source(self, tmp_path: Path):
        """Adding a view with a nonexistent source returns an error."""
        proj = _scaffold(tmp_path)

        result = config_writer.add_view(
            proj, "orphan", "nonexistent_source",
            steps=[{"op": "filter", "expr": "x > 0"}],
        )
        assert "**Error**" in result
        assert "nonexistent_source" in result

    def test_add_view_validates_steps(self, tmp_path: Path):
        """Adding a view with an invalid step op returns an error."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.standard_normal(20)})
        _make_csv(proj, "data/raw/dummy.csv", df)
        config_writer.add_source(proj, "dummy", "data/raw/dummy.csv")

        result = config_writer.add_view(
            proj, "bad_view", "dummy",
            steps=[{"op": "banana", "columns": ["x"]}],
        )
        assert "**Error**" in result

    def test_remove_view(self, tmp_path: Path):
        """Add a view and remove it; list_views should show no views."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.standard_normal(20)})
        _make_csv(proj, "data/raw/data.csv", df)
        config_writer.add_source(proj, "data", "data/raw/data.csv")
        config_writer.add_view(proj, "v1", "data", steps=[{"op": "filter", "expr": "x > 0"}])

        result = config_writer.remove_view(proj, "v1")
        assert "**Removed view**" in result

        listing = config_writer.list_views(proj)
        assert "No views" in listing

    def test_remove_view_clears_features_view(self, tmp_path: Path):
        """When the features_view is removed, pipeline.yaml should clear it."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.standard_normal(20)})
        _make_csv(proj, "data/raw/data.csv", df)
        config_writer.add_source(proj, "data", "data/raw/data.csv")
        config_writer.add_view(proj, "v1", "data", steps=[{"op": "filter", "expr": "x > 0"}])
        config_writer.set_features_view(proj, "v1")

        # Confirm features_view is set
        pipeline = _load_pipeline_yaml(proj)
        assert pipeline["data"]["features_view"] == "v1"

        config_writer.remove_view(proj, "v1")

        pipeline = _load_pipeline_yaml(proj)
        assert pipeline["data"]["features_view"] is None

    def test_list_views(self, tmp_path: Path):
        """list_views output includes both view names."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.standard_normal(20)})
        _make_csv(proj, "data/raw/data.csv", df)
        config_writer.add_source(proj, "data", "data/raw/data.csv")
        config_writer.add_view(proj, "alpha", "data", steps=[{"op": "filter", "expr": "x > 0"}])
        config_writer.add_view(proj, "beta", "data", steps=[{"op": "filter", "expr": "x < 0"}])

        listing = config_writer.list_views(proj)
        assert "alpha" in listing
        assert "beta" in listing

    def test_set_features_view(self, tmp_path: Path):
        """set_features_view writes the view name into pipeline.yaml."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.standard_normal(20)})
        _make_csv(proj, "data/raw/data.csv", df)
        config_writer.add_source(proj, "data", "data/raw/data.csv")
        config_writer.add_view(proj, "pred", "data", steps=[{"op": "filter", "expr": "x > 0"}])

        result = config_writer.set_features_view(proj, "pred")
        assert "pred" in result
        assert "prediction table" in result

        pipeline = _load_pipeline_yaml(proj)
        assert pipeline["data"]["features_view"] == "pred"

    def test_view_dag(self, tmp_path: Path):
        """DAG output includes dependency arrows for chained views."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df_a = pd.DataFrame({
            "Season": rng.integers(2020, 2025, size=20),
            "TeamID": rng.integers(1000, 1010, size=20),
            "Score": rng.integers(40, 100, size=20),
        })
        df_b = pd.DataFrame({
            "Season": rng.integers(2020, 2025, size=20),
            "TeamID": rng.integers(1000, 1010, size=20),
            "Seed": rng.integers(1, 17, size=20),
        })
        _make_csv(proj, "data/raw/teams.csv", df_a)
        _make_csv(proj, "data/raw/seeds.csv", df_b)
        config_writer.add_source(proj, "teams", "data/raw/teams.csv")
        config_writer.add_source(proj, "seeds", "data/raw/seeds.csv")

        config_writer.add_view(
            proj, "base", "teams",
            steps=[{"op": "filter", "expr": "Score > 50"}],
        )
        config_writer.add_view(
            proj, "enriched", "base",
            steps=[{"op": "join", "other": "seeds", "on": ["Season", "TeamID"]}],
        )
        config_writer.set_features_view(proj, "enriched")

        dag = config_writer.view_dag(proj)
        assert "base" in dag
        assert "enriched" in dag
        assert "seeds" in dag
        assert "teams" in dag


# ---------------------------------------------------------------------------
# TestViewPreview — preview_view tests
# ---------------------------------------------------------------------------


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_tabulate(), reason="preview_view uses df.to_markdown which requires tabulate")
class TestViewPreview:
    """Test preview_view for sources and derived views."""

    def test_preview_source(self, tmp_path: Path):
        """preview_view on a raw source shows rows and column info."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "Season": rng.integers(2020, 2025, size=20),
            "Score": rng.integers(40, 100, size=20),
        })
        _make_csv(proj, "data/raw/data.csv", df)
        config_writer.add_source(proj, "data", "data/raw/data.csv")

        result = config_writer.preview_view(proj, "data")
        assert "Preview" in result
        assert "Season" in result
        assert "Score" in result
        assert "Rows:" in result

    def test_preview_derived_view(self, tmp_path: Path):
        """preview_view on a view with a derive step shows the derived column."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.integers(1, 50, size=25),
            "b": rng.integers(1, 50, size=25),
        })
        _make_csv(proj, "data/raw/data.csv", df)
        config_writer.add_source(proj, "data", "data/raw/data.csv")
        config_writer.add_view(
            proj, "derived", "data",
            steps=[{"op": "derive", "columns": {"total": "a + b"}}],
        )

        result = config_writer.preview_view(proj, "derived")
        assert "total" in result
        assert "Preview" in result


# ---------------------------------------------------------------------------
# TestViewE2E — end-to-end pipeline tests
# ---------------------------------------------------------------------------


class TestViewE2E:
    """End-to-end tests: declare sources + views -> set features_view -> get_features_df."""

    def test_e2e_two_csv_join(self, tmp_path: Path):
        """Full E2E: two CSVs joined via views, loaded through get_features_df."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)

        n_teams = 40
        seasons = rng.choice([2022, 2023, 2024], size=n_teams)
        team_ids = rng.integers(1000, 1020, size=n_teams)
        teams_df = pd.DataFrame({
            "Season": seasons,
            "TeamID": team_ids,
            "WinPct": rng.uniform(0.1, 0.9, size=n_teams).round(3),
            "AvgScore": rng.integers(50, 90, size=n_teams).astype(float),
        })
        _make_csv(proj, "data/raw/teams.csv", teams_df)

        seeds_df = pd.DataFrame({
            "Season": seasons,
            "TeamID": team_ids,
            "Seed": rng.integers(1, 17, size=n_teams),
        })
        _make_csv(proj, "data/raw/seeds.csv", seeds_df)

        # Add sources
        config_writer.add_source(proj, "teams", "data/raw/teams.csv")
        config_writer.add_source(proj, "seeds", "data/raw/seeds.csv")

        # Add views
        config_writer.add_view(
            proj, "team_stats", "teams",
            steps=[{"op": "filter", "expr": "WinPct > 0.3"}],
        )
        config_writer.add_view(
            proj, "prediction_table", "team_stats",
            steps=[{"op": "join", "other": "seeds", "on": ["Season", "TeamID"]}],
        )

        # Set features_view
        config_writer.set_features_view(proj, "prediction_table")

        # Load config and get features
        config = load_data_config(proj)
        assert config.features_view == "prediction_table"

        df = get_features_df(proj, config)

        # Verify columns from both CSVs
        assert "WinPct" in df.columns
        assert "AvgScore" in df.columns
        assert "Seed" in df.columns
        assert "Season" in df.columns
        assert "TeamID" in df.columns

        # Verify filter was applied (all WinPct > 0.3)
        assert (df["WinPct"] > 0.3).all()

        # Verify join worked (Seed column present with non-null values)
        assert df["Seed"].notna().sum() > 0

    def test_e2e_group_by_and_derive(self, tmp_path: Path):
        """E2E: game-level data -> group_by to team-season level -> derive a difference."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)

        n_games = 50
        games_df = pd.DataFrame({
            "Season": rng.choice([2022, 2023], size=n_games),
            "TeamID": rng.choice([1001, 1002, 1003, 1004], size=n_games),
            "Score": rng.integers(50, 100, size=n_games),
            "Win": rng.integers(0, 2, size=n_games),
        })
        _make_csv(proj, "data/raw/games.csv", games_df)

        config_writer.add_source(proj, "games", "data/raw/games.csv")

        # Group by Season/TeamID, aggregate Score (mean) and Win (sum)
        config_writer.add_view(
            proj, "team_season", "games",
            steps=[{
                "op": "group_by",
                "keys": ["Season", "TeamID"],
                "aggs": {"Score": "mean", "Win": "sum"},
            }],
        )

        # Derive a new column: Score_mean minus Win_sum (arbitrary, just to test derive)
        config_writer.add_view(
            proj, "final", "team_season",
            steps=[{
                "op": "derive",
                "columns": {"score_minus_wins": "Score_mean - Win_sum"},
            }],
        )

        config_writer.set_features_view(proj, "final")

        config = load_data_config(proj)
        df = get_features_df(proj, config)

        # Verify aggregation columns exist
        assert "Score_mean" in df.columns
        assert "Win_sum" in df.columns
        assert "score_minus_wins" in df.columns

        # Verify grouping reduced rows (each Season/TeamID combination is unique)
        assert len(df) == len(df.drop_duplicates(subset=["Season", "TeamID"]))

        # Verify derived column math is correct
        expected = df["Score_mean"] - df["Win_sum"]
        pd.testing.assert_series_equal(
            df["score_minus_wins"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_backward_compat_no_views(self, tmp_path: Path):
        """When features_view is None, get_features_df reads from parquet file."""
        proj = _scaffold(tmp_path)
        rng = np.random.default_rng(42)

        # Create features parquet directly
        features_dir = proj / "data" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "Season": rng.integers(2020, 2025, size=30),
            "TeamID": rng.integers(1000, 1020, size=30),
            "feat_a": rng.standard_normal(30),
            "feat_b": rng.standard_normal(30),
            "result": rng.integers(0, 2, size=30),
        })
        df.to_parquet(features_dir / "features.parquet", index=False)

        config = load_data_config(proj)
        assert config.features_view is None

        loaded = get_features_df(proj, config)
        assert len(loaded) == 30
        assert "feat_a" in loaded.columns
        assert "feat_b" in loaded.columns
        pd.testing.assert_frame_equal(loaded, df)
