"""Integration test for declarative feature system end-to-end.

Exercises the full workflow: configure sources, add features of all four types
via the FeatureStore, verify auto-pairwise generation, resolve feature sets,
verify caching persists across instances, and verify cascade removal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.core.runner.feature_store import FeatureStore
from easyml.core.runner.schema import (
    DataConfig,
    FeatureDef,
    FeatureStoreConfig,
    FeatureType,
    PairwiseMode,
    SourceConfig,
)


# -----------------------------------------------------------------------
# Setup helper
# -----------------------------------------------------------------------


def _setup_full_project(tmp_path: Path) -> Path:
    """Create a full project with sources, matchups, and config."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    rng = np.random.default_rng(42)

    # Config
    config_dir = project_dir / "config"
    config_dir.mkdir()
    config = {
        "data": {
            "features_file": "features.parquet",
            "target_column": "result",
            "key_columns": [],
            "time_column": "period_id",
            "sources": {
                "team_stats": {
                    "name": "team_stats",
                    "path": "data/raw/team_stats.parquet",
                },
            },
        },
    }
    (config_dir / "pipeline.yaml").write_text(yaml.dump(config))

    # Entity-level data
    data_dir = project_dir / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir(parents=True)

    n_teams = 30
    entities = []
    for season in [2022, 2023, 2024]:
        for team_id in range(1, n_teams + 1):
            entities.append(
                {
                    "entity_id": team_id,
                    "period_id": season,
                    "adj_em": rng.standard_normal() * 10,
                    "adj_tempo": rng.standard_normal() * 5 + 65,
                    "adj_oe": rng.standard_normal() * 5 + 105,
                    "win_rate": rng.uniform(0.2, 0.9),
                }
            )
    pd.DataFrame(entities).to_parquet(raw_dir / "team_stats.parquet", index=False)

    # Matchup-level data
    n_matchups = 300
    matchups = []
    for i in range(n_matchups):
        season = rng.choice([2022, 2023, 2024])
        team_a = rng.integers(1, n_teams + 1)
        team_b = rng.integers(1, n_teams + 1)
        while team_b == team_a:
            team_b = rng.integers(1, n_teams + 1)
        matchups.append(
            {
                "entity_a_id": int(team_a),
                "entity_b_id": int(team_b),
                "period_id": int(season),
                "result": int(rng.integers(0, 2)),
                "day_num": int(rng.integers(1, 155)),
                "is_neutral": int(rng.integers(0, 2)),
            }
        )
    pd.DataFrame(matchups).to_parquet(
        features_dir / "features.parquet", index=False
    )

    return project_dir


def _make_data_config(project_dir: Path) -> DataConfig:
    """Load the DataConfig from the project's pipeline.yaml."""
    from easyml.core.runner.data_utils import load_data_config

    return load_data_config(project_dir)


def _make_store(project_dir: Path, config: DataConfig | None = None) -> FeatureStore:
    """Create a FeatureStore for the project."""
    if config is None:
        config = _make_data_config(project_dir)
    return FeatureStore(project_dir, config)


# -----------------------------------------------------------------------
# TestFullWorkflow
# -----------------------------------------------------------------------


class TestFullWorkflow:
    """End-to-end tests for adding features and verifying outputs."""

    def test_team_feature_with_auto_pairwise(self, tmp_path: Path) -> None:
        """Add adj_em and adj_tempo as team features.

        Verify both team and pairwise features show up in available().
        """
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        # Add two team features
        store.add(
            FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="team_stats",
                category="efficiency",
            )
        )
        store.add(
            FeatureDef(
                name="adj_tempo",
                type=FeatureType.TEAM,
                source="team_stats",
                category="pace",
            )
        )

        all_features = store.available()
        all_names = {f.name for f in all_features}

        # Team features should be registered
        assert "adj_em" in all_names
        assert "adj_tempo" in all_names

        # Auto-pairwise diff features should be registered
        assert "diff_adj_em" in all_names
        assert "diff_adj_tempo" in all_names

        # Check type of auto-generated features
        pairwise_features = store.available(type_filter=FeatureType.PAIRWISE)
        pairwise_names = {f.name for f in pairwise_features}
        assert "diff_adj_em" in pairwise_names
        assert "diff_adj_tempo" in pairwise_names

    def test_multiple_team_features_different_categories(
        self, tmp_path: Path
    ) -> None:
        """Add features in different categories (efficiency, pace).

        Verify resolve_sets returns correct columns per category.
        """
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="team_stats",
                category="efficiency",
            )
        )
        store.add(
            FeatureDef(
                name="adj_tempo",
                type=FeatureType.TEAM,
                source="team_stats",
                category="pace",
            )
        )

        efficiency_cols = store.resolve_sets(["efficiency"])
        pace_cols = store.resolve_sets(["pace"])

        assert "diff_adj_em" in efficiency_cols
        assert "diff_adj_tempo" not in efficiency_cols

        assert "diff_adj_tempo" in pace_cols
        assert "diff_adj_em" not in pace_cols

    def test_regime_feature(self, tmp_path: Path) -> None:
        """Add 'late_season' with condition 'day_num > 100'.

        Verify binary output (0/1 only).
        """
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="late_season",
                type=FeatureType.REGIME,
                condition="day_num > 100",
            )
        )

        series = store.compute("late_season")
        unique_vals = set(series.unique())
        assert unique_vals <= {0.0, 1.0}, f"Expected only 0/1 but got {unique_vals}"

    def test_matchup_feature(self, tmp_path: Path) -> None:
        """Add 'neutral_site' from column 'is_neutral'. Verify compute works."""
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="neutral_site",
                type=FeatureType.MATCHUP,
                column="is_neutral",
            )
        )

        series = store.compute("neutral_site")
        assert len(series) > 0
        # Should match length of matchup data
        matchup_df = store._load_matchup_data()
        assert len(series) == len(matchup_df)

    def test_formula_pairwise(self, tmp_path: Path) -> None:
        """Add 'day_squared' with formula 'day_num ** 2'. Verify compute works."""
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="day_squared",
                type=FeatureType.PAIRWISE,
                formula="day_num ** 2",
            )
        )

        series = store.compute("day_squared")
        matchup_df = store._load_matchup_data()
        assert len(series) == len(matchup_df)

        # Verify the formula was applied correctly
        expected = matchup_df["day_num"].astype(float) ** 2
        pd.testing.assert_series_equal(
            series.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_resolve_multiple_types(self, tmp_path: Path) -> None:
        """Add one of each type. Verify resolve() returns DataFrame with all columns."""
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        # Team
        store.add(
            FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="team_stats",
            )
        )
        # Pairwise formula
        store.add(
            FeatureDef(
                name="day_sq",
                type=FeatureType.PAIRWISE,
                formula="day_num ** 2",
            )
        )
        # Matchup
        store.add(
            FeatureDef(
                name="neutral",
                type=FeatureType.MATCHUP,
                column="is_neutral",
            )
        )
        # Regime
        store.add(
            FeatureDef(
                name="late",
                type=FeatureType.REGIME,
                condition="day_num > 100",
            )
        )

        # Resolve all pairwise/matchup/regime features
        resolved = store.resolve(["diff_adj_em", "day_sq", "neutral", "late"])

        assert "diff_adj_em" in resolved.columns
        assert "day_sq" in resolved.columns
        assert "neutral" in resolved.columns
        assert "late" in resolved.columns

        # Original matchup columns should still be present
        assert "result" in resolved.columns
        assert "entity_a_id" in resolved.columns

    def test_team_with_both_pairwise(self, tmp_path: Path) -> None:
        """Add with pairwise_mode=BOTH. Verify both diff_ and ratio_ generated."""
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="team_stats",
                pairwise_mode=PairwiseMode.BOTH,
            )
        )

        all_names = {f.name for f in store.available()}
        assert "diff_adj_em" in all_names
        assert "ratio_adj_em" in all_names

        # Both should be computable
        diff_series = store.compute("diff_adj_em")
        ratio_series = store.compute("ratio_adj_em")
        assert len(diff_series) > 0
        assert len(ratio_series) > 0

    def test_team_with_no_pairwise(self, tmp_path: Path) -> None:
        """Add with pairwise_mode=NONE. Verify no derivatives."""
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="team_stats",
                pairwise_mode=PairwiseMode.NONE,
            )
        )

        all_names = {f.name for f in store.available()}
        assert "adj_em" in all_names
        assert "diff_adj_em" not in all_names
        assert "ratio_adj_em" not in all_names


# -----------------------------------------------------------------------
# TestCaching
# -----------------------------------------------------------------------


class TestCaching:
    """Tests for cache persistence and invalidation."""

    def test_caching_persists_across_instances(self, tmp_path: Path) -> None:
        """Add feature, compute. Create new FeatureStore instance (same config).

        Compute same feature -- should match.
        """
        project_dir = _setup_full_project(tmp_path)
        config = _make_data_config(project_dir)

        # First instance: add and compute
        store1 = FeatureStore(project_dir, config)
        store1.add(
            FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="team_stats",
            )
        )
        series1 = store1.compute("diff_adj_em")

        # Second instance: new store with same config
        # Must re-register the feature definitions since the new store starts
        # with whatever is in config.feature_defs. Save registry first.
        store1.save_registry()

        store2 = FeatureStore(project_dir, config)
        series2 = store2.compute("diff_adj_em")

        pd.testing.assert_series_equal(
            series1.reset_index(drop=True),
            series2.reset_index(drop=True),
            check_names=False,
        )

    def test_cache_invalidation(self, tmp_path: Path) -> None:
        """Add feature, verify cached. Call store.refresh(). Verify still works."""
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="team_stats",
            )
        )

        # Verify it's cached
        assert "adj_em" in store._cache._entries
        assert "diff_adj_em" in store._cache._entries

        # Refresh (invalidates and recomputes)
        results = store.refresh()
        assert "adj_em" in results
        assert results["adj_em"] == "refreshed"

        # Should still be computable after refresh
        series = store.compute("diff_adj_em")
        assert len(series) > 0


# -----------------------------------------------------------------------
# TestRemoval
# -----------------------------------------------------------------------


class TestRemoval:
    """Tests for feature removal with cascade."""

    def test_remove_team_cascades(self, tmp_path: Path) -> None:
        """Add team feature (creates derivatives). Remove it. Verify all gone."""
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="team_stats",
                pairwise_mode=PairwiseMode.BOTH,
            )
        )

        # Verify they exist
        all_names = {f.name for f in store.available()}
        assert "adj_em" in all_names
        assert "diff_adj_em" in all_names
        assert "ratio_adj_em" in all_names

        # Remove parent
        store.remove("adj_em")

        # All should be gone
        remaining_names = {f.name for f in store.available()}
        assert "adj_em" not in remaining_names
        assert "diff_adj_em" not in remaining_names
        assert "ratio_adj_em" not in remaining_names

        # Cache should also be cleaned
        assert "adj_em" not in store._cache._entries
        assert "diff_adj_em" not in store._cache._entries
        assert "ratio_adj_em" not in store._cache._entries

    def test_remove_matchup(self, tmp_path: Path) -> None:
        """Add matchup feature. Remove it. Verify gone."""
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="neutral_site",
                type=FeatureType.MATCHUP,
                column="is_neutral",
            )
        )

        assert "neutral_site" in {f.name for f in store.available()}

        store.remove("neutral_site")

        assert "neutral_site" not in {f.name for f in store.available()}
        assert "neutral_site" not in store._cache._entries


# -----------------------------------------------------------------------
# TestResolveSets
# -----------------------------------------------------------------------


class TestResolveSets:
    """Tests for resolve_sets (category-based feature set resolution)."""

    def test_resolve_sets_by_category(self, tmp_path: Path) -> None:
        """Add features in multiple categories.

        Verify resolve_sets returns correct pairwise names per category.
        """
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        store.add(
            FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="team_stats",
                category="efficiency",
            )
        )
        store.add(
            FeatureDef(
                name="adj_oe",
                type=FeatureType.TEAM,
                source="team_stats",
                category="efficiency",
            )
        )
        store.add(
            FeatureDef(
                name="adj_tempo",
                type=FeatureType.TEAM,
                source="team_stats",
                category="pace",
            )
        )

        efficiency_cols = store.resolve_sets(["efficiency"])
        assert "diff_adj_em" in efficiency_cols
        assert "diff_adj_oe" in efficiency_cols
        assert "diff_adj_tempo" not in efficiency_cols

        pace_cols = store.resolve_sets(["pace"])
        assert "diff_adj_tempo" in pace_cols
        assert "diff_adj_em" not in pace_cols

        # Both categories at once
        all_cols = store.resolve_sets(["efficiency", "pace"])
        assert "diff_adj_em" in all_cols
        assert "diff_adj_oe" in all_cols
        assert "diff_adj_tempo" in all_cols

    def test_resolve_sets_empty_category(self, tmp_path: Path) -> None:
        """Resolve a category with no features. Returns empty list."""
        project_dir = _setup_full_project(tmp_path)
        store = _make_store(project_dir)

        result = store.resolve_sets(["nonexistent_category"])
        assert result == []


# -----------------------------------------------------------------------
# TestMCPToolIntegration
# -----------------------------------------------------------------------


class TestMCPToolIntegration:
    """Tests for config_writer.add_feature MCP tool integration."""

    def test_add_feature_mcp_tool(self, tmp_path: Path) -> None:
        """Use config_writer.add_feature() with type='team'.

        Verify markdown output mentions auto-pairwise.
        """
        from easyml.core.runner.config_writer import add_feature

        project_dir = _setup_full_project(tmp_path)

        result = add_feature(
            project_dir,
            "adj_em",
            type="team",
            source="team_stats",
            category="efficiency",
        )

        # Should be markdown string
        assert isinstance(result, str)
        # Should mention auto-pairwise
        assert "pairwise" in result.lower() or "diff_adj_em" in result
        # Should mention the feature name
        assert "adj_em" in result

    def test_add_feature_formula_backward_compat(self, tmp_path: Path) -> None:
        """Use config_writer.add_feature() with just formula (no type).

        Verify works via the old formula engine path.
        """
        from easyml.core.runner.config_writer import add_feature

        project_dir = _setup_full_project(tmp_path)

        result = add_feature(
            project_dir,
            "day_squared",
            formula="day_num ** 2",
            description="Day number squared",
        )

        assert isinstance(result, str)
        assert "day_squared" in result
