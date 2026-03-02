"""Tests for the declarative feature store."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.runner.feature_store import FeatureStore
from easyml.runner.schema import (
    DataConfig,
    FeatureDef,
    FeatureStoreConfig,
    FeatureType,
    PairwiseMode,
    SourceConfig,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_project(tmp_path: Path) -> Path:
    """Create a minimal project with entity-level and matchup-level data."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Config dir
    config_dir = project_dir / "config"
    config_dir.mkdir()
    (config_dir / "pipeline.yaml").write_text(
        "data:\n  features_file: features.parquet\n  target_column: result\n"
    )

    # Entity-level data
    rng = np.random.default_rng(42)
    n_teams = 20
    entities = []
    for season in [2022, 2023, 2024]:
        for team_id in range(1, n_teams + 1):
            entities.append({
                "entity_id": team_id,
                "period_id": season,
                "adj_em": rng.standard_normal() * 10,
                "adj_tempo": rng.standard_normal() * 5 + 65,
                "win_rate": rng.uniform(0.2, 0.9),
            })
    raw_dir = project_dir / "data" / "raw"
    raw_dir.mkdir(parents=True)
    pd.DataFrame(entities).to_parquet(raw_dir / "kenpom.parquet", index=False)

    # Matchup-level data
    n_matchups = 200
    features_dir = project_dir / "data" / "features"
    features_dir.mkdir(parents=True)
    matchups = []
    for i in range(n_matchups):
        season = rng.choice([2022, 2023, 2024])
        team_a = rng.integers(1, n_teams + 1)
        team_b = rng.integers(1, n_teams + 1)
        while team_b == team_a:
            team_b = rng.integers(1, n_teams + 1)
        matchups.append({
            "entity_a_id": int(team_a),
            "entity_b_id": int(team_b),
            "period_id": int(season),
            "result": int(rng.integers(0, 2)),
            "day_num": int(rng.integers(1, 155)),
            "is_neutral": int(rng.integers(0, 2)),
        })
    pd.DataFrame(matchups).to_parquet(features_dir / "features.parquet", index=False)

    return project_dir


def _make_config(project_dir: Path) -> DataConfig:
    """Build a DataConfig matching the project fixture."""
    return DataConfig(
        features_dir="data/features",
        features_file="features.parquet",
        target_column="result",
        raw_dir="data/raw",
        sources={
            "kenpom": SourceConfig(
                name="kenpom",
                path="data/raw/kenpom.parquet",
                format="parquet",
            ),
        },
        feature_store=FeatureStoreConfig(
            cache_dir="data/features/cache",
            auto_pairwise=True,
            default_pairwise_mode=PairwiseMode.DIFF,
        ),
    )


@pytest.fixture
def project_dir(tmp_path):
    return _make_project(tmp_path)


@pytest.fixture
def config(project_dir):
    return _make_config(project_dir)


@pytest.fixture
def store(project_dir, config):
    return FeatureStore(project_dir, config)


# ------------------------------------------------------------------
# Init
# ------------------------------------------------------------------


class TestFeatureStoreInit:
    def test_init_defaults(self, store: FeatureStore):
        assert store._registry == {}
        assert store._matchup_df is None
        assert store._source_dfs == {}

    def test_init_with_feature_defs(self, project_dir, config):
        feat = FeatureDef(name="adj_em", type=FeatureType.TEAM, source="kenpom")
        config.feature_defs = {"adj_em": feat}
        st = FeatureStore(project_dir, config)
        assert "adj_em" in st._registry
        assert st._registry["adj_em"].type == FeatureType.TEAM

    def test_cache_dir_created(self, store: FeatureStore):
        cache_dir = store.project_dir / "data" / "features" / "cache"
        assert cache_dir.exists()
        assert (cache_dir / "team").exists()
        assert (cache_dir / "pairwise").exists()


# ------------------------------------------------------------------
# Team features
# ------------------------------------------------------------------


class TestTeamFeatures:
    def test_add_team_feature(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        )
        result = store.add(feat)
        assert result.name == "adj_em"
        assert "adj_em" in store._registry

    def test_auto_pairwise_diff(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
            pairwise_mode=PairwiseMode.DIFF,
        )
        result = store.add(feat)
        # Should generate diff_adj_em
        assert "diff_adj_em" in store._registry
        assert store._registry["diff_adj_em"].type == FeatureType.PAIRWISE
        # Column added should be the diff name
        assert result.column_added == "diff_adj_em"

    def test_auto_pairwise_both(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
            pairwise_mode=PairwiseMode.BOTH,
        )
        store.add(feat)
        assert "diff_adj_em" in store._registry
        assert "ratio_adj_em" in store._registry

    def test_auto_pairwise_none(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
            pairwise_mode=PairwiseMode.NONE,
        )
        store.add(feat)
        assert "diff_adj_em" not in store._registry
        assert "ratio_adj_em" not in store._registry

    def test_auto_pairwise_disabled_in_config(self, project_dir, config):
        config.feature_store.auto_pairwise = False
        st = FeatureStore(project_dir, config)
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
            pairwise_mode=PairwiseMode.DIFF,
        )
        st.add(feat)
        assert "diff_adj_em" not in st._registry

    def test_compute_pairwise_series(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
            pairwise_mode=PairwiseMode.DIFF,
        )
        store.add(feat)
        series = store.compute("diff_adj_em")
        assert isinstance(series, pd.Series)
        matchup_df = store._load_matchup_data()
        assert len(series) == len(matchup_df)

    def test_caching_hit(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        )
        store.add(feat)
        # Second compute should hit cache
        series = store.compute("diff_adj_em")
        assert series is not None

    def test_missing_source_error(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="nonexistent_source", column="adj_em",
        )
        with pytest.raises(ValueError, match="not found in config.sources"):
            store.add(feat)

    def test_missing_column_error(self, store: FeatureStore):
        feat = FeatureDef(
            name="bad_col", type=FeatureType.TEAM,
            source="kenpom", column="nonexistent_column",
        )
        with pytest.raises(ValueError, match="not found in source"):
            store.add(feat)

    def test_team_feature_requires_source(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
        )
        with pytest.raises(ValueError, match="requires a 'source'"):
            store.add(feat)

    def test_column_defaults_to_name(self, store: FeatureStore):
        """When column is not specified, uses feature name as column name."""
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom",
        )
        result = store.add(feat)
        # Should succeed because 'adj_em' exists in the source
        assert result.name == "adj_em"


# ------------------------------------------------------------------
# Matchup features
# ------------------------------------------------------------------


class TestMatchupFeatures:
    def test_add_matchup_from_column(self, store: FeatureStore):
        feat = FeatureDef(
            name="day_num_feat", type=FeatureType.MATCHUP,
            column="day_num",
        )
        result = store.add(feat)
        assert result.name == "day_num_feat"
        assert result.column_added == "day_num_feat"
        assert "day_num_feat" in store._registry

    def test_add_matchup_with_formula(self, store: FeatureStore):
        feat = FeatureDef(
            name="day_neutral", type=FeatureType.MATCHUP,
            formula="day_num * is_neutral",
        )
        result = store.add(feat)
        assert result.name == "day_neutral"
        assert "mean" in result.stats

    def test_matchup_column_missing(self, store: FeatureStore):
        feat = FeatureDef(
            name="bad", type=FeatureType.MATCHUP,
            column="nonexistent",
        )
        with pytest.raises(ValueError, match="not found in matchup"):
            store.add(feat)

    def test_matchup_name_used_as_column(self, store: FeatureStore):
        """When neither column nor formula given, uses name as column."""
        feat = FeatureDef(
            name="day_num", type=FeatureType.MATCHUP,
        )
        result = store.add(feat)
        assert result.name == "day_num"


# ------------------------------------------------------------------
# Regime features
# ------------------------------------------------------------------


class TestRegimeFeatures:
    def test_add_regime_feature(self, store: FeatureStore):
        feat = FeatureDef(
            name="late_season", type=FeatureType.REGIME,
            condition="day_num > 100",
        )
        result = store.add(feat)
        assert result.name == "late_season"
        assert "late_season" in store._registry

    def test_regime_output_binary(self, store: FeatureStore):
        feat = FeatureDef(
            name="late_season", type=FeatureType.REGIME,
            condition="day_num > 100",
        )
        store.add(feat)
        series = store.compute("late_season")
        unique_vals = set(series.dropna().unique())
        assert unique_vals <= {0.0, 1.0}

    def test_regime_coverage_stats(self, store: FeatureStore):
        feat = FeatureDef(
            name="late_season", type=FeatureType.REGIME,
            condition="day_num > 100",
        )
        result = store.add(feat)
        assert "coverage" in result.stats
        assert 0.0 <= result.stats["coverage"] <= 1.0

    def test_regime_requires_condition(self, store: FeatureStore):
        feat = FeatureDef(
            name="bad_regime", type=FeatureType.REGIME,
        )
        with pytest.raises(ValueError, match="requires a 'condition'"):
            store.add(feat)

    def test_regime_formula_fallback(self, store: FeatureStore):
        """Regime can use formula field instead of condition."""
        feat = FeatureDef(
            name="neutral_game", type=FeatureType.REGIME,
            formula="is_neutral > 0",
        )
        result = store.add(feat)
        assert result.name == "neutral_game"
        series = store.compute("neutral_game")
        unique_vals = set(series.dropna().unique())
        assert unique_vals <= {0.0, 1.0}


# ------------------------------------------------------------------
# Pairwise formula features
# ------------------------------------------------------------------


class TestFormulaFeatures:
    def test_add_pairwise_formula(self, store: FeatureStore):
        feat = FeatureDef(
            name="day_x_neutral", type=FeatureType.PAIRWISE,
            formula="day_num * is_neutral",
        )
        result = store.add(feat)
        assert result.name == "day_x_neutral"
        assert isinstance(result.correlation, float)

    def test_pairwise_requires_formula(self, store: FeatureStore):
        feat = FeatureDef(
            name="bad_pw", type=FeatureType.PAIRWISE,
        )
        with pytest.raises(ValueError, match="requires a 'formula'"):
            store.add(feat)


# ------------------------------------------------------------------
# Resolve & query
# ------------------------------------------------------------------


class TestResolve:
    def test_resolve_features(self, store: FeatureStore):
        # Add a matchup feature
        feat = FeatureDef(
            name="day_num_f", type=FeatureType.MATCHUP,
            column="day_num",
        )
        store.add(feat)
        df = store.resolve(["day_num_f"])
        assert "day_num_f" in df.columns
        # Original matchup columns should still be there
        assert "result" in df.columns

    def test_resolve_sets_by_category(self, store: FeatureStore):
        # Add team features with categories
        feat1 = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
            category="efficiency",
        )
        feat2 = FeatureDef(
            name="adj_tempo", type=FeatureType.TEAM,
            source="kenpom", column="adj_tempo",
            category="tempo",
        )
        store.add(feat1)
        store.add(feat2)

        efficiency_cols = store.resolve_sets(["efficiency"])
        assert "diff_adj_em" in efficiency_cols

        tempo_cols = store.resolve_sets(["tempo"])
        assert "diff_adj_tempo" in tempo_cols

        # Should not cross categories
        assert "diff_adj_tempo" not in efficiency_cols
        assert "diff_adj_em" not in tempo_cols

    def test_available_all(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        )
        store.add(feat)
        available = store.available()
        names = {f.name for f in available}
        assert "adj_em" in names
        assert "diff_adj_em" in names

    def test_available_by_type(self, store: FeatureStore):
        feat1 = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        )
        feat2 = FeatureDef(
            name="day_num_f", type=FeatureType.MATCHUP,
            column="day_num",
        )
        store.add(feat1)
        store.add(feat2)

        team_feats = store.available(type_filter=FeatureType.TEAM)
        matchup_feats = store.available(type_filter=FeatureType.MATCHUP)

        team_names = {f.name for f in team_feats}
        matchup_names = {f.name for f in matchup_feats}

        assert "adj_em" in team_names
        assert "day_num_f" in matchup_names
        assert "day_num_f" not in team_names

    def test_compute_all(self, store: FeatureStore):
        feat1 = FeatureDef(
            name="day_num_f", type=FeatureType.MATCHUP,
            column="day_num",
        )
        feat2 = FeatureDef(
            name="neutral_f", type=FeatureType.MATCHUP,
            column="is_neutral",
        )
        store.add(feat1)
        store.add(feat2)

        df = store.compute_all(["day_num_f", "neutral_f"])
        assert "day_num_f" in df.columns
        assert "neutral_f" in df.columns

    def test_compute_unknown_feature_raises(self, store: FeatureStore):
        with pytest.raises(ValueError, match="not registered"):
            store.compute("nonexistent")


# ------------------------------------------------------------------
# Remove feature
# ------------------------------------------------------------------


class TestRemoveFeature:
    def test_remove_cascades_to_derivatives(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
            pairwise_mode=PairwiseMode.BOTH,
        )
        store.add(feat)
        assert "adj_em" in store._registry
        assert "diff_adj_em" in store._registry
        assert "ratio_adj_em" in store._registry

        store.remove("adj_em")
        assert "adj_em" not in store._registry
        assert "diff_adj_em" not in store._registry
        assert "ratio_adj_em" not in store._registry

    def test_remove_nonexistent_noop(self, store: FeatureStore):
        # Should not raise
        store.remove("nonexistent")

    def test_remove_matchup_feature(self, store: FeatureStore):
        feat = FeatureDef(
            name="day_num_f", type=FeatureType.MATCHUP,
            column="day_num",
        )
        store.add(feat)
        assert "day_num_f" in store._registry
        store.remove("day_num_f")
        assert "day_num_f" not in store._registry


# ------------------------------------------------------------------
# Refresh
# ------------------------------------------------------------------


class TestRefresh:
    def test_refresh_all(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        )
        store.add(feat)
        results = store.refresh()
        assert "adj_em" in results
        assert results["adj_em"] == "refreshed"

    def test_refresh_specific_source(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        )
        store.add(feat)
        results = store.refresh(sources=["kenpom"])
        assert "adj_em" in results


# ------------------------------------------------------------------
# Save registry
# ------------------------------------------------------------------


class TestSaveRegistry:
    def test_save_persists_to_config(self, store: FeatureStore):
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        )
        store.add(feat)
        store.save_registry()
        assert "adj_em" in store.config.feature_defs


# ------------------------------------------------------------------
# Source loading
# ------------------------------------------------------------------


class TestSourceLoading:
    def test_load_csv_source(self, project_dir, config):
        """Source with CSV format loads correctly."""
        rng = np.random.default_rng(42)
        csv_path = project_dir / "data" / "raw" / "extra.csv"
        df = pd.DataFrame({
            "entity_id": list(range(1, 21)) * 3,
            "period_id": [2022] * 20 + [2023] * 20 + [2024] * 20,
            "extra_stat": rng.standard_normal(60),
        })
        df.to_csv(csv_path, index=False)

        config.sources["extra"] = SourceConfig(
            name="extra",
            path="data/raw/extra.csv",
            format="csv",
        )
        st = FeatureStore(project_dir, config)
        feat = FeatureDef(
            name="extra_stat", type=FeatureType.TEAM,
            source="extra", column="extra_stat",
        )
        result = st.add(feat)
        assert result.name == "extra_stat"

    def test_auto_format_detection(self, store: FeatureStore):
        """Auto format detection for .parquet files."""
        # kenpom source has format="parquet", this tests the path
        feat = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        )
        result = store.add(feat)
        assert result.name == "adj_em"

    def test_source_no_path_error(self, project_dir, config):
        config.sources["nopath"] = SourceConfig(name="nopath")
        st = FeatureStore(project_dir, config)
        feat = FeatureDef(
            name="x", type=FeatureType.TEAM,
            source="nopath", column="x",
        )
        with pytest.raises(ValueError, match="no path"):
            st.add(feat)
