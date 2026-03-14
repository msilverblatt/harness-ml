"""Tests for FeatureStore integration with PipelineRunner._load_data().

Verifies that when feature_defs are configured in the DataConfig,
the pipeline computes features via FeatureStore and adds them to _df.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from harnessml.core.runner.pipeline import PipelineRunner


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _make_entity_source(path: Path, n_entities: int, n_seasons: int, season_start: int) -> None:
    """Create entity-level source data with entity_id, period_id, and a metric column."""
    rng = np.random.default_rng(99)
    rows = []
    for season in range(season_start, season_start + n_seasons):
        for eid in range(1, n_entities + 1):
            rows.append({
                "entity_id": eid,
                "period_id": season,
                "adj_em": rng.standard_normal(),
            })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _make_matchup_features(
    path: Path,
    n_rows: int,
    n_entities: int,
    n_seasons: int,
    season_start: int,
    extra_columns: dict | None = None,
) -> None:
    """Create matchup-level features.parquet with entity_a_id, entity_b_id, period_id, result."""
    rng = np.random.default_rng(42)
    seasons = rng.choice(
        list(range(season_start, season_start + n_seasons)),
        size=n_rows,
    )
    data = {
        "entity_a_id": rng.integers(1, n_entities + 1, size=n_rows),
        "entity_b_id": rng.integers(1, n_entities + 1, size=n_rows),
        "period_id": seasons,
        "season": seasons,
        "result": rng.integers(0, 2, size=n_rows),
        "margin": rng.standard_normal(n_rows) * 10,
        "diff_prior": rng.integers(-15, 16, size=n_rows).astype(float),
        "existing_feat": rng.standard_normal(n_rows),
    }
    if extra_columns:
        for col_name, values in extra_columns.items():
            data[col_name] = values if len(values) == n_rows else rng.standard_normal(n_rows)
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


class TestFeatureStorePipelineWiring:
    """Verify that PipelineRunner._load_data() integrates with FeatureStore."""

    def test_team_feature_computed_via_store(self, tmp_path):
        """When feature_defs contain a team feature, the pipeline computes
        pairwise derivatives and adds them to _df."""
        n_entities = 10
        n_seasons = 3
        season_start = 2022
        n_rows = 100

        # Create entity-level source data
        source_path = tmp_path / "data" / "raw" / "kenpom.parquet"
        _make_entity_source(source_path, n_entities, n_seasons, season_start)

        # Create matchup-level features.parquet
        features_path = tmp_path / "data" / "features" / "features.parquet"
        _make_matchup_features(
            features_path, n_rows, n_entities, n_seasons, season_start,
        )

        # Write config with feature_defs referencing the source
        config_dir = tmp_path / "config"

        _write_yaml(config_dir / "pipeline.yaml", {
            "data": {
                "raw_dir": str(tmp_path / "data" / "raw"),
                "processed_dir": str(tmp_path / "data" / "processed"),
                "features_dir": str(tmp_path / "data" / "features"),
                "sources": {
                    "kenpom": {
                        "name": "kenpom",
                        "path": str(source_path),
                        "format": "parquet",
                    },
                },
                "feature_defs": {
                    "adj_em": {
                        "name": "adj_em",
                        "type": "entity",
                        "source": "kenpom",
                        "column": "adj_em",
                        "pairwise_mode": "diff",
                        "category": "efficiency",
                    },
                },
                "feature_store": {
                    "cache_dir": str(tmp_path / "data" / "features" / "cache"),
                    "auto_pairwise": True,
                    "entity_a_column": "entity_a_id",
                    "entity_b_column": "entity_b_id",
                    "entity_column": "entity_id",
                    "period_column": "period_id",
                },
            },
            "backtest": {
                "cv_strategy": "leave_one_out",
                "fold_column": "season",
                "fold_values": [2022, 2023, 2024],
                "metrics": ["brier"],
            },
        })

        _write_yaml(config_dir / "models.yaml", {
            "models": {
                "logreg_test": {
                    "type": "logistic_regression",
                    "features": ["diff_prior", "diff_adj_em"],
                    "params": {"C": 1.0, "max_iter": 200},
                },
            },
        })

        _write_yaml(config_dir / "ensemble.yaml", {
            "ensemble": {
                "method": "average",
            },
        })

        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()

        # The team feature "adj_em" should have generated "diff_adj_em"
        # and it should now be in _df
        assert "diff_adj_em" in runner._df.columns, (
            f"Expected 'diff_adj_em' in _df columns. Got: {sorted(runner._df.columns.tolist())}"
        )
        # The column should have valid numeric data
        assert runner._df["diff_adj_em"].notna().any()

    def test_matchup_feature_computed_via_store(self, tmp_path):
        """When feature_defs contain a matchup feature referencing an existing column,
        the pipeline adds it to _df."""
        n_rows = 80

        # Create matchup-level features.parquet with an extra column
        np.random.default_rng(7)
        features_path = tmp_path / "data" / "features" / "features.parquet"
        _make_matchup_features(
            features_path, n_rows, n_entities=5, n_seasons=2, season_start=2023,
        )

        config_dir = tmp_path / "config"

        _write_yaml(config_dir / "pipeline.yaml", {
            "data": {
                "features_dir": str(tmp_path / "data" / "features"),
                "feature_defs": {
                    "existing_feat_renamed": {
                        "name": "existing_feat_renamed",
                        "type": "instance",
                        "column": "existing_feat",
                    },
                },
                "feature_store": {
                    "cache_dir": str(tmp_path / "data" / "features" / "cache"),
                },
            },
            "backtest": {
                "cv_strategy": "leave_one_out",
                "fold_column": "season",
                "fold_values": [2023, 2024],
                "metrics": ["brier"],
            },
        })

        _write_yaml(config_dir / "models.yaml", {
            "models": {
                "logreg_test": {
                    "type": "logistic_regression",
                    "features": ["diff_prior", "existing_feat_renamed"],
                    "params": {"C": 1.0, "max_iter": 200},
                },
            },
        })

        _write_yaml(config_dir / "ensemble.yaml", {
            "ensemble": {"method": "average"},
        })

        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()

        assert "existing_feat_renamed" in runner._df.columns, (
            f"Expected 'existing_feat_renamed' in _df columns. Got: {sorted(runner._df.columns.tolist())}"
        )

    def test_no_feature_defs_skips_store(self, tmp_path):
        """When feature_defs is empty, the FeatureStore is not invoked
        and _load_data works normally."""
        features_path = tmp_path / "data" / "features" / "features.parquet"
        _make_matchup_features(
            features_path, n_rows=50, n_entities=5, n_seasons=2, season_start=2023,
        )

        config_dir = tmp_path / "config"

        _write_yaml(config_dir / "pipeline.yaml", {
            "data": {
                "features_dir": str(tmp_path / "data" / "features"),
            },
            "backtest": {
                "cv_strategy": "leave_one_out",
                "fold_column": "season",
                "fold_values": [2023, 2024],
                "metrics": ["brier"],
            },
        })

        _write_yaml(config_dir / "models.yaml", {
            "models": {
                "logreg_test": {
                    "type": "logistic_regression",
                    "features": ["diff_prior"],
                    "params": {"C": 1.0, "max_iter": 200},
                },
            },
        })

        _write_yaml(config_dir / "ensemble.yaml", {
            "ensemble": {"method": "average"},
        })

        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()

        # Should load normally without error
        assert "diff_prior" in runner._df.columns
        assert "result" in runner._df.columns
