"""Tests for feature pruning from model configs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from harnessml.core.runner.config_writer.features import prune_features


class TestPruneFeatures:
    """Feature pruning based on importance threshold."""

    def _setup_project(self, tmp_path, features, models, target_col="target"):
        project_dir = tmp_path / "project"
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        features_dir = project_dir / "data" / "features"
        features_dir.mkdir(parents=True)

        pipeline = {
            "data": {
                "features_dir": "data/features",
                "features_file": "features.parquet",
                "target_column": target_col,
            }
        }
        (config_dir / "pipeline.yaml").write_text(yaml.dump(pipeline))
        (config_dir / "sources.yaml").write_text("guardrails: {}")
        (config_dir / "models.yaml").write_text(yaml.dump({"models": models}))

        features.to_parquet(features_dir / "features.parquet", index=False)
        return project_dir

    def test_dry_run_does_not_modify(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "important": rng.standard_normal(n),
            "useless": rng.standard_normal(n) * 0.001,
            "target": rng.integers(0, 2, n),
        })
        models = {
            "xgb": {"type": "xgboost", "features": ["important", "useless"]},
        }
        project_dir = self._setup_project(tmp_path, df, models)

        result = prune_features(project_dir, threshold=0.5, dry_run=True)
        assert "DRY RUN" in result

        # Models file should be unchanged
        models_data = yaml.safe_load((project_dir / "config" / "models.yaml").read_text())
        assert "useless" in models_data["models"]["xgb"]["features"]

    def test_apply_removes_features(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 200
        target = rng.integers(0, 2, n)
        df = pd.DataFrame({
            "strong": target + rng.standard_normal(n) * 0.1,
            "noise": rng.standard_normal(n),
            "target": target,
        })
        models = {
            "xgb": {"type": "xgboost", "features": ["strong", "noise"]},
        }
        project_dir = self._setup_project(tmp_path, df, models)

        result = prune_features(project_dir, threshold=0.3, dry_run=False)
        assert "APPLIED" in result

        models_data = yaml.safe_load((project_dir / "config" / "models.yaml").read_text())
        remaining = models_data["models"]["xgb"]["features"]
        assert "strong" in remaining

    def test_no_features_below_threshold(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 200
        target = rng.integers(0, 2, n)
        df = pd.DataFrame({
            "a": target + rng.standard_normal(n) * 0.1,
            "b": target + rng.standard_normal(n) * 0.2,
            "target": target,
        })
        models = {
            "xgb": {"type": "xgboost", "features": ["a", "b"]},
        }
        project_dir = self._setup_project(tmp_path, df, models)

        result = prune_features(project_dir, threshold=0.0, dry_run=True)
        assert "Nothing to prune" in result
