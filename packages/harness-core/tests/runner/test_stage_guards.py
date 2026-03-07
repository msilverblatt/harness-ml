"""Tests for PipelineGuards — stage prerequisite validation."""
from __future__ import annotations

import time

import pandas as pd
import pytest
from harnessml.core.runner.guards import GuardrailViolationError
from harnessml.core.runner.schema import DataConfig
from harnessml.core.runner.stage_guards import PipelineGuards


class TestGuardTrain:
    """guard_train() validates feature file existence."""

    def test_passes_with_features(self, tmp_path):
        """Features parquet exists, guard passes."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        df = pd.DataFrame({"a": range(20)})
        df.to_parquet(features_dir / "features.parquet")

        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(features_dir)
        )
        guards = PipelineGuards(data_config, tmp_path)
        guards.guard_train()  # Should not raise

    def test_fails_missing_features(self, tmp_path):
        """No parquet, raises GuardrailViolationError."""
        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(tmp_path / "features")
        )
        guards = PipelineGuards(data_config, tmp_path)
        with pytest.raises(GuardrailViolationError):
            guards.guard_train()


class TestGuardPredict:
    """guard_predict() validates features and optional models dir."""

    def test_passes_with_features(self, tmp_path):
        """Features parquet exists, guard passes without models_dir."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        df = pd.DataFrame({"a": range(20)})
        df.to_parquet(features_dir / "features.parquet")

        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(features_dir)
        )
        guards = PipelineGuards(data_config, tmp_path)
        guards.guard_predict()  # Should not raise

    def test_fails_missing_models_dir(self, tmp_path):
        """Features exist but models_dir doesn't, raises error."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        df = pd.DataFrame({"a": range(20)})
        df.to_parquet(features_dir / "features.parquet")

        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(features_dir)
        )
        guards = PipelineGuards(data_config, tmp_path)
        with pytest.raises(GuardrailViolationError):
            guards.guard_predict(models_dir=tmp_path / "nonexistent_models")


class TestGuardBacktest:
    """guard_backtest() validates features exist with sufficient rows."""

    def test_passes_with_sufficient_rows(self, tmp_path):
        """Parquet with >= 10 rows passes."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        df = pd.DataFrame({"a": range(20)})
        df.to_parquet(features_dir / "features.parquet")

        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(features_dir)
        )
        guards = PipelineGuards(data_config, tmp_path)
        guards.guard_backtest()  # Should not raise

    def test_fails_insufficient_rows(self, tmp_path):
        """Parquet exists but too few rows."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        df = pd.DataFrame({"a": range(5)})  # < 10 rows
        df.to_parquet(features_dir / "features.parquet")

        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(features_dir)
        )
        guards = PipelineGuards(data_config, tmp_path)
        with pytest.raises(GuardrailViolationError):
            guards.guard_backtest()

    def test_fails_missing_parquet(self, tmp_path):
        """No parquet at all, raises error."""
        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(tmp_path / "features")
        )
        guards = PipelineGuards(data_config, tmp_path)
        with pytest.raises(GuardrailViolationError):
            guards.guard_backtest()


class TestGuardsDisabled:
    """enabled=False makes all guard checks no-ops."""

    def test_all_guards_noop_when_disabled(self, tmp_path):
        """All guards pass even without any files when disabled."""
        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(tmp_path / "features")
        )
        guards = PipelineGuards(data_config, tmp_path, enabled=False)
        guards.guard_train()
        guards.guard_predict()
        guards.guard_backtest()


class TestWarnIfStale:
    """warn_if_stale() detects staleness via file modification times."""

    def test_stale_artifact(self, tmp_path):
        """Artifact older than source returns True."""
        artifact = tmp_path / "old_artifact.parquet"
        source = tmp_path / "new_source.yaml"

        # Create artifact first
        artifact.write_text("old")
        time.sleep(0.05)
        # Create source after artifact
        source.write_text("new")

        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(tmp_path / "features")
        )
        guards = PipelineGuards(data_config, tmp_path)
        result = guards.warn_if_stale(
            artifacts=[str(artifact)],
            sources=[str(source)],
            stage_name="test",
        )
        assert result is True

    def test_fresh_artifact(self, tmp_path):
        """Artifact newer than source returns False."""
        source = tmp_path / "old_source.yaml"
        artifact = tmp_path / "new_artifact.parquet"

        # Create source first
        source.write_text("old")
        time.sleep(0.05)
        # Create artifact after source
        artifact.write_text("new")

        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(tmp_path / "features")
        )
        guards = PipelineGuards(data_config, tmp_path)
        result = guards.warn_if_stale(
            artifacts=[str(artifact)],
            sources=[str(source)],
            stage_name="test",
        )
        assert result is False

    def test_disabled_returns_false(self, tmp_path):
        """Disabled guards always return False for staleness."""
        data_config = DataConfig(
            raw_dir="raw", processed_dir="proc", features_dir=str(tmp_path / "features")
        )
        guards = PipelineGuards(data_config, tmp_path, enabled=False)
        result = guards.warn_if_stale(
            artifacts=["nonexistent"],
            sources=["also_nonexistent"],
            stage_name="test",
        )
        assert result is False


class TestConfigDrivenFeatureFile:
    """Guards use DataConfig.features_file instead of hardcoded name."""

    def test_custom_features_file(self, tmp_path):
        """Guard finds parquet using config's features_file."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        df = pd.DataFrame({"a": range(20)})
        df.to_parquet(features_dir / "my_data.parquet")

        data_config = DataConfig(
            raw_dir="raw",
            processed_dir="proc",
            features_dir=str(features_dir),
            features_file="my_data.parquet",
        )
        guards = PipelineGuards(data_config, tmp_path)
        guards.guard_train()  # Should not raise

    def test_default_features_file(self, tmp_path):
        """Guard uses 'features.parquet' as default."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        df = pd.DataFrame({"a": range(20)})
        df.to_parquet(features_dir / "features.parquet")

        data_config = DataConfig(
            raw_dir="raw",
            processed_dir="proc",
            features_dir=str(features_dir),
        )
        guards = PipelineGuards(data_config, tmp_path)
        guards.guard_train()  # Should not raise


class TestRelativePathResolution:
    """Guards resolve relative features_dir against project_dir."""

    def test_relative_features_dir(self, tmp_path):
        """Relative features_dir is resolved against project_dir."""
        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)
        df = pd.DataFrame({"a": range(20)})
        df.to_parquet(features_dir / "features.parquet")

        data_config = DataConfig(
            raw_dir="data/raw",
            processed_dir="data/processed",
            features_dir="data/features",
        )
        guards = PipelineGuards(data_config, tmp_path)
        guards.guard_train()  # Should not raise
