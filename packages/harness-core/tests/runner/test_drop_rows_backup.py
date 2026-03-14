"""Tests for drop_rows backup and cache invalidation.

drop_rows creates a backup (features_pre_drop.parquet) and auto-invalidates
the prediction cache to prevent length mismatches on backtest.
"""
from __future__ import annotations

import pandas as pd
import pytest
from harnessml.core.runner.training.prediction_cache import PredictionCache


@pytest.fixture()
def project(tmp_path):
    """Set up a minimal project structure with features and config."""
    project_dir = tmp_path / "project"
    features_dir = project_dir / "data" / "features"
    features_dir.mkdir(parents=True)
    config_dir = project_dir / "config"
    config_dir.mkdir(parents=True)

    # Write minimal pipeline.yaml so load_data_config works
    (config_dir / "pipeline.yaml").write_text(
        "data:\n  features_dir: data/features\n  features_file: features.parquet\n"
    )

    # Create features with some null rows
    df = pd.DataFrame({
        "feature_a": [1.0, 2.0, None, 4.0, 5.0],
        "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    df.to_parquet(features_dir / "features.parquet", index=False)

    return project_dir


@pytest.fixture()
def project_with_cache(project):
    """Project with a populated prediction cache."""
    cache_dir = project / ".cache" / "predictions"
    cache = PredictionCache(cache_dir)

    # Store some cached predictions (5 rows to match original data)
    preds = pd.DataFrame({"prediction": [0.1, 0.2, 0.3, 0.4, 0.5]})
    cache.store("model_a", 2024, "fp_abc", preds)
    cache.store("model_b", 2024, "fp_def", preds)

    return project


# -----------------------------------------------------------------------
# Backup creation
# -----------------------------------------------------------------------

class TestDropRowsBackup:
    def test_creates_backup_on_first_drop(self, project):
        from harnessml.core.runner.data.ingest import drop_rows

        drop_rows(project, column="feature_a", condition="null")

        backup = project / "data" / "features" / "features_pre_drop.parquet"
        assert backup.exists(), "drop_rows should create features_pre_drop.parquet"

        # Backup should have original row count
        backup_df = pd.read_parquet(backup)
        assert len(backup_df) == 5

    def test_does_not_overwrite_existing_backup(self, project):
        from harnessml.core.runner.data.ingest import drop_rows

        # First drop
        drop_rows(project, column="feature_a", condition="null")
        backup = project / "data" / "features" / "features_pre_drop.parquet"
        first_backup_df = pd.read_parquet(backup)

        # Modify features and drop again
        features_path = project / "data" / "features" / "features.parquet"
        df = pd.read_parquet(features_path)
        assert len(df) == 4  # 1 row dropped

        # Second drop — backup should still have original 5 rows
        drop_rows(project, condition="feature_b > 40")
        backup_df = pd.read_parquet(backup)
        assert len(backup_df) == 5, "Backup should preserve original data, not intermediate"

    def test_features_reduced_after_drop(self, project):
        from harnessml.core.runner.data.ingest import drop_rows

        result = drop_rows(project, column="feature_a", condition="null")

        features_path = project / "data" / "features" / "features.parquet"
        df = pd.read_parquet(features_path)
        assert len(df) == 4  # 1 null row dropped
        assert "Dropped 1 rows" in result

    def test_no_drop_when_no_match(self, project):
        from harnessml.core.runner.data.ingest import drop_rows

        result = drop_rows(project, condition="feature_b > 999")

        assert "nothing dropped" in result.lower()
        # No backup should be created if nothing was dropped
        backup = project / "data" / "features" / "features_pre_drop.parquet"
        assert not backup.exists()

    def test_result_mentions_backup(self, project):
        from harnessml.core.runner.data.ingest import drop_rows

        result = drop_rows(project, column="feature_a", condition="null")
        assert "backup" in result.lower() or "features_pre_drop" in result.lower()


# -----------------------------------------------------------------------
# Cache invalidation
# -----------------------------------------------------------------------

class TestDropRowsCacheInvalidation:
    def test_clears_prediction_cache(self, project_with_cache):
        from harnessml.core.runner.data.ingest import drop_rows

        cache_dir = project_with_cache / ".cache" / "predictions"
        cache = PredictionCache(cache_dir)

        # Verify cache is populated
        assert cache.lookup("model_a", 2024, "fp_abc") is not None

        # Drop rows — should invalidate cache
        drop_rows(project_with_cache, column="feature_a", condition="null")

        # Cache should be empty now
        assert cache.lookup("model_a", 2024, "fp_abc") is None
        assert cache.lookup("model_b", 2024, "fp_def") is None

    def test_no_error_when_no_cache_exists(self, project):
        """drop_rows should not fail if there's no cache directory."""
        from harnessml.core.runner.data.ingest import drop_rows

        cache_dir = project / ".cache" / "predictions"
        assert not cache_dir.exists()

        # Should not raise
        result = drop_rows(project, column="feature_a", condition="null")
        assert "Dropped" in result


# -----------------------------------------------------------------------
# Restore integration
# -----------------------------------------------------------------------

class TestRestoreAfterDrop:
    def test_restore_finds_pre_drop_backup(self, project):
        from harnessml.core.runner.data.ingest import drop_rows, restore_full_data

        drop_rows(project, column="feature_a", condition="null")

        features_path = project / "data" / "features" / "features.parquet"
        assert len(pd.read_parquet(features_path)) == 4

        result = restore_full_data(project)

        assert "Restored" in result
        assert "features_pre_drop" in result
        restored_df = pd.read_parquet(features_path)
        assert len(restored_df) == 5

    def test_restore_prefers_features_full_over_pre_drop(self, project):
        """features_full.parquet (from sample) takes priority over
        features_pre_drop.parquet (from drop_rows)."""
        from harnessml.core.runner.data.ingest import drop_rows, restore_full_data

        features_dir = project / "data" / "features"

        # Create a features_full.parquet (simulating prior sample backup)
        full_df = pd.DataFrame({
            "feature_a": range(100),
            "feature_b": range(100),
            "target": range(100),
        })
        full_df.to_parquet(features_dir / "features_full.parquet", index=False)

        # Now drop rows — also creates features_pre_drop.parquet
        drop_rows(project, column="feature_a", condition="null")

        result = restore_full_data(project)

        # Should restore from features_full.parquet (higher priority)
        assert "features_full" in result
        restored_df = pd.read_parquet(features_dir / "features.parquet")
        assert len(restored_df) == 100

    def test_restore_error_when_no_backup(self, project):
        from harnessml.core.runner.data.ingest import restore_full_data

        result = restore_full_data(project)
        assert "Error" in result
