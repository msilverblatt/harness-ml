"""Tests for DataPipeline orchestrator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.core.runner.schema import ColumnCleaningRule, DataConfig, SourceConfig


class TestDataPipelineRefresh:
    """DataPipeline.refresh() processes declared sources."""

    @pytest.fixture
    def project_with_source(self, tmp_path):
        """Project dir with a CSV source and pipeline config."""
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "team_id": [1, 2, 3],
            "season": [2020, 2020, 2020],
            "adj_oe": [110.5, 105.2, np.nan],
            "adj_de": ["98.1", "102.3", "95.0"],
        })
        df.to_csv(raw_dir / "kenpom.csv", index=False)

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config = {
            "data": {
                "features_dir": "data/features",
                "features_file": "features.parquet",
                "target_column": "outcome",
                "key_columns": ["team_id", "season"],
                "sources": {
                    "kenpom": {
                        "name": "kenpom",
                        "path": "data/raw/kenpom.csv",
                        "format": "csv",
                        "join_on": ["team_id", "season"],
                        "temporal_safety": "pre_event",
                        "columns": {
                            "adj_oe": {"null_strategy": "zero"},
                            "adj_de": {"coerce_numeric": True},
                        },
                    },
                },
            },
        }
        (config_dir / "pipeline.yaml").write_text(yaml.dump(config))
        return tmp_path

    def test_refresh_bootstrap(self, project_with_source):
        from easyml.core.runner.data_pipeline import DataPipeline
        from easyml.core.runner.data_utils import load_data_config

        config = load_data_config(project_with_source)
        pipeline = DataPipeline(project_with_source, config)
        result = pipeline.refresh()

        assert result.sources_processed == 1
        features_path = project_with_source / "data" / "features" / "features.parquet"
        assert features_path.exists()

        df = pd.read_parquet(features_path)
        assert len(df) == 3
        assert df["adj_oe"].isna().sum() == 0
        assert pd.api.types.is_numeric_dtype(df["adj_de"])

    def test_refresh_specific_source(self, project_with_source):
        from easyml.core.runner.data_pipeline import DataPipeline
        from easyml.core.runner.data_utils import load_data_config

        config = load_data_config(project_with_source)
        pipeline = DataPipeline(project_with_source, config)
        result = pipeline.refresh(sources=["kenpom"])
        assert result.sources_processed == 1

    def test_refresh_nonexistent_source_error(self, project_with_source):
        from easyml.core.runner.data_pipeline import DataPipeline
        from easyml.core.runner.data_utils import load_data_config

        config = load_data_config(project_with_source)
        pipeline = DataPipeline(project_with_source, config)
        with pytest.raises(ValueError, match="not_real"):
            pipeline.refresh(sources=["not_real"])

    def test_refresh_disabled_source_skipped(self, tmp_path):
        """Disabled sources are skipped during refresh."""
        from easyml.core.runner.data_pipeline import DataPipeline

        config = DataConfig(
            sources={
                "disabled_src": SourceConfig(name="disabled_src", path="fake.csv", enabled=False),
            },
        )
        pipeline = DataPipeline(tmp_path, config)
        result = pipeline.refresh()
        assert result.sources_processed == 0


class TestCleaningRuleCascade:
    """Cleaning rules cascade: column > source > global."""

    def test_column_overrides_source_default(self):
        from easyml.core.runner.data_pipeline import resolve_cleaning_rule

        source = SourceConfig(
            name="test",
            default_cleaning=ColumnCleaningRule(null_strategy="median"),
            columns={"col_a": ColumnCleaningRule(null_strategy="zero")},
        )
        global_default = ColumnCleaningRule(null_strategy="mode")

        rule = resolve_cleaning_rule("col_a", source, global_default)
        assert rule.null_strategy == "zero"

    def test_source_default_used_when_no_column_rule(self):
        from easyml.core.runner.data_pipeline import resolve_cleaning_rule

        source = SourceConfig(
            name="test",
            default_cleaning=ColumnCleaningRule(null_strategy="median"),
        )
        global_default = ColumnCleaningRule(null_strategy="mode")

        rule = resolve_cleaning_rule("col_b", source, global_default)
        assert rule.null_strategy == "median"

    def test_global_used_as_final_fallback(self):
        from easyml.core.runner.data_pipeline import resolve_cleaning_rule

        source = SourceConfig(name="test")
        global_default = ColumnCleaningRule(null_strategy="mode")

        rule = resolve_cleaning_rule("col_c", source, global_default)
        assert rule.null_strategy == "mode"


class TestApplyCleaningRule:
    """apply_cleaning_rule applies transformations correctly."""

    def test_coerce_numeric(self):
        from easyml.core.runner.data_pipeline import apply_cleaning_rule

        s = pd.Series(["1.5", "2.3", "bad", "4.0"])
        rule = ColumnCleaningRule(coerce_numeric=True, null_strategy="zero")
        result = apply_cleaning_rule(s, rule)
        assert pd.api.types.is_numeric_dtype(result)
        assert result.isna().sum() == 0  # "bad" coerced to NaN then filled with zero

    def test_null_fill_median(self):
        from easyml.core.runner.data_pipeline import apply_cleaning_rule

        s = pd.Series([1.0, 2.0, np.nan, 4.0])
        rule = ColumnCleaningRule(null_strategy="median")
        result = apply_cleaning_rule(s, rule)
        assert result.isna().sum() == 0
        assert result.iloc[2] == 2.0  # median of [1, 2, 4] = 2

    def test_null_fill_zero(self):
        from easyml.core.runner.data_pipeline import apply_cleaning_rule

        s = pd.Series([1.0, np.nan, 3.0])
        rule = ColumnCleaningRule(null_strategy="zero")
        result = apply_cleaning_rule(s, rule)
        assert result.iloc[1] == 0.0

    def test_null_fill_constant(self):
        from easyml.core.runner.data_pipeline import apply_cleaning_rule

        s = pd.Series([1.0, np.nan, 3.0])
        rule = ColumnCleaningRule(null_strategy="constant", null_fill_value=-999)
        result = apply_cleaning_rule(s, rule)
        assert result.iloc[1] == -999

    def test_zscore_normalize(self):
        from easyml.core.runner.data_pipeline import apply_cleaning_rule

        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        rule = ColumnCleaningRule(normalize="zscore")
        result = apply_cleaning_rule(s, rule)
        assert abs(result.mean()) < 0.01
        assert abs(result.std() - 1.0) < 0.01

    def test_minmax_normalize(self):
        from easyml.core.runner.data_pipeline import apply_cleaning_rule

        s = pd.Series([10.0, 20.0, 30.0])
        rule = ColumnCleaningRule(normalize="minmax")
        result = apply_cleaning_rule(s, rule)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_clip_outliers(self):
        from easyml.core.runner.data_pipeline import apply_cleaning_rule

        s = pd.Series(list(range(100)) + [1000])
        rule = ColumnCleaningRule(clip_outliers=(1.0, 99.0))
        result = apply_cleaning_rule(s, rule)
        assert result.max() < 1000

    def test_log_transform(self):
        from easyml.core.runner.data_pipeline import apply_cleaning_rule

        s = pd.Series([1.0, 10.0, 100.0])
        rule = ColumnCleaningRule(log_transform=True)
        result = apply_cleaning_rule(s, rule)
        assert result.iloc[0] < s.iloc[0]  # log1p(1) < 1... actually log1p(1) = 0.693 < 1


class TestAddSource:
    """DataPipeline.add_source() registers and ingests."""

    def test_add_source(self, tmp_path):
        from easyml.core.runner.data_pipeline import DataPipeline

        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        df = pd.DataFrame({"id": [1, 2], "y": [0, 1], "feat": [1.0, 2.0]})
        df.to_csv(raw_dir / "new_source.csv", index=False)

        config = DataConfig(target_column="y")
        pipeline = DataPipeline(tmp_path, config)
        source_config = pipeline.add_source("new_data", "data/raw/new_source.csv")

        assert source_config.name == "new_data"
        features_path = tmp_path / "data" / "features" / "features.parquet"
        assert features_path.exists()


class TestRemoveSource:
    """DataPipeline.remove_source() removes tracked source."""

    def test_remove_source(self, tmp_path):
        from easyml.core.runner.data_pipeline import DataPipeline

        config = DataConfig(
            sources={
                "old": SourceConfig(name="old", path="data/raw/old.csv"),
            },
        )
        pipeline = DataPipeline(tmp_path, config)
        pipeline.remove_source("old")
        assert "old" not in pipeline.config.sources
