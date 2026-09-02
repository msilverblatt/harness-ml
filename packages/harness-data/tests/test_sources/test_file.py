import pandas as pd
import pytest
from pathlib import Path

from harness.data.sources.file import FileSource
from harness.data.sources.protocol import SourceConfig


class TestFileSource:
    def test_load_csv(self, sample_csv):
        source = FileSource()
        config = SourceConfig(name="test", path=str(sample_csv))
        df = source.load(config)
        assert len(df) == 5
        assert "score" in df.columns

    def test_load_parquet(self, temp_workspace, sample_df):
        path = temp_workspace / "data" / "raw" / "sample.parquet"
        sample_df.to_parquet(path, index=False)
        source = FileSource()
        config = SourceConfig(name="test", path=str(path))
        df = source.load(config)
        assert len(df) == 5

    def test_load_csv_with_base_dir(self, temp_workspace, sample_csv):
        source = FileSource()
        config = SourceConfig(name="test", path="data/raw/sample.csv")
        df = source.load(config, base_dir=str(temp_workspace))
        assert len(df) == 5

    def test_auto_detect_format(self, sample_csv):
        source = FileSource()
        config = SourceConfig(name="test", path=str(sample_csv), format="auto")
        df = source.load(config)
        assert len(df) == 5

    def test_validate_missing_path(self):
        source = FileSource()
        config = SourceConfig(name="test", path=None)
        errors = source.validate(config)
        assert len(errors) > 0
        assert "path" in errors[0].lower()

    def test_validate_nonexistent_file(self):
        source = FileSource()
        config = SourceConfig(name="test", path="/nonexistent/file.csv")
        errors = source.validate(config)
        assert len(errors) > 0

    def test_validate_valid_file(self, sample_csv):
        source = FileSource()
        config = SourceConfig(name="test", path=str(sample_csv))
        errors = source.validate(config)
        assert len(errors) == 0

    def test_implements_source_protocol(self):
        from harness.data.sources.protocol import Source
        assert isinstance(FileSource(), Source)
