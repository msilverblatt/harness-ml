"""Tests for PipelineRunner."""

import json
from pathlib import Path

import pandas as pd
import pytest

from harness.data.runner import PipelineRunner, PipelineResult


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "clean").mkdir(parents=True)
    return tmp_path


@pytest.fixture
def sample_csv(workspace):
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "label": ["a", "b", "c"],
        }
    )
    path = workspace / "data" / "raw" / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_single_source_no_transforms(workspace, sample_csv):
    runner = PipelineRunner(workspace)
    result = runner.run(
        sources=[{"name": "s1", "source_type": "file", "path": str(sample_csv)}],
        transforms=[],
    )
    assert isinstance(result, PipelineResult)
    assert result.row_count == 3
    assert result.column_count == 3
    assert set(result.columns) == {"id", "value", "label"}
    assert Path(result.output_path).exists()
    assert Path(result.schema_path).exists()


def test_output_parquet_is_readable(workspace, sample_csv):
    runner = PipelineRunner(workspace)
    result = runner.run(
        sources=[{"name": "s1", "source_type": "file", "path": str(sample_csv)}],
        transforms=[],
    )
    loaded = pd.read_parquet(result.output_path)
    assert len(loaded) == 3
    assert list(loaded.columns) == ["id", "value", "label"]


def test_with_transforms(workspace, sample_csv):
    runner = PipelineRunner(workspace)
    result = runner.run(
        sources=[{"name": "s1", "source_type": "file", "path": str(sample_csv)}],
        transforms=[
            {"op": "select", "params": {"columns": ["id", "value"]}},
        ],
    )
    assert result.column_count == 2
    assert set(result.columns) == {"id", "value"}


def test_schema_json_contents(workspace, sample_csv):
    runner = PipelineRunner(workspace)
    result = runner.run(
        sources=[{"name": "s1", "source_type": "file", "path": str(sample_csv)}],
        transforms=[],
    )
    schema = json.loads(Path(result.schema_path).read_text())
    assert schema["row_count"] == 3
    assert schema["column_count"] == 3
    assert "columns" in schema
    assert "column_types" in schema
    assert "data_hash" in schema
    assert len(schema["data_hash"]) == 64  # SHA256 hex


def test_data_hash_in_result(workspace, sample_csv):
    runner = PipelineRunner(workspace)
    result = runner.run(
        sources=[{"name": "s1", "source_type": "file", "path": str(sample_csv)}],
        transforms=[],
    )
    assert len(result.data_hash) == 64


def test_staging_failure_preserves_previous_outputs(workspace, sample_csv, monkeypatch):
    runner = PipelineRunner(workspace)
    source = [{"name": "s1", "source_type": "file", "path": str(sample_csv)}]
    first = runner.run(sources=source, transforms=[])
    output = Path(first.output_path)
    schema = Path(first.schema_path)
    old_output = output.read_bytes()
    old_schema = schema.read_bytes()

    def fail_schema_write(path, value):
        raise OSError("simulated write failure")

    monkeypatch.setattr("harness.data.runner.atomic_write_text", fail_schema_write)
    with pytest.raises(OSError, match="simulated write failure"):
        runner.run(sources=source, transforms=[])

    assert output.read_bytes() == old_output
    assert schema.read_bytes() == old_schema
    assert not list(output.parent.glob(".*.tmp"))


def test_error_on_bad_source_type(workspace):
    runner = PipelineRunner(workspace)
    with pytest.raises(ValueError, match="Unknown source_type"):
        runner.run(
            sources=[
                {"name": "s1", "source_type": "nonexistent", "path": "/fake/path.csv"}
            ],
            transforms=[],
        )


def test_error_on_no_sources(workspace):
    runner = PipelineRunner(workspace)
    with pytest.raises(ValueError, match="At least one source"):
        runner.run(sources=[], transforms=[])


def test_idempotent_output(workspace, sample_csv):
    runner = PipelineRunner(workspace)
    result1 = runner.run(
        sources=[{"name": "s1", "source_type": "file", "path": str(sample_csv)}],
        transforms=[],
    )
    result2 = runner.run(
        sources=[{"name": "s1", "source_type": "file", "path": str(sample_csv)}],
        transforms=[],
    )
    assert result1.data_hash == result2.data_hash
    assert result1.row_count == result2.row_count


def test_schema_path_is_in_result(workspace, sample_csv):
    runner = PipelineRunner(workspace)
    result = runner.run(
        sources=[{"name": "s1", "source_type": "file", "path": str(sample_csv)}],
        transforms=[],
    )
    assert result.schema_path.endswith("schema.json")
    assert result.output_path.endswith("dataset.parquet")


def test_multiple_sources_concat(workspace):
    """Two sources with no common columns are concatenated."""
    df1 = pd.DataFrame({"x": [1, 2]})
    df2 = pd.DataFrame({"y": [3, 4]})
    p1 = workspace / "data" / "raw" / "a.csv"
    p2 = workspace / "data" / "raw" / "b.csv"
    df1.to_csv(p1, index=False)
    df2.to_csv(p2, index=False)

    runner = PipelineRunner(workspace)
    result = runner.run(
        sources=[
            {"name": "s1", "source_type": "file", "path": str(p1)},
            {"name": "s2", "source_type": "file", "path": str(p2)},
        ],
        transforms=[],
    )
    # concat: 4 rows (2 + 2), 2 columns
    assert result.row_count == 4
    assert result.column_count == 2
