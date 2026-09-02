"""Tests for DataWorkspace."""

from pathlib import Path

import pandas as pd
import pytest

from harness.data.workspace import DataWorkspace


@pytest.fixture
def ws(tmp_path):
    return DataWorkspace(tmp_path)


@pytest.fixture
def sample_csv(tmp_path):
    """Write a CSV file outside the workspace (absolute path)."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10.0, 20.0, 30.0],
            "label": ["a", "b", "c"],
        }
    )
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_init_creates_directory_structure(ws, tmp_path):
    ws.init()
    assert (tmp_path / "data" / "raw").exists()
    assert (tmp_path / "data" / "clean").exists()
    assert (tmp_path / "data" / "sources.yaml").exists()
    assert (tmp_path / "data" / "transforms.yaml").exists()


def test_init_idempotent(ws):
    ws.init()
    ws.init()  # Should not raise


def test_add_source(ws, sample_csv):
    ws.init()
    ws.add_source("my_source", str(sample_csv))
    sources = ws.list_sources()
    assert len(sources) == 1
    assert sources[0].name == "my_source"
    assert sources[0].path == str(sample_csv)


def test_list_sources_empty(ws):
    ws.init()
    assert ws.list_sources() == []


def test_add_multiple_sources(ws, sample_csv):
    ws.init()
    ws.add_source("source_a", str(sample_csv))
    ws.add_source("source_b", str(sample_csv))
    sources = ws.list_sources()
    assert len(sources) == 2
    names = [s.name for s in sources]
    assert "source_a" in names
    assert "source_b" in names


def test_add_transform(ws):
    ws.init()
    ws.add_transform({"op": "select", "params": {"columns": ["id"]}})
    steps = ws.load_transforms()
    assert len(steps) == 1
    assert steps[0]["op"] == "select"


def test_load_transforms_empty(ws):
    ws.init()
    assert ws.load_transforms() == []


def test_add_multiple_transforms(ws):
    ws.init()
    ws.add_transform({"op": "select", "params": {"columns": ["id", "value"]}})
    ws.add_transform({"op": "sort", "params": {"by": "id"}})
    steps = ws.load_transforms()
    assert len(steps) == 2
    assert steps[0]["op"] == "select"
    assert steps[1]["op"] == "sort"


def test_run_pipeline(ws, sample_csv):
    ws.init()
    ws.add_source("main", str(sample_csv))
    result = ws.run_pipeline()
    assert result.row_count == 3
    assert result.column_count == 3


def test_run_pipeline_with_transforms(ws, sample_csv):
    ws.init()
    ws.add_source("main", str(sample_csv))
    ws.add_transform({"op": "select", "params": {"columns": ["id", "value"]}})
    result = ws.run_pipeline()
    assert result.column_count == 2
    assert set(result.columns) == {"id", "value"}


def test_load_clean_data(ws, sample_csv):
    ws.init()
    ws.add_source("main", str(sample_csv))
    ws.run_pipeline()
    df = ws.load_clean_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_load_clean_data_not_found(ws):
    ws.init()
    with pytest.raises(FileNotFoundError):
        ws.load_clean_data()


def test_load_schema(ws, sample_csv):
    ws.init()
    ws.add_source("main", str(sample_csv))
    ws.run_pipeline()
    schema = ws.load_schema()
    assert isinstance(schema, dict)
    assert "row_count" in schema
    assert "column_count" in schema
    assert "data_hash" in schema


def test_load_schema_rejects_dataset_sidecar_mismatch(ws, sample_csv):
    ws.init()
    ws.add_source("main", str(sample_csv))
    ws.run_pipeline()
    dataset = ws._root / "data" / "clean" / "dataset.parquet"
    dataset.write_bytes(dataset.read_bytes() + b"changed")

    with pytest.raises(RuntimeError, match="inconsistent"):
        ws.load_schema()


def test_load_schema_not_found(ws):
    ws.init()
    with pytest.raises(FileNotFoundError):
        ws.load_schema()


def test_sources_yaml_persisted(ws, sample_csv):
    """Sources persist across separate DataWorkspace instances."""
    ws.init()
    ws.add_source("persistent", str(sample_csv))

    ws2 = DataWorkspace(ws._root)
    sources = ws2.list_sources()
    assert len(sources) == 1
    assert sources[0].name == "persistent"


def test_transforms_yaml_persisted(ws):
    """Transforms persist across separate DataWorkspace instances."""
    ws.init()
    ws.add_transform({"op": "select", "params": {"columns": ["x"]}})

    ws2 = DataWorkspace(ws._root)
    steps = ws2.load_transforms()
    assert len(steps) == 1
