import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_df():
    """A simple DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [85.0, 92.0, 78.0, 95.0, 88.0],
        "grade": ["B", "A", "C", "A", "B"],
        "enrolled": [True, True, False, True, True],
    })


@pytest.fixture
def numeric_df():
    """DataFrame with numeric columns for transform testing."""
    return pd.DataFrame({
        "entity_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "period": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "points": [10.0, 15.0, 12.0, 20.0, 18.0, 22.0, 8.0, 9.0, 11.0],
        "rebounds": [5.0, 7.0, 6.0, 10.0, 9.0, 11.0, 3.0, 4.0, 5.0],
        "target": [1, 0, 1, 1, 1, 0, 0, 0, 1],
    })


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "clean").mkdir()
    return tmp_path


@pytest.fixture
def sample_csv(temp_workspace, sample_df):
    """Write sample_df to a CSV in the temp workspace."""
    path = temp_workspace / "data" / "raw" / "sample.csv"
    sample_df.to_csv(path, index=False)
    return path
