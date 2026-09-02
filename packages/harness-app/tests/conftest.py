import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temp directory suitable for workspace operations."""
    return tmp_path / "test_workspace"


@pytest.fixture
def sample_binary_data(tmp_path):
    """Create a sample binary classification CSV and return its path."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
        "feature_c": rng.uniform(0, 1, n),
        "target": rng.randint(0, 2, n),
    })
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def initialized_workspace(tmp_workspace):
    """Create and return an initialized workspace with harness.yaml and config."""
    from harness.app.workspace.manager import WorkspaceManager
    ws = WorkspaceManager.init(tmp_workspace)
    return ws
