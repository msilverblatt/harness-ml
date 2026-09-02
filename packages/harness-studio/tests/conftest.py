import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient
from harness.studio.server import create_app
from harness.app.workspace.manager import WorkspaceManager


@pytest.fixture
def workspace(tmp_path):
    ws = WorkspaceManager.init(tmp_path, task_type="binary", target_column="target")
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    df["target"] = (df["a"] + rng.randn(n) * 0.5 > 0).astype(int)
    csv = tmp_path / "data" / "raw" / "data.csv"
    df.to_csv(csv, index=False)
    ws.data.add_source("main", str(csv))
    ws.data.run_pipeline()
    ws.run_experiment("baseline", "Baseline LR", {
        "models": {"lr": {"model_type": "logistic", "features": ["a", "b"]}},
    })
    return tmp_path


@pytest.fixture
def client(workspace):
    app = create_app(workspace)
    return TestClient(app)
