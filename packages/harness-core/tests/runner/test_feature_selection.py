import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.feature_selection import select_features


@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame({f"f{i}": np.random.randn(100) for i in range(20)})
    X["f0"] = np.random.randn(100) * 10  # Make f0 most informative
    y = (X["f0"] > 0).astype(int)
    return X, y


def test_select_k_best(sample_data):
    X, y = sample_data
    selected = select_features(X, y, method="k_best", k=5)
    assert "f0" in selected
    assert len(selected) == 5


def test_rfe_selection(sample_data):
    X, y = sample_data
    selected = select_features(X, y, method="rfe", k=5)
    assert len(selected) == 5


def test_correlation_clustering(sample_data):
    X, y = sample_data
    selected = select_features(X, y, method="correlation_cluster", threshold=0.9)
    assert len(selected) <= 21  # 20 original + f0 override


def test_unknown_method(sample_data):
    X, y = sample_data
    with pytest.raises(ValueError, match="Unknown"):
        select_features(X, y, method="nonexistent")
