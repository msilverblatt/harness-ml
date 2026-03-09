import numpy as np
import pytest


@pytest.fixture
def synthetic_binary_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y
