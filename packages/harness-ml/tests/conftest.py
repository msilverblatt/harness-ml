import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def binary_dataset():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
        "feature_c": rng.rand(n) * 10,
        "feature_d": rng.randint(0, 5, n).astype(float),
    })
    logits = 0.5 * X["feature_a"] - 0.3 * X["feature_b"] + 0.1 * X["feature_c"]
    y = (logits + rng.randn(n) * 0.5 > 0).astype(int)
    return X, pd.Series(y, name="target")


@pytest.fixture
def multiclass_dataset():
    rng = np.random.RandomState(42)
    n = 300
    X = pd.DataFrame({
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
        "feature_c": rng.rand(n) * 10,
    })
    y = pd.Series(rng.randint(0, 3, n), name="target")
    return X, y


@pytest.fixture
def regression_dataset():
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
        "feature_c": rng.rand(n) * 10,
    })
    y = pd.Series(
        2.0 * X["feature_a"] - 1.5 * X["feature_b"] + 0.5 * X["feature_c"]
        + rng.randn(n) * 0.5,
        name="target",
    )
    return X, y
