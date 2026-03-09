"""Tests for training_filter non-destructive row exclusion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.core.models.registry import ModelRegistry
from harnessml.core.runner.schema import ModelDef
from harnessml.core.runner.training.trainer import train_single_model


@pytest.fixture()
def registry():
    return ModelRegistry.with_defaults()


@pytest.fixture()
def sample_df():
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "f1": rng.randn(200),
        "f2": rng.randn(200),
        "target": rng.randint(0, 2, 200),
        "fold": np.repeat([0, 1], 100),
        "is_expired": [1] * 20 + [0] * 180,
    })


def test_training_filter_excludes_rows(registry, sample_df):
    """training_filter should exclude matching rows from training data."""
    model_def = ModelDef(
        type="logistic_regression",
        features=["f1", "f2"],
        training_filter={"exclude": "is_expired == 1"},
    )
    model, features, meta = train_single_model(
        "test", model_def, sample_df, registry, target_column="target"
    )
    # Should have trained on 180 rows, not 200
    assert meta["n_train_rows"] == 180


def test_training_filter_none_keeps_all_rows(registry, sample_df):
    """No training_filter should keep all rows."""
    model_def = ModelDef(
        type="logistic_regression",
        features=["f1", "f2"],
    )
    model, features, meta = train_single_model(
        "test", model_def, sample_df, registry, target_column="target"
    )
    assert meta["n_train_rows"] == 200


def test_training_filter_does_not_mutate_input(registry, sample_df):
    """training_filter must not modify the original DataFrame."""
    original_len = len(sample_df)
    model_def = ModelDef(
        type="logistic_regression",
        features=["f1", "f2"],
        training_filter={"exclude": "is_expired == 1"},
    )
    train_single_model(
        "test", model_def, sample_df, registry, target_column="target"
    )
    assert len(sample_df) == original_len


def test_training_filter_complex_expression(registry):
    """training_filter should handle complex pandas query expressions."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "f1": rng.randn(100),
        "f2": rng.randn(100),
        "target": rng.randint(0, 2, 100),
        "status": ["active"] * 60 + ["expired"] * 30 + ["pending"] * 10,
        "score": rng.uniform(0, 1, 100),
    })
    model_def = ModelDef(
        type="logistic_regression",
        features=["f1", "f2"],
        training_filter={"exclude": "status == 'expired' or score < 0.1"},
    )
    model, features, meta = train_single_model(
        "test", model_def, df, registry, target_column="target"
    )
    # Verify some rows were excluded
    assert meta["n_train_rows"] < 100
