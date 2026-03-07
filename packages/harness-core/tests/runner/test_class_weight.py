"""Tests for class_weight support in model wrappers and training pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harnessml.core.models.registry import ModelRegistry
from harnessml.core.runner.schema import ModelDef
from harnessml.core.runner.training import train_single_model


@pytest.fixture
def imbalanced_data():
    """Create an imbalanced binary classification dataset."""
    np.random.seed(42)
    n = 500
    X = np.random.randn(n, 5)
    y = np.array([0] * 450 + [1] * 50)
    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3", "f4"])
    df["target"] = y
    df["fold"] = 0
    return df


@pytest.fixture
def feature_names():
    return ["f0", "f1", "f2", "f3", "f4"]


@pytest.fixture
def registry():
    return ModelRegistry.with_defaults()


def test_class_weight_balanced_changes_predictions(imbalanced_data, feature_names, registry):
    """Models with class_weight='balanced' should predict differently than unweighted."""
    model_def_plain = ModelDef(
        type="logistic_regression",
        features=feature_names,
    )
    model_def_balanced = ModelDef(
        type="logistic_regression",
        features=feature_names,
        class_weight="balanced",
    )

    model_plain, _, _ = train_single_model(
        "plain", model_def_plain, imbalanced_data, registry, target_column="target",
    )
    model_balanced, _, _ = train_single_model(
        "balanced", model_def_balanced, imbalanced_data, registry, target_column="target",
    )

    X = imbalanced_data[feature_names].values
    probs_plain = model_plain.predict_proba(X)
    probs_balanced = model_balanced.predict_proba(X)

    # Balanced weighting should increase average predicted probability
    # because the minority class (1) gets upweighted
    assert probs_balanced.mean() > probs_plain.mean()


def test_class_weight_none_is_default(imbalanced_data, feature_names, registry):
    """ModelDef with no class_weight should default to None."""
    model_def = ModelDef(type="logistic_regression", features=feature_names)
    assert model_def.class_weight is None


def test_class_weight_dict(imbalanced_data, feature_names, registry):
    """class_weight as a dict should be accepted and affect training."""
    model_def = ModelDef(
        type="logistic_regression",
        features=feature_names,
        class_weight={0: 1.0, 1: 9.0},
    )
    model, _, _ = train_single_model(
        "dict_weight", model_def, imbalanced_data, registry, target_column="target",
    )
    X = imbalanced_data[feature_names].values
    probs = model.predict_proba(X)
    # With heavy weight on class 1, predictions should shift upward
    assert probs.mean() > 0.1


def test_class_weight_random_forest(imbalanced_data, feature_names, registry):
    """RandomForest should accept class_weight via sample_weight."""
    model_def = ModelDef(
        type="random_forest",
        features=feature_names,
        class_weight="balanced",
        params={"n_estimators": 10, "random_state": 42},
    )
    model, _, _ = train_single_model(
        "rf_balanced", model_def, imbalanced_data, registry, target_column="target",
    )
    probs = model.predict_proba(imbalanced_data[feature_names].values)
    assert len(probs) == len(imbalanced_data)
