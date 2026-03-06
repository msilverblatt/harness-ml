"""Tests for RandomForest param filtering."""
import logging

import numpy as np
import pytest


def test_rf_filters_boosting_params_with_warning(caplog):
    """Boosting-only params should be silently dropped with a warning."""
    from easyml.core.models.wrappers.random_forest import RandomForestModel

    boosting_params = {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "colsample_bytree": 0.8,
        "reg_alpha": 1.0,
        "num_leaves": 31,
    }
    with caplog.at_level(logging.WARNING):
        model = RandomForestModel(params=boosting_params)

    # Valid RF params kept
    assert "n_estimators" in model.params
    assert "max_depth" in model.params

    # Boosting params removed
    assert "learning_rate" not in model.params
    assert "colsample_bytree" not in model.params
    assert "reg_alpha" not in model.params
    assert "num_leaves" not in model.params

    # Warnings emitted for each dropped param
    warning_text = caplog.text
    assert "learning_rate" in warning_text
    assert "colsample_bytree" in warning_text
    assert "reg_alpha" in warning_text
    assert "num_leaves" in warning_text


def test_rf_no_warning_for_valid_params(caplog):
    """No warnings when only valid RF params are passed."""
    from easyml.core.models.wrappers.random_forest import RandomForestModel

    with caplog.at_level(logging.WARNING):
        model = RandomForestModel(params={"n_estimators": 100, "max_depth": 5})

    assert "n_estimators" in model.params
    assert "max_depth" in model.params
    assert caplog.text == ""


def test_rf_fit_predict_with_filtered_params():
    """Model should work correctly after filtering invalid params."""
    from easyml.core.models.wrappers.random_forest import RandomForestModel

    model = RandomForestModel(params={
        "n_estimators": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
    })
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)


def test_rf_none_params_no_error():
    """Passing None params should still work fine."""
    from easyml.core.models.wrappers.random_forest import RandomForestModel

    model = RandomForestModel(params=None)
    assert model.params == {}
