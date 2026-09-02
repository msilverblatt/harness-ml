"""Parametrized contract tests -- every available model must pass."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from harness.ml.models.registry import ModelRegistry


def available_models():
    return ModelRegistry.list_available()


@pytest.fixture(params=available_models())
def model_wrapper(request):
    return ModelRegistry.get(request.param)


class TestModelContract:
    def test_has_required_attributes(self, model_wrapper):
        assert isinstance(model_wrapper.name, str)
        assert isinstance(model_wrapper.supports_tasks, list)
        assert isinstance(model_wrapper.requires_packages, list)

    def test_default_params(self, model_wrapper):
        for task in model_wrapper.supports_tasks:
            params = model_wrapper.default_params(task)
            assert isinstance(params, dict)

    def test_param_schema(self, model_wrapper):
        schema = model_wrapper.param_schema()
        assert isinstance(schema, dict)

    def test_fit_predict_binary(self, model_wrapper, binary_dataset):
        if "binary" not in model_wrapper.supports_tasks:
            pytest.skip("No binary support")
        X, y = binary_dataset
        params = model_wrapper.default_params("binary")
        result = model_wrapper.fit(X, y, None, None, params)
        assert result.model is not None
        preds = model_wrapper.predict(result.model, X)
        assert len(preds) == len(X)
        assert np.all(np.isfinite(preds))

    def test_fit_predict_regression(self, model_wrapper, regression_dataset):
        if "regression" not in model_wrapper.supports_tasks:
            pytest.skip("No regression support")
        X, y = regression_dataset
        params = model_wrapper.default_params("regression")
        result = model_wrapper.fit(X, y, None, None, params)
        preds = model_wrapper.predict(result.model, X)
        assert len(preds) == len(X)

    def test_save_load_roundtrip(self, model_wrapper, binary_dataset, tmp_path):
        if "binary" not in model_wrapper.supports_tasks:
            pytest.skip("No binary support")
        X, y = binary_dataset
        params = model_wrapper.default_params("binary")
        result = model_wrapper.fit(X, y, None, None, params)

        path = tmp_path / f"{model_wrapper.name}.model"
        model_wrapper.save(result.model, path)
        loaded = model_wrapper.load(path)

        preds1 = model_wrapper.predict(result.model, X)
        preds2 = model_wrapper.predict(loaded, X)
        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_supports_multi_seed(self, model_wrapper):
        result = model_wrapper.supports_multi_seed()
        assert isinstance(result, bool)
