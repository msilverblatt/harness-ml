import numpy as np
import pandas as pd
import pytest

from harness.ml.runners.provider_context import ProviderContext


class TestInstanceStorage:
    def test_store_get_instance_roundtrip(self):
        ctx = ProviderContext()
        train_preds = np.array([0.1, 0.2, 0.3])
        test_preds = np.array([0.4, 0.5])
        ctx.store_instance("model_a", train_preds, test_preds)
        result = ctx.get_instance("model_a")
        assert result is not None
        np.testing.assert_array_equal(result[0], train_preds)
        np.testing.assert_array_equal(result[1], test_preds)

    def test_get_missing_instance_returns_none(self):
        ctx = ProviderContext()
        assert ctx.get_instance("nonexistent") is None


class TestEntityStorage:
    def test_store_get_entity_roundtrip(self):
        ctx = ProviderContext()
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        ctx.store_entity("model_b", df)
        result = ctx.get_entity("model_b")
        assert result is not None
        pd.testing.assert_frame_equal(result, df)

    def test_get_missing_entity_returns_none(self):
        ctx = ProviderContext()
        assert ctx.get_entity("nonexistent") is None


class TestInjectFeatures:
    def setup_method(self):
        self.ctx = ProviderContext()
        self.train_preds = np.array([0.1, 0.2, 0.3, 0.4])
        self.test_preds = np.array([0.5, 0.6])
        self.ctx.store_instance("provider_model", self.train_preds, self.test_preds)

    def test_inject_features_train_split(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        result = self.ctx.inject_features(df, split="train", model_deps=["provider_model"])
        assert "pred_provider_model" in result.columns
        np.testing.assert_array_equal(result["pred_provider_model"].values, self.train_preds)

    def test_inject_features_test_split(self):
        df = pd.DataFrame({"x": [10, 20]})
        result = self.ctx.inject_features(df, split="test", model_deps=["provider_model"])
        assert "pred_provider_model" in result.columns
        np.testing.assert_array_equal(result["pred_provider_model"].values, self.test_preds)

    def test_inject_features_does_not_mutate_original(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        original_cols = list(df.columns)
        self.ctx.inject_features(df, split="train", model_deps=["provider_model"])
        assert list(df.columns) == original_cols

    def test_inject_features_skips_size_mismatch(self):
        df = pd.DataFrame({"x": [1, 2]})  # length 2, but train_preds has length 4
        result = self.ctx.inject_features(df, split="train", model_deps=["provider_model"])
        assert "pred_provider_model" not in result.columns


class TestAvailableProviders:
    def test_available_providers_lists_all_stored(self):
        ctx = ProviderContext()
        ctx.store_instance("model_x", np.array([1.0]), np.array([2.0]))
        ctx.store_entity("model_y", pd.DataFrame({"a": [1]}))
        providers = ctx.available_providers()
        assert "model_x" in providers
        assert "model_y" in providers

    def test_available_providers_empty_initially(self):
        ctx = ProviderContext()
        assert ctx.available_providers() == []
