"""Tests for TabNet wrapper: normalize, val_fraction, scheduler, eval_set, learning_rate."""
import json
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

pytab = pytest.importorskip("pytorch_tabnet")


@pytest.fixture
def tabnet_data():
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _make_model(**kwargs):
    from harnessml.core.models.wrappers.tabnet import TabNetModel

    defaults = {"n_d": 8, "n_a": 8, "n_steps": 3, "max_epochs": 3, "patience": 3, "verbose": 0}
    params = {k: v for k, v in defaults.items() if k not in kwargs.get("params", {})}
    params.update(kwargs.pop("params", {}))
    return TabNetModel(params=params, **kwargs)


class TestNormalize:
    def test_normalize_standardizes_data(self, tabnet_data):
        """When normalize=True, fit should standardize X and store means/stds."""
        X, y = tabnet_data
        model = _make_model(normalize=True)

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            mock_instance.predict_proba.return_value = np.column_stack([
                np.zeros(len(X)), np.ones(len(X)) * 0.5
            ])
            MockCls.return_value = mock_instance

            model.fit(X, y)

            # Check that the data passed to fit was standardized
            call_args = mock_instance.fit.call_args
            X_passed = call_args[0][0]
            # Standardized data should have near-zero mean and near-unit std
            assert abs(X_passed.mean()) < 0.2
            assert abs(X_passed.std() - 1.0) < 0.3

        # means and stds should be stored
        assert model._feature_means is not None
        assert model._feature_stds is not None
        assert model._feature_means.shape == (5,)
        assert model._feature_stds.shape == (5,)

    def test_normalize_applied_at_predict(self, tabnet_data):
        """predict_proba should apply the same normalization."""
        X, y = tabnet_data
        model = _make_model(normalize=True)

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            mock_instance.predict_proba.return_value = np.column_stack([
                np.zeros(len(X)), np.ones(len(X)) * 0.5
            ])
            MockCls.return_value = mock_instance

            model.fit(X, y)
            model.predict_proba(X)

            # predict_proba call should receive standardized X
            pred_call_args = mock_instance.predict_proba.call_args
            X_pred = pred_call_args[0][0]
            assert abs(X_pred.mean()) < 0.2

    def test_normalize_in_meta(self, tabnet_data, tmp_path):
        """means/stds should be persisted in meta.json."""
        X, y = tabnet_data
        model = _make_model(normalize=True)

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            mock_instance.predict_proba.return_value = np.column_stack([
                np.zeros(len(X)), np.ones(len(X)) * 0.5
            ])
            MockCls.return_value = mock_instance
            model.fit(X, y)

            model.save(tmp_path / "model")

        meta = json.loads((tmp_path / "model" / "meta.json").read_text())
        assert "feature_means" in meta
        assert "feature_stds" in meta
        assert len(meta["feature_means"]) == 5

    def test_no_normalize_by_default(self, tabnet_data):
        """Default normalize=False should NOT standardize."""
        model = _make_model()
        assert model._normalize is False
        assert model._feature_means is None
        assert model._feature_stds is None


class TestValFraction:
    def test_val_fraction_splits_data(self, tabnet_data):
        """val_fraction=0.2 should auto-split and pass eval_set to fit."""
        X, y = tabnet_data
        model = _make_model(val_fraction=0.2)

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance

            model.fit(X, y)

            call_kwargs = mock_instance.fit.call_args[1]
            assert "eval_set" in call_kwargs
            eval_set = call_kwargs["eval_set"]
            assert len(eval_set) == 1
            X_val, y_val = eval_set[0]
            # ~20% of 100 = 20 rows
            assert X_val.shape[0] == 20
            # Training data should be ~80 rows
            X_train = mock_instance.fit.call_args[0][0]
            assert X_train.shape[0] == 80


class TestLearningRate:
    def test_learning_rate_injected(self, tabnet_data):
        """learning_rate should be injected into optimizer_params."""
        X, y = tabnet_data
        model = _make_model(learning_rate=0.005)

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance

            model.fit(X, y)

            init_kwargs = MockCls.call_args[1]
            assert "optimizer_params" in init_kwargs
            assert init_kwargs["optimizer_params"]["lr"] == 0.005

    def test_learning_rate_merges_with_existing_optimizer_params(self, tabnet_data):
        """learning_rate should merge with existing optimizer_params."""
        X, y = tabnet_data
        model = _make_model(
            params={"optimizer_params": {"weight_decay": 1e-5}},
            learning_rate=0.01,
        )

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance

            model.fit(X, y)

            init_kwargs = MockCls.call_args[1]
            assert init_kwargs["optimizer_params"]["lr"] == 0.01
            assert init_kwargs["optimizer_params"]["weight_decay"] == 1e-5


class TestScheduler:
    def test_scheduler_params_built(self, tabnet_data):
        """scheduler_step_size + scheduler_gamma should build scheduler_params."""
        X, y = tabnet_data
        model = _make_model(scheduler_step_size=10, scheduler_gamma=0.9)

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance

            model.fit(X, y)

            call_kwargs = mock_instance.fit.call_args[1]
            assert "scheduler_params" in call_kwargs
            assert call_kwargs["scheduler_params"]["step_size"] == 10
            assert call_kwargs["scheduler_params"]["gamma"] == 0.9

    def test_scheduler_not_set_by_default(self, tabnet_data):
        """Without scheduler params, scheduler_params should not be passed."""
        X, y = tabnet_data
        model = _make_model()

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance

            model.fit(X, y)

            call_kwargs = mock_instance.fit.call_args[1]
            assert "scheduler_params" not in call_kwargs


class TestEvalSet:
    def test_eval_set_passed_through(self, tabnet_data):
        """Explicit eval_set kwarg should be forwarded to TabNet.fit."""
        X, y = tabnet_data
        X_val = X[:20]
        y_val = y[:20]
        model = _make_model()

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance

            model.fit(X, y, eval_set=[(X_val, y_val)])

            call_kwargs = mock_instance.fit.call_args[1]
            assert "eval_set" in call_kwargs
            np.testing.assert_array_equal(call_kwargs["eval_set"][0][0], X_val)

    def test_eval_set_takes_precedence_over_val_fraction(self, tabnet_data):
        """Explicit eval_set should override val_fraction."""
        X, y = tabnet_data
        X_val = X[:10]
        y_val = y[:10]
        model = _make_model(val_fraction=0.2)

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance

            model.fit(X, y, eval_set=[(X_val, y_val)])

            call_kwargs = mock_instance.fit.call_args[1]
            # Should use the explicit eval_set (10 rows), not auto-split (20 rows)
            assert call_kwargs["eval_set"][0][0].shape[0] == 10
            # Training data should be full X (100), not 80
            X_train = mock_instance.fit.call_args[0][0]
            assert X_train.shape[0] == 100


class TestParamRenames:
    def test_relaxation_factor_renamed_to_gamma(self, tabnet_data):
        """relaxation_factor should be renamed to gamma for TabNet."""
        X, y = tabnet_data
        model = _make_model(params={"relaxation_factor": 1.5})

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance

            model.fit(X, y)

            init_kwargs = MockCls.call_args[1]
            assert "gamma" in init_kwargs
            assert "relaxation_factor" not in init_kwargs
            assert init_kwargs["gamma"] == 1.5


class TestSeedStride:
    def test_seed_stride(self, tabnet_data):
        """seed_stride should offset each seed's value."""
        X, y = tabnet_data
        model = _make_model(n_seeds=2, seed_stride=100)

        with patch("pytorch_tabnet.tab_model.TabNetClassifier") as MockCls:
            mock_instance = MagicMock()
            MockCls.return_value = mock_instance

            model.fit(X, y)

            # Two calls, seeds should be 0 and 100
            assert MockCls.call_count == 2
            seed0 = MockCls.call_args_list[0][1]["seed"]
            seed1 = MockCls.call_args_list[1][1]["seed"]
            assert seed0 == 0
            assert seed1 == 100
