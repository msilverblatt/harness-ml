import numpy as np
import pandas as pd
import pytest

from harness.ml.config.models import SingleModelConfig
from harness.ml.runners.training import train_single_model


def make_binary_data(n=100, n_features=4, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(
        rng.randn(n, n_features),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series((rng.randn(n) > 0).astype(int), name="target")
    return X, y


def logistic_config(**kwargs) -> SingleModelConfig:
    defaults = dict(name="lr", model_type="logistic")
    defaults.update(kwargs)
    return SingleModelConfig(**defaults)


class TestTrainSingleModelBasic:
    def test_predictions_correct_length(self):
        X, y = make_binary_data(n=80)
        X_train, y_train = X.iloc[:60], y.iloc[:60]
        X_test = X.iloc[60:]

        config = logistic_config()
        result = train_single_model(config, X_train, y_train, X_test, task_type="binary")

        assert result.error is None
        assert len(result.train_predictions) == len(X_train)
        assert len(result.test_predictions) == len(X_test)

    def test_predictions_finite_and_in_range(self):
        X, y = make_binary_data(n=80)
        X_train, y_train = X.iloc[:60], y.iloc[:60]
        X_test = X.iloc[60:]

        config = logistic_config()
        result = train_single_model(config, X_train, y_train, X_test, task_type="binary")

        assert result.error is None
        assert np.all(np.isfinite(result.train_predictions))
        assert np.all(np.isfinite(result.test_predictions))
        assert np.all(result.train_predictions >= 0)
        assert np.all(result.train_predictions <= 1)
        assert np.all(result.test_predictions >= 0)
        assert np.all(result.test_predictions <= 1)

    def test_fit_result_attached(self):
        X, y = make_binary_data()
        config = logistic_config()
        result = train_single_model(config, X, y, X.iloc[:10], task_type="binary")

        assert result.error is None
        assert result.fit_result is not None
        assert result.fit_result.model is not None

    def test_duration_recorded(self):
        X, y = make_binary_data()
        config = logistic_config()
        result = train_single_model(config, X, y, X.iloc[:10], task_type="binary")

        assert result.duration_s >= 0.0


class TestMultiSeed:
    def test_n_seeds_produces_correct_length(self):
        X, y = make_binary_data(n=80)
        X_train, y_train = X.iloc[:60], y.iloc[:60]
        X_test = X.iloc[60:]

        config = logistic_config(n_seeds=3)
        result = train_single_model(config, X_train, y_train, X_test, task_type="binary")

        assert result.error is None
        assert len(result.train_predictions) == len(X_train)
        assert len(result.test_predictions) == len(X_test)

    def test_n_seeds_predictions_finite(self):
        X, y = make_binary_data(n=80)
        X_train, y_train = X.iloc[:60], y.iloc[:60]
        X_test = X.iloc[60:]

        config = logistic_config(n_seeds=3)
        result = train_single_model(config, X_train, y_train, X_test, task_type="binary")

        assert np.all(np.isfinite(result.train_predictions))
        assert np.all(np.isfinite(result.test_predictions))


class TestTrainingFilter:
    def test_training_filter_reduces_training_rows(self):
        X, y = make_binary_data(n=100)
        X_test = X.iloc[:10]

        # Filter keeps only rows where f0 > 0 (roughly half)
        config = logistic_config(training_filter="f0 > 0")
        result = train_single_model(config, X, y, X_test, task_type="binary")

        assert result.error is None
        # Train predictions should have length equal to filtered X, but we return
        # predictions on the original filtered training set, so check it's <= full size
        # (the spec says predict on X_tr after filter)
        assert len(result.train_predictions) < len(X)
        assert len(result.test_predictions) == len(X_test)

    def test_training_filter_predictions_still_finite(self):
        X, y = make_binary_data(n=100)
        X_test = X.iloc[:10]

        config = logistic_config(training_filter="f0 > 0")
        result = train_single_model(config, X, y, X_test, task_type="binary")

        assert result.error is None
        assert np.all(np.isfinite(result.train_predictions))
        assert np.all(np.isfinite(result.test_predictions))


class TestAugmentSymmetry:
    def test_augment_symmetry_doubles_training(self):
        X, y = make_binary_data(n=60)
        X_train, y_train = X.iloc[:50], y.iloc[:50]
        X_test = X.iloc[50:]

        config_no_aug = logistic_config(augment_symmetry=False)
        config_aug = logistic_config(augment_symmetry=True)

        result_no_aug = train_single_model(
            config_no_aug, X_train, y_train, X_test, task_type="binary"
        )
        result_aug = train_single_model(
            config_aug, X_train, y_train, X_test, task_type="binary"
        )

        # Augmented training set is doubled, so train_predictions length doubles
        assert len(result_aug.train_predictions) == 2 * len(result_no_aug.train_predictions)
        # Test predictions length unchanged
        assert len(result_aug.test_predictions) == len(result_no_aug.test_predictions)

    def test_augment_symmetry_test_predictions_valid(self):
        X, y = make_binary_data(n=60)
        X_train, y_train = X.iloc[:50], y.iloc[:50]
        X_test = X.iloc[50:]

        config = logistic_config(augment_symmetry=True)
        result = train_single_model(config, X_train, y_train, X_test, task_type="binary")

        assert result.error is None
        assert np.all(np.isfinite(result.test_predictions))


class TestUnknownModelType:
    def test_unknown_model_type_returns_error(self):
        X, y = make_binary_data()
        config = logistic_config(model_type="does_not_exist_xyz")
        result = train_single_model(config, X, y, X.iloc[:10], task_type="binary")

        assert result.error is not None
        assert "does_not_exist_xyz" in result.error
        assert len(result.train_predictions) == 0
        assert len(result.test_predictions) == 0
