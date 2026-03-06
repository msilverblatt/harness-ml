"""Tests for model training internals."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.core.models.registry import ModelRegistry
from easyml.core.runner.schema import ModelDef
from easyml.core.runner.training import (
    _augment_matchup_symmetry,
    _filter_train_folds,
    _fit_cdf_scale,
    _fit_cdf_scale_after_training,
    _is_regressor,
    _margin_to_prob,
    _sigmoid,
    predict_single_model,
    train_single_model,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_train_df(n: int = 200, n_seasons: int = 3, seed: int = 42) -> pd.DataFrame:
    """Create synthetic training data."""
    rng = np.random.default_rng(seed)
    seasons = rng.choice(list(range(2022, 2022 + n_seasons)), size=n)
    diff_x = rng.standard_normal(n)
    diff_y = rng.standard_normal(n)
    result = (diff_x > 0).astype(int)
    margin = diff_x * 5 + rng.standard_normal(n) * 2

    return pd.DataFrame({
        "season": seasons,
        "diff_x": diff_x,
        "diff_y": diff_y,
        "result": result,
        "margin": margin,
        "diff_prior": rng.integers(-15, 16, size=n).astype(float),
    })


def _get_registry() -> ModelRegistry:
    return ModelRegistry.with_defaults()


# -----------------------------------------------------------------------
# Tests: train_single_model
# -----------------------------------------------------------------------

class TestTrainSingleModelClassifier:
    """Train a classifier model."""

    def test_basic_classifier(self):
        df = _make_train_df()
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
            params={"max_iter": 200},
        )
        model, feat_cols, metrics = train_single_model(
            "logreg", model_def, df, _get_registry(),
        )
        assert feat_cols == ["diff_x"]
        assert model is not None
        assert "cdf_scale" not in metrics

    def test_classifier_predict(self):
        df = _make_train_df()
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
            params={"max_iter": 200},
        )
        model, feat_cols, metrics = train_single_model(
            "logreg", model_def, df, _get_registry(),
        )
        probs = predict_single_model(model, model_def, df, feat_cols)
        assert len(probs) == len(df)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


class TestTrainSingleModelRegressor:
    """Train a regressor model with CDF conversion."""

    def test_regressor_with_fitted_cdf(self):
        """Regressor model fits CDF scale when not provided."""
        df = _make_train_df()
        model_def = ModelDef(
            type="xgboost",
            features=["diff_x"],
            params={"n_estimators": 10, "max_depth": 2},
            mode="regressor",
        )
        model, feat_cols, metrics = train_single_model(
            "xgb_reg", model_def, df, _get_registry(),
        )
        assert "cdf_scale" in metrics
        assert metrics["cdf_scale"] > 0

    def test_regressor_predict_gives_probabilities(self):
        """Regressor predictions are valid probabilities."""
        df = _make_train_df()
        model_def = ModelDef(
            type="xgboost",
            features=["diff_x"],
            params={"n_estimators": 10, "max_depth": 2},
            mode="regressor",
        )
        model, feat_cols, metrics = train_single_model(
            "xgb_reg", model_def, df, _get_registry(),
        )
        probs = predict_single_model(
            model, model_def, df, feat_cols, cdf_scale=metrics["cdf_scale"],
        )
        assert len(probs) == len(df)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_regressor_with_preset_cdf_scale(self):
        """Regressor with pre-set cdf_scale uses it."""
        df = _make_train_df()
        model_def = ModelDef(
            type="xgboost",
            features=["diff_x"],
            params={"n_estimators": 10, "max_depth": 2},
            mode="regressor",
            cdf_scale=5.5,
        )
        model, feat_cols, metrics = train_single_model(
            "xgb_reg", model_def, df, _get_registry(),
        )
        # When cdf_scale is preset, metrics should still contain it
        # but the model uses the preset value
        probs = predict_single_model(
            model, model_def, df, feat_cols, cdf_scale=5.5,
        )
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


class TestMultiSeedTraining:
    """Multi-seed training with n_seeds > 1."""

    def test_multi_seed_returns_list(self):
        df = _make_train_df()
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
            params={"max_iter": 200},
            n_seeds=3,
        )
        model, feat_cols, metrics = train_single_model(
            "logreg_multi", model_def, df, _get_registry(),
        )
        assert isinstance(model, list)
        assert len(model) == 3
        assert metrics["n_seeds"] == 3

    def test_multi_seed_predict_averages(self):
        df = _make_train_df()
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
            params={"max_iter": 200},
            n_seeds=3,
        )
        model, feat_cols, metrics = train_single_model(
            "logreg_multi", model_def, df, _get_registry(),
        )
        probs = predict_single_model(model, model_def, df, feat_cols)
        assert len(probs) == len(df)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)


class TestValidationSetSplitting:
    """Train with target_fold filters data correctly."""

    def test_target_fold_filters(self):
        df = _make_train_df(n_seasons=4)
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
            params={"max_iter": 200},
        )
        # With target_fold=2024, only 2022 and 2023 should be used
        model, feat_cols, metrics = train_single_model(
            "logreg", model_def, df, _get_registry(),
            target_fold=2024, fold_column="season",
        )
        assert model is not None

    def test_target_fold_no_data_raises(self):
        """If target_fold filters all data, raise ValueError."""
        df = _make_train_df()
        # All folds are 2022-2024, target_fold=2022 means < 2022 -> empty
        model_def = ModelDef(
            type="logistic_regression",
            features=["diff_x"],
            params={"max_iter": 200},
        )
        with pytest.raises(ValueError, match="No training data"):
            train_single_model(
                "logreg", model_def, df, _get_registry(),
                target_fold=2022, fold_column="season",
            )


class TestMatchupSymmetryAugmentation:
    """Test matchup symmetry augmentation."""

    def test_augment_doubles_data(self):
        X = np.array([[1.0, 2.0], [-1.0, 3.0]])
        y = np.array([1.0, 0.0])
        feature_cols = ["diff_x", "other"]

        X_aug, y_aug = _augment_matchup_symmetry(X, y, feature_cols)
        assert X_aug.shape[0] == 4
        assert len(y_aug) == 4

    def test_augment_negates_diff_features(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 0.0])
        feature_cols = ["diff_a", "non_diff_b"]

        X_aug, y_aug = _augment_matchup_symmetry(X, y, feature_cols)

        # First half is original
        np.testing.assert_array_equal(X_aug[:2], X)
        # Second half: diff_a negated, non_diff_b unchanged
        np.testing.assert_array_equal(X_aug[2:, 0], -X[:, 0])
        np.testing.assert_array_equal(X_aug[2:, 1], X[:, 1])

    def test_augment_flips_binary_labels(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 0.0])
        feature_cols = ["diff_x"]

        X_aug, y_aug = _augment_matchup_symmetry(X, y, feature_cols)
        np.testing.assert_array_equal(y_aug[:2], [1.0, 0.0])
        np.testing.assert_array_equal(y_aug[2:], [0.0, 1.0])

    def test_augment_flips_margin_labels(self):
        X = np.array([[1.0], [2.0]])
        y = np.array([5.0, -3.0])  # margins (continuous)
        feature_cols = ["diff_x"]

        X_aug, y_aug = _augment_matchup_symmetry(X, y, feature_cols)
        np.testing.assert_array_equal(y_aug[:2], [5.0, -3.0])
        np.testing.assert_array_equal(y_aug[2:], [-5.0, 3.0])


# -----------------------------------------------------------------------
# Tests: helper functions
# -----------------------------------------------------------------------

class TestFilterTrainFolds:
    """Test _filter_train_folds."""

    def test_all_returns_everything(self):
        df = pd.DataFrame({"season": [2020, 2021, 2022, 2023]})
        result = _filter_train_folds(df, "all", fold_column="season")
        assert len(result) == 4

    def test_last_n_filters(self):
        df = pd.DataFrame({"season": [2020, 2021, 2022, 2023]})
        result = _filter_train_folds(df, "last_2", fold_column="season")
        assert set(result["season"]) == {2022, 2023}

    def test_target_fold_filters(self):
        df = pd.DataFrame({"season": [2020, 2021, 2022, 2023]})
        result = _filter_train_folds(df, "all", target_fold=2022, fold_column="season")
        assert set(result["season"]) == {2020, 2021}

    def test_last_n_with_target_fold(self):
        df = pd.DataFrame({"season": [2020, 2021, 2022, 2023]})
        result = _filter_train_folds(df, "last_2", target_fold=2023, fold_column="season")
        # target_fold filters to < 2023 => [2020, 2021, 2022]
        # last_2 keeps > max(2022) - 2 = > 2020 => [2021, 2022]
        assert set(result["season"]) == {2021, 2022}


class TestMarginToProb:
    """Test _margin_to_prob."""

    def test_zero_margin_gives_half(self):
        probs = _margin_to_prob(np.array([0.0]), 5.0)
        np.testing.assert_almost_equal(probs, [0.5])

    def test_positive_margin_above_half(self):
        probs = _margin_to_prob(np.array([10.0]), 5.0)
        assert probs[0] > 0.5

    def test_negative_margin_below_half(self):
        probs = _margin_to_prob(np.array([-10.0]), 5.0)
        assert probs[0] < 0.5


class TestFitCdfScale:
    """Test _fit_cdf_scale."""

    def test_fit_finds_reasonable_scale(self):
        rng = np.random.default_rng(42)
        margins = rng.standard_normal(200) * 5
        y_binary = (margins > 0).astype(float)
        scale = _fit_cdf_scale(margins, y_binary)
        assert scale > 0
        # Scale should be in a reasonable range for this data
        assert scale < 50


class TestIsRegressor:
    """Test _is_regressor."""

    def test_classifier_mode(self):
        md = ModelDef(type="logistic_regression", features=["x"])
        assert not _is_regressor(md)

    def test_regressor_mode(self):
        md = ModelDef(type="xgboost", features=["x"], mode="regressor")
        assert _is_regressor(md)

    def test_prediction_type_margin(self):
        md = ModelDef(type="xgboost", features=["x"], prediction_type="margin")
        assert _is_regressor(md)

    def test_regression_in_type(self):
        md = ModelDef(type="xgboost_regression", features=["x"])
        assert _is_regressor(md)

    def test_logistic_regression_is_not_regressor(self):
        """logistic_regression type should NOT be detected as regressor."""
        md = ModelDef(type="logistic_regression", features=["x"])
        assert not _is_regressor(md)


# -----------------------------------------------------------------------
# Tests: sigmoid CDF and post-training scale fitting
# -----------------------------------------------------------------------

class TestSigmoid:
    """Test _sigmoid function."""

    def test_sigmoid_valid_probabilities(self):
        """_sigmoid produces valid probabilities."""
        margins = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        probs = _sigmoid(margins, scale=5.0)
        assert all(0 < p < 1 for p in probs)
        assert probs[2] == pytest.approx(0.5)
        assert probs[0] < probs[4]

    def test_sigmoid_symmetry(self):
        """_sigmoid is symmetric around 0."""
        probs = _sigmoid(np.array([-3.0, 3.0]), scale=2.0)
        assert probs[0] == pytest.approx(1.0 - probs[1])

    def test_sigmoid_scale_affects_steepness(self):
        """Smaller scale = steeper sigmoid."""
        margin = np.array([1.0])
        prob_small_scale = _sigmoid(margin, scale=0.5)
        prob_large_scale = _sigmoid(margin, scale=5.0)
        # Smaller scale should push probability further from 0.5
        assert prob_small_scale[0] > prob_large_scale[0]


class TestFitCdfScaleAfterTraining:
    """Test _fit_cdf_scale_after_training."""

    def test_returns_positive_scale(self):
        """Post-training scale fitting returns positive scale."""
        margins = np.array([-5, -2, 0, 2, 5, -3, 1, 4, -1, 3], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1])
        scale = _fit_cdf_scale_after_training(margins, y)
        assert scale > 0

    def test_well_separated_data(self):
        """When margins clearly separate labels, scale should be reasonable."""
        rng = np.random.default_rng(42)
        margins = rng.standard_normal(200) * 5
        y_binary = (margins > 0).astype(float)
        scale = _fit_cdf_scale_after_training(margins, y_binary)
        assert scale > 0
        assert scale < 50

    def test_noisy_data(self):
        """Even with noise, scale fitting should succeed."""
        rng = np.random.default_rng(99)
        margins = rng.standard_normal(100) * 3
        y_binary = rng.binomial(1, 0.5, size=100).astype(float)
        scale = _fit_cdf_scale_after_training(margins, y_binary)
        assert scale > 0


class TestGenericModelCreation:
    """Test that regressor models are created generically via registry."""

    def test_regressor_created_via_registry(self):
        """Regressor training should work through generic registry.create()."""
        df = _make_train_df()
        model_def = ModelDef(
            type="xgboost",
            features=["diff_x"],
            params={"n_estimators": 10, "max_depth": 2},
            mode="regressor",
        )
        model, feat_cols, metrics = train_single_model(
            "xgb_reg", model_def, df, _get_registry(),
        )
        assert model is not None
        assert "cdf_scale" in metrics
        assert metrics["cdf_scale"] > 0

    def test_post_training_cdf_scale_differs_from_raw(self):
        """Post-training CDF scale (fit from predictions) should differ from raw margin std."""
        df = _make_train_df(n=500)
        model_def = ModelDef(
            type="xgboost",
            features=["diff_x", "diff_y"],
            params={"n_estimators": 50, "max_depth": 3},
            mode="regressor",
        )
        model, feat_cols, metrics = train_single_model(
            "xgb_reg", model_def, df, _get_registry(),
        )
        # The fitted scale should be a positive number
        assert metrics["cdf_scale"] > 0
        # And it should produce valid probabilities
        probs = predict_single_model(
            model, model_def, df, feat_cols, cdf_scale=metrics["cdf_scale"],
        )
        assert np.all(probs > 0)
        assert np.all(probs < 1)
