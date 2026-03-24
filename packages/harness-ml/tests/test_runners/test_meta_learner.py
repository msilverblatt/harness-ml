import numpy as np
import pandas as pd
import pytest

from harness.ml.config.ensemble import EnsembleConfig
from harness.ml.runners.meta_learner import MetaLearner, MetaLearnerResult


def make_fold_predictions(n_folds=3, n_samples=50, n_models=2, seed=42):
    """Build a dict of fold DataFrames with prob_* columns and a target column."""
    rng = np.random.RandomState(seed)
    folds = {}
    for fold_id in range(n_folds):
        data = {f"prob_model{i}": rng.uniform(0, 1, n_samples) for i in range(n_models)}
        data["target"] = rng.randint(0, 2, n_samples)
        folds[str(fold_id)] = pd.DataFrame(data)
    return folds


class TestSimpleAverage:
    def test_simple_average_produces_mean_of_model_predictions(self):
        fold_preds = make_fold_predictions(n_folds=3, n_models=2, seed=0)
        config = EnsembleConfig(method="average")
        result = MetaLearner().train(fold_preds, config)
        assert result.method == "average"
        for fold_id, df in fold_preds.items():
            expected = df[["prob_model0", "prob_model1"]].mean(axis=1).values
            np.testing.assert_allclose(result.fold_predictions[fold_id], expected)

    def test_simple_average_result_keys_match_input_folds(self):
        fold_preds = make_fold_predictions(n_folds=4, n_models=3)
        config = EnsembleConfig(method="average")
        result = MetaLearner().train(fold_preds, config)
        assert set(result.fold_predictions.keys()) == set(fold_preds.keys())


class TestStackedMeta:
    def test_stacked_produces_predictions_per_fold(self):
        fold_preds = make_fold_predictions(n_folds=3, n_models=2)
        config = EnsembleConfig(method="stacked", meta_learner_type="logistic")
        result = MetaLearner().train(fold_preds, config)
        assert result.method == "stacked"
        assert set(result.fold_predictions.keys()) == {"0", "1", "2"}
        for preds in result.fold_predictions.values():
            assert len(preds) == 50

    def test_meta_coefficients_has_entries_when_stacked(self):
        fold_preds = make_fold_predictions(n_folds=3, n_models=2)
        config = EnsembleConfig(method="stacked", meta_learner_type="logistic")
        result = MetaLearner().train(fold_preds, config)
        assert len(result.meta_coefficients) == 2
        assert "model0" in result.meta_coefficients
        assert "model1" in result.meta_coefficients

    def test_production_meta_model_is_not_none(self):
        fold_preds = make_fold_predictions(n_folds=3, n_models=2)
        config = EnsembleConfig(method="stacked", meta_learner_type="logistic")
        result = MetaLearner().train(fold_preds, config)
        assert result.meta_model is not None

    def test_predictions_are_probabilities(self):
        fold_preds = make_fold_predictions(n_folds=3, n_models=2)
        config = EnsembleConfig(method="stacked", meta_learner_type="logistic")
        result = MetaLearner().train(fold_preds, config)
        for preds in result.fold_predictions.values():
            assert np.all(preds >= 0.0)
            assert np.all(preds <= 1.0)


class TestExcludeModels:
    def test_exclude_models_removes_from_meta_feature_matrix(self):
        fold_preds = make_fold_predictions(n_folds=3, n_models=3)
        config = EnsembleConfig(
            method="stacked",
            meta_learner_type="logistic",
            exclude_models=["model2"],
        )
        result = MetaLearner().train(fold_preds, config)
        # Only 2 models active, so coefficients for model0 and model1 only
        assert "model2" not in result.meta_coefficients
        assert "model0" in result.meta_coefficients
        assert "model1" in result.meta_coefficients

    def test_all_excluded_falls_back_to_average(self):
        fold_preds = make_fold_predictions(n_folds=3, n_models=2)
        config = EnsembleConfig(
            method="stacked",
            exclude_models=["model0", "model1"],
        )
        result = MetaLearner().train(fold_preds, config)
        # Fallback returns average method
        assert result.method == "average"


class TestFallback:
    def test_fallback_to_average_when_single_fold(self):
        """Single fold: no training data available, should fallback to mean."""
        rng = np.random.RandomState(7)
        fold_preds = {
            "0": pd.DataFrame(
                {
                    "prob_model0": rng.uniform(0, 1, 20),
                    "prob_model1": rng.uniform(0, 1, 20),
                    "target": rng.randint(0, 2, 20),
                }
            )
        }
        config = EnsembleConfig(method="stacked", meta_learner_type="logistic")
        result = MetaLearner().train(fold_preds, config)
        # Single fold: train_dfs is empty, so should fall back to simple mean
        expected = fold_preds["0"][["prob_model0", "prob_model1"]].mean(axis=1).values
        np.testing.assert_allclose(result.fold_predictions["0"], expected)


class TestRidgeMeta:
    def test_ridge_meta_learner_type(self):
        fold_preds = make_fold_predictions(n_folds=3, n_models=2)
        config = EnsembleConfig(method="stacked", meta_learner_type="ridge")
        result = MetaLearner().train(fold_preds, config)
        assert result.method == "stacked"
        assert set(result.fold_predictions.keys()) == {"0", "1", "2"}
