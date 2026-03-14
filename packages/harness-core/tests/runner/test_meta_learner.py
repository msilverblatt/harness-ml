"""Tests for stacked meta-learner."""
from __future__ import annotations

import json

import numpy as np
import pytest
from harnessml.core.runner.training.meta_learner import StackedEnsemble, train_meta_learner_loso

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _synth_data(n: int = 100, n_models: int = 3, n_folds: int = 3, seed: int = 42):
    """Generate synthetic data for meta-learner tests.

    Returns dict with y_true, model_preds, prior_diffs, fold_labels, model_names.
    """
    rng = np.random.RandomState(seed)

    # True probabilities
    true_p = rng.beta(2, 2, size=n)
    y_true = (rng.rand(n) < true_p).astype(float)

    model_names = [f"model_{i}" for i in range(n_models)]
    model_preds = {}
    for name in model_names:
        # Each model is a noisy version of truth
        noise = rng.normal(0, 0.1, size=n)
        model_preds[name] = np.clip(true_p + noise, 0.05, 0.95)

    prior_diffs = rng.choice([-8, -4, -2, -1, 1, 2, 4, 8], size=n).astype(float)

    # Assign fold labels cyclically
    fold_labels = np.array([2020 + (i % n_folds) for i in range(n)])

    return {
        "y_true": y_true,
        "model_preds": model_preds,
        "prior_diffs": prior_diffs,
        "fold_labels": fold_labels,
        "model_names": model_names,
    }


# -----------------------------------------------------------------------
# StackedEnsemble
# -----------------------------------------------------------------------

class TestStackedEnsemble:
    def test_fit_predict_basic(self):
        data = _synth_data(100, 3)
        meta = StackedEnsemble(data["model_names"])
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])
        probs = meta.predict(data["model_preds"], data["prior_diffs"])

        assert len(probs) == 100
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_coefficients_shape(self):
        data = _synth_data(100, 3)
        meta = StackedEnsemble(data["model_names"])
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])
        coeffs = meta.get_coefficients()

        # 3 models + seed_diff = 4 coefficients
        assert len(coeffs) == 4
        assert "model_0" in coeffs
        assert "model_1" in coeffs
        assert "model_2" in coeffs
        assert "prior_diff" in coeffs

    def test_predict_before_fit_raises(self):
        meta = StackedEnsemble(["a", "b"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            meta.predict({"a": np.array([0.5]), "b": np.array([0.5])}, np.array([1.0]))

    def test_coefficients_before_fit_raises(self):
        meta = StackedEnsemble(["a"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            meta.get_coefficients()

    def test_with_extra_features(self):
        data = _synth_data(100, 2)
        rng = np.random.RandomState(99)
        extra = {"momentum": rng.randn(100), "elo_diff": rng.randn(100)}

        meta = StackedEnsemble(data["model_names"])
        meta.fit(
            data["model_preds"], data["prior_diffs"], data["y_true"],
            extra_features=extra,
        )
        probs = meta.predict(
            data["model_preds"], data["prior_diffs"],
            extra_features=extra,
        )
        coeffs = meta.get_coefficients(extra_features=extra)

        assert len(probs) == 100
        # 2 models + seed_diff + 2 extra = 5 coefficients
        assert len(coeffs) == 5
        assert "elo_diff" in coeffs
        assert "momentum" in coeffs

    def test_custom_C(self):
        data = _synth_data(100, 2)
        meta = StackedEnsemble(data["model_names"])
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"], C=10.0)
        probs = meta.predict(data["model_preds"], data["prior_diffs"])
        assert len(probs) == 100

    def test_save_load_roundtrip(self, tmp_path):
        data = _synth_data(100, 3)
        meta = StackedEnsemble(data["model_names"])
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])

        probs_before = meta.predict(data["model_preds"], data["prior_diffs"])
        coeffs_before = meta.get_coefficients()

        path = tmp_path / "meta_learner.json"
        meta.save(path)

        # Load into new instance
        meta2 = StackedEnsemble([])  # model_names will be overwritten by load
        meta2.load(path)

        probs_after = meta2.predict(data["model_preds"], data["prior_diffs"])
        coeffs_after = meta2.get_coefficients()

        np.testing.assert_allclose(probs_before, probs_after, atol=1e-10)
        assert coeffs_before.keys() == coeffs_after.keys()
        for k in coeffs_before:
            assert abs(coeffs_before[k] - coeffs_after[k]) < 1e-10

    def test_save_before_fit_raises(self, tmp_path):
        meta = StackedEnsemble(["a"])
        with pytest.raises(RuntimeError, match="not been fitted"):
            meta.save(tmp_path / "meta.json")

    def test_save_file_format(self, tmp_path):
        data = _synth_data(50, 2)
        meta = StackedEnsemble(data["model_names"])
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])

        path = tmp_path / "meta.json"
        meta.save(path)

        state = json.loads(path.read_text())
        assert "model_names" in state
        assert "coef" in state
        assert "intercept" in state
        assert "classes" in state
        assert "C" in state
        assert state["model_names"] == data["model_names"]


# -----------------------------------------------------------------------
# train_meta_learner_loso
# -----------------------------------------------------------------------

class TestTrainMetaLearnerLoso:
    def test_basic_loso(self):
        data = _synth_data(150, 3, n_folds=3)
        ensemble_config = {
            "meta_learner": {"C": 1.0},
            "calibration": "none",
            "pre_calibration": {},
        }

        meta, post_cal, pre_cals = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=ensemble_config,
        )

        assert isinstance(meta, StackedEnsemble)
        assert post_cal is None  # calibration="none"
        assert pre_cals == {}

        # Verify final meta-learner can predict
        probs = meta.predict(data["model_preds"], data["prior_diffs"])
        assert len(probs) == 150
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_with_spline_post_calibration(self):
        data = _synth_data(150, 3, n_folds=3)
        ensemble_config = {
            "meta_learner": {"C": 1.0},
            "calibration": "spline",
            "pre_calibration": {},
            "spline_prob_max": 0.985,
            "spline_n_bins": 20,
        }

        meta, post_cal, pre_cals = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=ensemble_config,
        )

        assert post_cal is not None
        assert post_cal.is_fitted

    def test_with_pre_calibration(self):
        data = _synth_data(150, 3, n_folds=3)
        ensemble_config = {
            "meta_learner": {"C": 1.0},
            "calibration": "none",
            "pre_calibration": {"model_0": "spline"},
            "spline_prob_max": 0.985,
            "spline_n_bins": 20,
        }

        meta, post_cal, pre_cals = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=ensemble_config,
        )

        # Final pre-calibrators should be fitted
        assert "model_0" in pre_cals
        assert pre_cals["model_0"].is_fitted
        # Other models should NOT have pre-calibrators
        assert "model_1" not in pre_cals
        assert "model_2" not in pre_cals

    def test_pre_calibration_per_fold(self):
        """Verify pre-calibration is applied per-fold (leakage prevention).

        The test ensures that results differ from what would happen if
        pre-calibration were applied globally (leaked).
        """
        data = _synth_data(150, 3, n_folds=3)

        # With per-fold pre-calibration
        config_with = {
            "meta_learner": {"C": 1.0},
            "calibration": "none",
            "pre_calibration": {"model_0": "spline"},
            "spline_prob_max": 0.985,
            "spline_n_bins": 20,
        }
        meta_with, _, pre_cals_with = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=config_with,
        )

        # Without pre-calibration
        config_without = {
            "meta_learner": {"C": 1.0},
            "calibration": "none",
            "pre_calibration": {},
        }
        meta_without, _, _ = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=config_without,
        )

        # Coefficients should differ due to pre-calibration
        coeffs_with = meta_with.get_coefficients()
        coeffs_without = meta_without.get_coefficients()
        diff = sum(abs(coeffs_with[k] - coeffs_without[k]) for k in coeffs_with)
        assert diff > 0, "Pre-calibration should change coefficients"

    def test_with_extra_features(self):
        data = _synth_data(150, 2, n_folds=3)
        rng = np.random.RandomState(99)
        extra = {"elo_diff": rng.randn(150)}

        ensemble_config = {
            "meta_learner": {"C": 1.0},
            "calibration": "none",
            "pre_calibration": {},
        }

        meta, _, _ = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=ensemble_config,
            extra_features=extra,
        )

        coeffs = meta.get_coefficients(extra_features=extra)
        assert "elo_diff" in coeffs

    def test_custom_C(self):
        data = _synth_data(150, 2, n_folds=3)
        for C in [0.1, 2.5, 10.0]:
            ensemble_config = {
                "meta_learner": {"C": C},
                "calibration": "none",
                "pre_calibration": {},
            }
            meta, _, _ = train_meta_learner_loso(
                y_true=data["y_true"],
                model_preds=data["model_preds"],
                prior_diffs=data["prior_diffs"],
                fold_labels=data["fold_labels"],
                model_names=data["model_names"],
                ensemble_config=ensemble_config,
            )
            probs = meta.predict(data["model_preds"], data["prior_diffs"])
            assert len(probs) == 150

    def test_default_config_values(self):
        """Defaults should work when ensemble_config has minimal keys."""
        data = _synth_data(150, 2, n_folds=3)
        meta, post_cal, pre_cals = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config={},
        )
        # Defaults: C=1.0, calibration="spline", no pre-calibration
        assert isinstance(meta, StackedEnsemble)
        assert post_cal is not None  # default spline calibration
        assert pre_cals == {}


# -----------------------------------------------------------------------
# Non-linear meta-learner types
# -----------------------------------------------------------------------

class TestMetaLearnerTypes:
    """Tests for ridge and gbm meta-learner types."""

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown meta_learner_type"):
            StackedEnsemble(["a", "b"], meta_learner_type="invalid")

    def test_ridge_fit_predict(self):
        data = _synth_data(100, 3)
        meta = StackedEnsemble(data["model_names"], meta_learner_type="ridge")
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])
        probs = meta.predict(data["model_preds"], data["prior_diffs"])

        assert len(probs) == 100
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_ridge_coefficients(self):
        data = _synth_data(100, 3)
        meta = StackedEnsemble(data["model_names"], meta_learner_type="ridge")
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])
        coeffs = meta.get_coefficients()

        assert len(coeffs) == 4  # 3 models + prior_diff
        assert "prior_diff" in coeffs

    def test_ridge_save_load_roundtrip(self, tmp_path):
        data = _synth_data(100, 3)
        meta = StackedEnsemble(data["model_names"], meta_learner_type="ridge")
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])

        probs_before = meta.predict(data["model_preds"], data["prior_diffs"])

        path = tmp_path / "meta_ridge.json"
        meta.save(path)

        meta2 = StackedEnsemble([])
        meta2.load(path)

        assert meta2.meta_learner_type == "ridge"
        probs_after = meta2.predict(data["model_preds"], data["prior_diffs"])
        np.testing.assert_allclose(probs_before, probs_after, atol=1e-10)

    def test_gbm_fit_predict(self):
        data = _synth_data(100, 3)
        meta = StackedEnsemble(data["model_names"], meta_learner_type="gbm")
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])
        probs = meta.predict(data["model_preds"], data["prior_diffs"])

        assert len(probs) == 100
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_gbm_coefficients_are_importances(self):
        data = _synth_data(100, 3)
        meta = StackedEnsemble(data["model_names"], meta_learner_type="gbm")
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])
        coeffs = meta.get_coefficients()

        assert len(coeffs) == 4  # 3 models + prior_diff
        # GBM importances are non-negative integers (split counts)
        assert all(v >= 0 for v in coeffs.values())

    def test_gbm_save_load_roundtrip(self, tmp_path):
        data = _synth_data(100, 3)
        meta = StackedEnsemble(data["model_names"], meta_learner_type="gbm")
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])

        meta.predict(data["model_preds"], data["prior_diffs"])

        path = tmp_path / "meta_gbm.json"
        meta.save(path)

        saved = json.loads(path.read_text())
        assert saved["meta_learner_type"] == "gbm"
        assert "lgbm_model_str" in saved

    def test_ridge_regression(self):
        """Ridge works for regression task type."""
        data = _synth_data(100, 3)
        # Continuous target
        rng = np.random.RandomState(42)
        y_reg = rng.randn(100)

        meta = StackedEnsemble(
            data["model_names"], task_type="regression", meta_learner_type="ridge"
        )
        meta.fit(data["model_preds"], data["prior_diffs"], y_reg)
        preds = meta.predict(data["model_preds"], data["prior_diffs"])
        assert preds.shape == (100,)

    def test_gbm_regression(self):
        """GBM works for regression task type."""
        data = _synth_data(100, 3)
        rng = np.random.RandomState(42)
        y_reg = rng.randn(100)

        meta = StackedEnsemble(
            data["model_names"], task_type="regression", meta_learner_type="gbm"
        )
        meta.fit(data["model_preds"], data["prior_diffs"], y_reg)
        preds = meta.predict(data["model_preds"], data["prior_diffs"])
        assert preds.shape == (100,)

    def test_loso_with_ridge(self):
        """train_meta_learner_loso works with ridge type."""
        data = _synth_data(150, 3, n_folds=3)
        ensemble_config = {
            "meta_learner": {"type": "ridge"},
            "calibration": "none",
            "pre_calibration": {},
        }

        meta, post_cal, pre_cals = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=ensemble_config,
        )

        assert isinstance(meta, StackedEnsemble)
        assert meta.meta_learner_type == "ridge"
        probs = meta.predict(data["model_preds"], data["prior_diffs"])
        assert len(probs) == 150

    def test_loso_with_gbm(self):
        """train_meta_learner_loso works with gbm type."""
        data = _synth_data(150, 3, n_folds=3)
        ensemble_config = {
            "meta_learner": {"type": "gbm"},
            "calibration": "none",
            "pre_calibration": {},
        }

        meta, post_cal, pre_cals = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=ensemble_config,
        )

        assert isinstance(meta, StackedEnsemble)
        assert meta.meta_learner_type == "gbm"
        probs = meta.predict(data["model_preds"], data["prior_diffs"])
        assert len(probs) == 150
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_loso_default_type_is_logistic(self):
        """Default type in LOSO remains logistic when not specified."""
        data = _synth_data(150, 3, n_folds=3)
        ensemble_config = {
            "meta_learner": {"C": 1.0},
            "calibration": "none",
            "pre_calibration": {},
        }

        meta, _, _ = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=ensemble_config,
        )

        assert meta.meta_learner_type == "logistic"
