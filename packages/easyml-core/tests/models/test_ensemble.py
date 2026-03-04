"""Tests for StackedEnsemble."""
import numpy as np
import pytest

from easyml.core.models.ensemble import StackedEnsemble
from easyml.core.models.cv import LeaveOneSeasonOut
from easyml.core.models.calibration import SplineCalibrator


def test_stacked_ensemble_basic():
    ensemble = StackedEnsemble(
        method="stacked",
        meta_learner_type="logistic",
        meta_learner_params={"C": 1.0},
    )
    preds = {
        "model_a": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),
        "model_b": np.array([0.8, 0.2, 0.7, 0.3, 0.6, 0.4]),
    }
    y = np.array([1, 0, 1, 0, 1, 0])
    fold_ids = np.array([1, 1, 2, 2, 3, 3])

    ensemble.fit(preds, y, cv=LeaveOneSeasonOut(), fold_ids=fold_ids)
    final = ensemble.predict(preds)
    assert final.shape == (6,)
    assert all(0 < p < 1 for p in final)

    coeffs = ensemble.coefficients()
    assert "model_a" in coeffs
    assert "model_b" in coeffs


def test_stacked_ensemble_average():
    ensemble = StackedEnsemble(method="average")
    preds = {
        "model_a": np.array([0.8, 0.2]),
        "model_b": np.array([0.6, 0.4]),
    }
    y = np.array([1, 0])
    fold_ids = np.array([1, 2])
    ensemble.fit(preds, y, cv=LeaveOneSeasonOut(), fold_ids=fold_ids)
    final = ensemble.predict(preds)
    np.testing.assert_array_almost_equal(final, [0.7, 0.3])


def test_ensemble_coefficients_only_for_stacked():
    ensemble = StackedEnsemble(method="average")
    preds = {"a": np.array([0.5, 0.5])}
    y = np.array([1, 0])
    ensemble.fit(preds, y, cv=LeaveOneSeasonOut(), fold_ids=np.array([1, 2]))
    with pytest.raises(ValueError, match="stacked"):
        ensemble.coefficients()


def test_ensemble_save_load(tmp_path):
    ensemble = StackedEnsemble(method="stacked", meta_learner_params={"C": 1.0})
    preds = {"a": np.array([0.9, 0.1, 0.8, 0.2]), "b": np.array([0.8, 0.2, 0.7, 0.3])}
    y = np.array([1, 0, 1, 0])
    fold_ids = np.array([1, 1, 2, 2])
    ensemble.fit(preds, y, cv=LeaveOneSeasonOut(), fold_ids=fold_ids)
    ensemble.save(tmp_path / "ensemble")
    loaded = StackedEnsemble.load(tmp_path / "ensemble")
    np.testing.assert_array_almost_equal(ensemble.predict(preds), loaded.predict(preds))


def test_stacked_ensemble_predict_before_fit():
    ensemble = StackedEnsemble(method="stacked")
    with pytest.raises(RuntimeError, match="not fitted"):
        ensemble.predict({"a": np.array([0.5])})


def test_stacked_requires_cv_and_fold_ids():
    ensemble = StackedEnsemble(method="stacked")
    preds = {"a": np.array([0.5, 0.5])}
    y = np.array([1, 0])
    with pytest.raises(ValueError, match="cv and fold_ids"):
        ensemble.fit(preds, y, cv=None, fold_ids=None)


def test_invalid_method():
    with pytest.raises(ValueError, match="Unknown method"):
        StackedEnsemble(method="invalid")


def test_stacked_with_pre_calibrate():
    """Per-fold pre-calibration should apply calibrator inside the CV loop."""
    ensemble = StackedEnsemble(
        method="stacked",
        meta_learner_params={"C": 1.0},
        pre_calibrate={"model_a": SplineCalibrator(n_bins=5)},
    )
    # Use enough data for spline to fit
    np.random.seed(42)
    n = 60
    fold_ids = np.array([1] * 20 + [2] * 20 + [3] * 20)
    y = np.concatenate([
        np.array([1] * 10 + [0] * 10),
        np.array([1] * 10 + [0] * 10),
        np.array([1] * 10 + [0] * 10),
    ])
    preds = {
        "model_a": np.where(y == 1, np.random.uniform(0.6, 0.95, n), np.random.uniform(0.05, 0.4, n)),
        "model_b": np.where(y == 1, np.random.uniform(0.55, 0.9, n), np.random.uniform(0.1, 0.45, n)),
    }

    ensemble.fit(preds, y, cv=LeaveOneSeasonOut(), fold_ids=fold_ids)
    result = ensemble.predict(preds)
    assert result.shape == (n,)
    assert all(0 < p < 1 for p in result)
