"""Tests for multiclass stacked meta-learner."""
from __future__ import annotations

import numpy as np
import pytest
from harnessml.core.runner.training.meta_learner import StackedEnsemble, train_meta_learner_loso

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _synth_multiclass_data(
    n: int = 150, n_models: int = 3, n_classes: int = 3, n_folds: int = 3, seed: int = 42,
):
    """Generate synthetic multiclass data for meta-learner tests.

    Each model produces (n, n_classes) probability arrays that sum to 1.
    """
    rng = np.random.RandomState(seed)

    # True class labels
    y_true = rng.randint(0, n_classes, size=n).astype(float)

    model_names = [f"model_{i}" for i in range(n_models)]
    model_preds = {}
    for name in model_names:
        # Generate raw scores, softmax to get valid probabilities
        raw = rng.randn(n, n_classes)
        # Bias toward the true class so models are informative
        for i in range(n):
            raw[i, int(y_true[i])] += 1.5
        exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
        model_preds[name] = exp_raw / exp_raw.sum(axis=1, keepdims=True)

    prior_diffs = np.zeros(n)  # No prior concept for multiclass
    fold_labels = np.array([2020 + (i % n_folds) for i in range(n)])

    return {
        "y_true": y_true,
        "model_preds": model_preds,
        "prior_diffs": prior_diffs,
        "fold_labels": fold_labels,
        "model_names": model_names,
        "n_classes": n_classes,
    }


# -----------------------------------------------------------------------
# StackedEnsemble (multiclass)
# -----------------------------------------------------------------------

class TestStackedEnsembleMulticlass:
    def test_fit_predict_basic(self):
        """Multiclass predict returns (n_samples, n_classes) summing to ~1."""
        data = _synth_multiclass_data(100, 3, n_classes=3)
        meta = StackedEnsemble(data["model_names"], n_classes=data["n_classes"])
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])
        probs = meta.predict(data["model_preds"], data["prior_diffs"])

        assert probs.shape == (100, 3)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_coefficients_per_class(self):
        """Multiclass coefficients returned per class."""
        data = _synth_multiclass_data(100, 3, n_classes=3)
        meta = StackedEnsemble(data["model_names"], n_classes=data["n_classes"])
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])
        coeffs = meta.get_coefficients()

        # Should have n_classes entries, each with features
        assert isinstance(coeffs, dict)
        assert len(coeffs) == 3
        for cls_idx in range(3):
            key = f"class_{cls_idx}"
            assert key in coeffs
            # Each class has: 3 models * 3 classes per model + 1 prior_diff = 10 features
            assert len(coeffs[key]) == 10

    def test_four_classes(self):
        """Verify 4-class support works correctly."""
        data = _synth_multiclass_data(200, 2, n_classes=4)
        meta = StackedEnsemble(data["model_names"], n_classes=data["n_classes"])
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])
        probs = meta.predict(data["model_preds"], data["prior_diffs"])

        assert probs.shape == (200, 4)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_before_fit_raises(self):
        meta = StackedEnsemble(["a", "b"], n_classes=3)
        preds = {"a": np.ones((5, 3)) / 3, "b": np.ones((5, 3)) / 3}
        with pytest.raises(RuntimeError, match="not been fitted"):
            meta.predict(preds, np.zeros(5))

    def test_with_extra_features(self):
        data = _synth_multiclass_data(100, 2, n_classes=3)
        rng = np.random.RandomState(99)
        extra = {"feat_a": rng.randn(100)}

        meta = StackedEnsemble(data["model_names"], n_classes=data["n_classes"])
        meta.fit(
            data["model_preds"], data["prior_diffs"], data["y_true"],
            extra_features=extra,
        )
        probs = meta.predict(
            data["model_preds"], data["prior_diffs"],
            extra_features=extra,
        )

        assert probs.shape == (100, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

        coeffs = meta.get_coefficients(extra_features=extra)
        # Each class should have feat_a in its coefficients
        for cls_idx in range(3):
            assert "feat_a" in coeffs[f"class_{cls_idx}"]

    def test_save_load_roundtrip(self, tmp_path):
        data = _synth_multiclass_data(100, 3, n_classes=3)
        meta = StackedEnsemble(data["model_names"], n_classes=data["n_classes"])
        meta.fit(data["model_preds"], data["prior_diffs"], data["y_true"])

        probs_before = meta.predict(data["model_preds"], data["prior_diffs"])

        path = tmp_path / "meta_multiclass.json"
        meta.save(path)

        meta2 = StackedEnsemble([], n_classes=2)  # Will be overwritten by load
        meta2.load(path)

        assert meta2.n_classes == 3
        probs_after = meta2.predict(data["model_preds"], data["prior_diffs"])
        np.testing.assert_allclose(probs_before, probs_after, atol=1e-10)

    def test_binary_backward_compatible(self):
        """Default n_classes=2 keeps existing binary behavior."""
        rng = np.random.RandomState(42)
        n = 100
        y = (rng.rand(n) > 0.5).astype(float)
        model_preds = {"m1": np.clip(rng.rand(n), 0.05, 0.95)}
        prior_diffs = rng.randn(n)

        meta = StackedEnsemble(["m1"])  # n_classes defaults to 2
        meta.fit(model_preds, prior_diffs, y)
        probs = meta.predict(model_preds, prior_diffs)

        # Binary: returns 1D array
        assert probs.ndim == 1
        assert len(probs) == n


# -----------------------------------------------------------------------
# train_meta_learner_loso (multiclass)
# -----------------------------------------------------------------------

class TestTrainMetaLearnerLosoMulticlass:
    def test_basic_loso_multiclass(self):
        data = _synth_multiclass_data(150, 3, n_classes=3, n_folds=3)
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
            n_classes=data["n_classes"],
        )

        assert isinstance(meta, StackedEnsemble)
        assert meta.n_classes == 3
        assert post_cal is None  # calibration="none"
        assert pre_cals == {}

        probs = meta.predict(data["model_preds"], data["prior_diffs"])
        assert probs.shape == (150, 3)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_loso_skips_pre_calibration_multiclass(self):
        """Pre-calibration should be skipped for multiclass."""
        data = _synth_multiclass_data(150, 3, n_classes=3, n_folds=3)
        ensemble_config = {
            "meta_learner": {"C": 1.0},
            "calibration": "none",
            "pre_calibration": {"model_0": "spline"},
        }

        meta, post_cal, pre_cals = train_meta_learner_loso(
            y_true=data["y_true"],
            model_preds=data["model_preds"],
            prior_diffs=data["prior_diffs"],
            fold_labels=data["fold_labels"],
            model_names=data["model_names"],
            ensemble_config=ensemble_config,
            n_classes=data["n_classes"],
        )

        # Pre-calibrators should be empty for multiclass
        assert pre_cals == {}

    def test_loso_default_n_classes_is_binary(self):
        """Default n_classes=2 keeps binary behavior in train_meta_learner_loso."""
        rng = np.random.RandomState(42)
        n = 150
        y = (rng.rand(n) > 0.5).astype(float)
        model_preds = {"m1": np.clip(rng.rand(n), 0.05, 0.95)}
        prior_diffs = rng.randn(n)
        fold_labels = np.array([2020 + (i % 3) for i in range(n)])

        meta, _, _ = train_meta_learner_loso(
            y_true=y,
            model_preds=model_preds,
            prior_diffs=prior_diffs,
            fold_labels=fold_labels,
            model_names=["m1"],
            ensemble_config={"calibration": "none", "pre_calibration": {}},
        )

        probs = meta.predict(model_preds, prior_diffs)
        assert probs.ndim == 1
