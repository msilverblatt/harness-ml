"""Tests for the 5 new guardrails added in PR-05."""

import numpy as np
import pytest
from harnessml.core.guardrails.base import GuardrailError
from harnessml.core.guardrails.inventory import (
    ClassImbalanceGuard,
    DataDistributionGuard,
    FeatureCountGuard,
    ModelComplexityGuard,
    PredictionSanityGuard,
)

# ---------------------------------------------------------------------------
# DataDistributionGuard
# ---------------------------------------------------------------------------


class TestDataDistributionGuard:
    def test_passes_on_clean_data(self):
        g = DataDistributionGuard()
        g.check(context={"missing_rate": 0.1, "skewness": 0.5})

    def test_fails_on_high_missing(self):
        g = DataDistributionGuard()
        with pytest.raises(GuardrailError, match="High missing rate"):
            g.check(context={"missing_rate": 0.75})

    def test_fails_on_extreme_skew(self):
        g = DataDistributionGuard()
        with pytest.raises(GuardrailError, match="Extreme skewness"):
            g.check(context={"skewness": 15.0})

    def test_passes_on_negative_skew_within_bounds(self):
        g = DataDistributionGuard()
        g.check(context={"skewness": -5.0})

    def test_custom_thresholds(self):
        g = DataDistributionGuard(max_missing_rate=0.3, max_skewness=5.0)
        with pytest.raises(GuardrailError):
            g.check(context={"missing_rate": 0.4})
        with pytest.raises(GuardrailError):
            g.check(context={"skewness": 6.0})

    def test_overridable(self):
        g = DataDistributionGuard()
        assert g.overridable is True
        g.check(context={"missing_rate": 0.9}, human_override=True)


# ---------------------------------------------------------------------------
# ClassImbalanceGuard
# ---------------------------------------------------------------------------


class TestClassImbalanceGuard:
    def test_passes_on_balanced(self):
        g = ClassImbalanceGuard()
        g.check(context={"class_distribution": {"0": 0.45, "1": 0.55}})

    def test_fails_on_imbalanced(self):
        g = ClassImbalanceGuard()
        with pytest.raises(GuardrailError, match="Class imbalance"):
            g.check(context={"class_distribution": {"0": 0.97, "1": 0.03}})

    def test_passes_on_empty_distribution(self):
        g = ClassImbalanceGuard()
        g.check(context={"class_distribution": {}})

    def test_custom_threshold(self):
        g = ClassImbalanceGuard(min_class_fraction=0.10)
        with pytest.raises(GuardrailError):
            g.check(context={"class_distribution": {"a": 0.92, "b": 0.08}})
        # Just above threshold passes
        g.check(context={"class_distribution": {"a": 0.89, "b": 0.11}})

    def test_overridable(self):
        g = ClassImbalanceGuard()
        assert g.overridable is True
        g.check(
            context={"class_distribution": {"0": 0.99, "1": 0.01}},
            human_override=True,
        )


# ---------------------------------------------------------------------------
# ModelComplexityGuard
# ---------------------------------------------------------------------------


class TestModelComplexityGuard:
    def test_passes_reasonable_count(self):
        g = ModelComplexityGuard()
        # 5 models, 100 samples -> sqrt(100) = 10 -> pass
        g.check(context={"n_models": 5, "n_samples": 100})

    def test_fails_too_many_models(self):
        g = ModelComplexityGuard()
        # 20 models, 100 samples -> sqrt(100) = 10 -> fail
        with pytest.raises(GuardrailError, match="Too many models"):
            g.check(context={"n_models": 20, "n_samples": 100})

    def test_skips_on_zero(self):
        g = ModelComplexityGuard()
        g.check(context={"n_models": 0, "n_samples": 100})
        g.check(context={"n_models": 5, "n_samples": 0})

    def test_overridable(self):
        g = ModelComplexityGuard()
        assert g.overridable is True
        g.check(
            context={"n_models": 100, "n_samples": 50},
            human_override=True,
        )


# ---------------------------------------------------------------------------
# FeatureCountGuard
# ---------------------------------------------------------------------------


class TestFeatureCountGuard:
    def test_passes_reasonable_count(self):
        g = FeatureCountGuard()
        # 5 features, 100 samples -> sqrt(100) = 10 -> pass
        g.check(context={"n_features": 5, "n_samples": 100})

    def test_fails_too_many_features(self):
        g = FeatureCountGuard()
        # 20 features, 100 samples -> sqrt(100) = 10 -> fail
        with pytest.raises(GuardrailError, match="Too many features"):
            g.check(context={"n_features": 20, "n_samples": 100})

    def test_skips_on_zero(self):
        g = FeatureCountGuard()
        g.check(context={"n_features": 0, "n_samples": 100})
        g.check(context={"n_features": 5, "n_samples": 0})

    def test_overridable(self):
        g = FeatureCountGuard()
        assert g.overridable is True


# ---------------------------------------------------------------------------
# PredictionSanityGuard
# ---------------------------------------------------------------------------


class TestPredictionSanityGuard:
    def test_passes_valid_predictions(self):
        g = PredictionSanityGuard()
        g.check(context={"predictions": [0.1, 0.4, 0.6, 0.9]})

    def test_fails_out_of_range(self):
        g = PredictionSanityGuard()
        with pytest.raises(GuardrailError, match="out of.*range"):
            g.check(context={"predictions": [0.1, -0.5, 0.8]})

    def test_fails_above_one(self):
        g = PredictionSanityGuard()
        with pytest.raises(GuardrailError, match="out of.*range"):
            g.check(context={"predictions": [0.1, 0.5, 1.5]})

    def test_fails_degenerate_all_same(self):
        g = PredictionSanityGuard()
        with pytest.raises(GuardrailError, match="Degenerate"):
            g.check(context={"predictions": [0.5, 0.5, 0.5, 0.5]})

    def test_passes_with_sufficient_variance(self):
        g = PredictionSanityGuard()
        g.check(context={"predictions": [0.2, 0.5, 0.8]})

    def test_passes_on_empty(self):
        g = PredictionSanityGuard()
        g.check(context={"predictions": []})

    def test_passes_on_none(self):
        g = PredictionSanityGuard()
        g.check(context={})

    def test_non_overridable(self):
        g = PredictionSanityGuard()
        assert g.overridable is False
        with pytest.raises(GuardrailError):
            g.check(
                context={"predictions": [-0.1, 0.5]},
                human_override=True,
            )

    def test_numpy_array_input(self):
        g = PredictionSanityGuard()
        g.check(context={"predictions": np.array([0.1, 0.4, 0.7, 0.9])})

    def test_custom_degenerate_threshold(self):
        g = PredictionSanityGuard(degenerate_threshold=0.05)
        # std of [0.49, 0.5, 0.51] ≈ 0.008 < 0.05
        with pytest.raises(GuardrailError, match="Degenerate"):
            g.check(context={"predictions": [0.49, 0.5, 0.51]})
