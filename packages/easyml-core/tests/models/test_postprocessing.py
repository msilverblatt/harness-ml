"""Tests for ensemble post-processing steps."""
import numpy as np
import pytest

from easyml.core.models.postprocessing import (
    EnsemblePostprocessor,
    ProbabilityClipping,
    TemperatureScaling,
)


def test_probability_clipping():
    clip = ProbabilityClipping(floor=0.05, ceiling=0.95)
    probs = np.array([0.01, 0.5, 0.99])
    result = clip.apply(probs)
    np.testing.assert_array_almost_equal(result, [0.05, 0.5, 0.95])


def test_temperature_scaling():
    temp = TemperatureScaling(T=2.0)
    probs = np.array([0.8, 0.2])
    result = temp.apply(probs)
    # T>1 should push toward 0.5
    assert result[0] < 0.8
    assert result[1] > 0.2


def test_temperature_scaling_identity():
    """T=1.0 should leave probabilities unchanged."""
    temp = TemperatureScaling(T=1.0)
    probs = np.array([0.1, 0.5, 0.9])
    result = temp.apply(probs)
    np.testing.assert_array_almost_equal(result, probs, decimal=10)


def test_temperature_scaling_sharpening():
    """T<1 should push probabilities toward extremes."""
    temp = TemperatureScaling(T=0.5)
    probs = np.array([0.8, 0.2])
    result = temp.apply(probs)
    assert result[0] > 0.8
    assert result[1] < 0.2


def test_temperature_invalid():
    with pytest.raises(ValueError, match="positive"):
        TemperatureScaling(T=0.0)
    with pytest.raises(ValueError, match="positive"):
        TemperatureScaling(T=-1.0)


def test_postprocessor_chain():
    chain = EnsemblePostprocessor(steps=[
        ("clip", ProbabilityClipping(floor=0.05, ceiling=0.95)),
        ("temperature", TemperatureScaling(T=1.5)),
    ])
    probs = np.array([0.01, 0.5, 0.99])
    result = chain.apply(probs)
    assert all(0 < p < 1 for p in result)


def test_empty_chain():
    chain = EnsemblePostprocessor(steps=[])
    probs = np.array([0.3, 0.7])
    np.testing.assert_array_equal(chain.apply(probs), probs)


def test_clipping_invalid_range():
    with pytest.raises(ValueError, match="Invalid clip range"):
        ProbabilityClipping(floor=0.9, ceiling=0.1)
    with pytest.raises(ValueError, match="Invalid clip range"):
        ProbabilityClipping(floor=-0.1, ceiling=0.5)


def test_chain_ordering_matters():
    """Clip then temperature should differ from temperature then clip."""
    probs = np.array([0.01, 0.99])

    chain_a = EnsemblePostprocessor(steps=[
        ("clip", ProbabilityClipping(floor=0.05, ceiling=0.95)),
        ("temp", TemperatureScaling(T=2.0)),
    ])
    chain_b = EnsemblePostprocessor(steps=[
        ("temp", TemperatureScaling(T=2.0)),
        ("clip", ProbabilityClipping(floor=0.05, ceiling=0.95)),
    ])

    result_a = chain_a.apply(probs)
    result_b = chain_b.apply(probs)
    # Results should differ because order matters
    assert not np.allclose(result_a, result_b)
