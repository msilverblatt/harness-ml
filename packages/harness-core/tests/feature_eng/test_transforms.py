import numpy as np
from harnessml.core.feature_eng.transforms import cyclical_encode


def test_cyclical_encoding():
    months = np.array([1, 6, 12])
    sin_vals, cos_vals = cyclical_encode(months, period=12)
    # January and December should be close in sin values
    assert abs(sin_vals[0] - sin_vals[2]) < 0.6
    # June (6) maps to sin(2*pi*6/12) = sin(pi) ≈ 0, cos = -1
    assert abs(cos_vals[1] - (-1.0)) < 1e-10


def test_cyclical_encoding_hours():
    hours = np.array([0, 6, 12, 18])
    sin_vals, cos_vals = cyclical_encode(hours, period=24)
    assert len(sin_vals) == 4
    assert len(cos_vals) == 4
    # 0 and 24 should produce same values (cyclical)
    s0, c0 = cyclical_encode(np.array([0]), period=24)
    s24, c24 = cyclical_encode(np.array([24]), period=24)
    np.testing.assert_almost_equal(s0, s24)
    np.testing.assert_almost_equal(c0, c24)
