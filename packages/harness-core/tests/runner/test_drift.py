import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.analysis.drift import detect_drift, detect_multi_feature_drift


def test_ks_no_drift():
    np.random.seed(42)
    ref = np.random.randn(1000)
    cur = np.random.randn(1000)
    result = detect_drift(ref, cur, method="ks")
    assert not result.is_drifted
    assert result.p_value > 0.05


def test_ks_detects_drift():
    np.random.seed(42)
    ref = np.random.randn(1000)
    cur = np.random.randn(1000) + 3.0
    result = detect_drift(ref, cur, method="ks")
    assert result.is_drifted
    assert result.p_value < 0.05


def test_psi_no_drift():
    np.random.seed(42)
    ref = np.random.randn(1000)
    cur = np.random.randn(1000)
    result = detect_drift(ref, cur, method="psi")
    assert not result.is_drifted
    assert result.psi < 0.2


def test_psi_detects_drift():
    np.random.seed(42)
    ref = np.random.randn(1000)
    cur = np.random.randn(1000) + 5.0
    result = detect_drift(ref, cur, method="psi")
    assert result.is_drifted
    assert result.psi > 0.2


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown"):
        detect_drift(np.array([1.0]), np.array([1.0]), method="bogus")


def test_multi_feature_drift():
    np.random.seed(42)
    ref_df = pd.DataFrame({"a": np.random.randn(500), "b": np.random.randn(500)})
    cur_df = pd.DataFrame({"a": np.random.randn(500) + 5.0, "b": np.random.randn(500)})
    results = detect_multi_feature_drift(ref_df, cur_df)
    assert results["a"].is_drifted
    assert not results["b"].is_drifted
