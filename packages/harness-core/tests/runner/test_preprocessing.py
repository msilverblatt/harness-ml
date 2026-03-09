import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.preprocessing import Preprocessor


def test_preprocessing_fits_on_train_transforms_test():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "c"]})
    test = pd.DataFrame({"x": [5.0, 6.0], "cat": ["a", "d"]})
    pp = Preprocessor(numeric_strategy="zscore", categorical_strategy="frequency")
    train_transformed = pp.fit_transform(train)
    test_transformed = pp.transform(test)
    # Train should be standardized based on train stats
    assert abs(train_transformed["x"].mean()) < 0.01
    # Test should be transformed with train stats (not test stats)
    assert test_transformed["x"].mean() != 0  # Not centered on test mean


def test_preprocessing_handles_missing():
    pp = Preprocessor(numeric_strategy="zscore")
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0]})
    result = pp.fit_transform(train)
    assert not result["x"].isna().any()


def test_robust_scaler():
    pp = Preprocessor(numeric_strategy="robust")
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0]})
    result = pp.fit_transform(train)
    # Robust scaler should not be dominated by outlier
    assert abs(result["x"].iloc[1]) < 1  # Middle value close to 0


def test_frequency_encoding():
    pp = Preprocessor(categorical_strategy="frequency")
    train = pd.DataFrame({"cat": ["a", "a", "b", "c"]})
    result = pp.fit_transform(train)
    assert result["cat"].iloc[0] == 0.5  # "a" appears 2/4 = 0.5


def test_transform_before_fit_raises():
    pp = Preprocessor()
    with pytest.raises(RuntimeError, match="fit_transform"):
        pp.transform(pd.DataFrame({"x": [1.0]}))


def test_quantile_strategy():
    pp = Preprocessor(numeric_strategy="quantile")
    train = pd.DataFrame({"x": np.random.randn(100)})
    result = pp.fit_transform(train)
    assert not result["x"].isna().any()
