import numpy as np
import pandas as pd
import pytest

from harness.ml.runners.preprocessing import Preprocessor


def make_train():
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, np.nan],
            "b": [10.0, np.nan, 30.0, 40.0, 50.0],
        }
    )


def make_test():
    return pd.DataFrame(
        {
            "a": [np.nan, 100.0],
            "b": [np.nan, 200.0],
        }
    )


class TestPreprocessorFit:
    def test_fit_computes_medians_from_train(self):
        X_train = make_train()
        p = Preprocessor()
        p.fit(X_train)
        # median of [1, 2, 3, 4] (ignoring nan) = 2.5
        assert p.feature_medians["a"] == pytest.approx(2.5)
        # median of [10, 30, 40, 50] (ignoring nan) = 35.0
        assert p.feature_medians["b"] == pytest.approx(35.0)

    def test_fit_sets_fitted_flag(self):
        p = Preprocessor()
        assert not p._fitted
        p.fit(make_train())
        assert p._fitted

    def test_fit_returns_self(self):
        p = Preprocessor()
        result = p.fit(make_train())
        assert result is p


class TestPreprocessorTransform:
    def test_transform_fills_nan_with_fitted_medians(self):
        X_train = make_train()
        p = Preprocessor().fit(X_train)
        result = p.transform(X_train)
        assert result["a"].isna().sum() == 0
        assert result["b"].isna().sum() == 0
        # The NaN in 'a' at index 4 should be filled with 2.5
        assert result["a"].iloc[4] == pytest.approx(2.5)
        # The NaN in 'b' at index 1 should be filled with 35.0
        assert result["b"].iloc[1] == pytest.approx(35.0)

    def test_transform_raises_if_not_fitted(self):
        p = Preprocessor()
        with pytest.raises(RuntimeError, match="fit before transform"):
            p.transform(make_train())

    def test_transform_does_not_mutate_original(self):
        X_train = make_train()
        p = Preprocessor().fit(X_train)
        original_nan_count = X_train["a"].isna().sum()
        p.transform(X_train)
        assert X_train["a"].isna().sum() == original_nan_count

    def test_train_median_used_for_test_not_test_stats(self):
        """Fit on train (small values), transform test (large values) — median from train is used."""
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        X_test = pd.DataFrame({"a": [np.nan, 1000.0, 2000.0]})

        p = Preprocessor().fit(X_train)
        # Train median is 2.0
        assert p.feature_medians["a"] == pytest.approx(2.0)

        result = p.transform(X_test)
        # NaN in test should be filled with train median (2.0), NOT test median (~1500)
        assert result["a"].iloc[0] == pytest.approx(2.0)
        # Non-NaN values unchanged
        assert result["a"].iloc[1] == pytest.approx(1000.0)
