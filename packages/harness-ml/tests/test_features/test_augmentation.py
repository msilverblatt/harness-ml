import numpy as np
import pandas as pd
import pytest
from harness.ml.features.augmentation import augment_symmetric


class TestAugmentSymmetric:
    def _base_df(self):
        return pd.DataFrame({
            "diff_elo": [10.0, -5.0],
            "ratio_elo": [2.0, 0.5],
            "surface_clay": [1, 0],
            "target": [1, 0],
        })

    def test_doubles_rows(self):
        df = self._base_df()
        result = augment_symmetric(df, target_col="target")
        assert len(result) == 2 * len(df)

    def test_diff_negated(self):
        df = self._base_df()
        result = augment_symmetric(df, target_col="target")
        original_diff = result["diff_elo"].iloc[:2].values
        reversed_diff = result["diff_elo"].iloc[2:].values
        np.testing.assert_array_almost_equal(reversed_diff, -original_diff)

    def test_ratio_inverted(self):
        df = self._base_df()
        result = augment_symmetric(df, target_col="target")
        np.testing.assert_almost_equal(result["ratio_elo"].iloc[2], 0.5)
        np.testing.assert_almost_equal(result["ratio_elo"].iloc[3], 2.0)

    def test_binary_target_flipped(self):
        df = self._base_df()
        result = augment_symmetric(df, target_col="target", task_type="binary")
        assert result["target"].iloc[0] == 1
        assert result["target"].iloc[1] == 0
        assert result["target"].iloc[2] == 0
        assert result["target"].iloc[3] == 1

    def test_regression_target_negated(self):
        df = pd.DataFrame({
            "diff_x": [3.0],
            "value": [5.0],
        })
        result = augment_symmetric(df, target_col="value", task_type="regression")
        assert result["value"].iloc[0] == 5.0
        assert result["value"].iloc[1] == -5.0

    def test_non_diff_features_unchanged(self):
        df = self._base_df()
        result = augment_symmetric(df, target_col="target")
        np.testing.assert_array_equal(
            result["surface_clay"].iloc[:2].values,
            result["surface_clay"].iloc[2:].values,
        )

    def test_zero_ratio_safety(self):
        df = pd.DataFrame({
            "ratio_x": [0.0, 3.0],
            "target": [1, 0],
        })
        result = augment_symmetric(df, target_col="target")
        assert result["ratio_x"].iloc[2] == 0.0
        np.testing.assert_almost_equal(result["ratio_x"].iloc[3], 1.0 / 3.0)
