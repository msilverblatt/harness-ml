import numpy as np
import pandas as pd
import pytest
from harness.ml.features.pairwise import generate_pairwise_derivatives


class TestGeneratePairwiseDerivatives:
    def test_diff(self):
        df = pd.DataFrame({
            "entity_a_score": [1.0, 3.0, 5.0],
            "entity_b_score": [2.0, 1.0, 4.0],
        })
        result = generate_pairwise_derivatives(df, "score", methods=["diff"])
        expected = [-1.0, 2.0, 1.0]
        np.testing.assert_array_almost_equal(result["diff_score"].values, expected)

    def test_ratio(self):
        df = pd.DataFrame({
            "entity_a_rating": [80.0, 60.0],
            "entity_b_rating": [40.0, 30.0],
        })
        result = generate_pairwise_derivatives(df, "rating", methods=["ratio"])
        expected = [2.0, 2.0]
        np.testing.assert_array_almost_equal(result["ratio_rating"].values, expected)

    def test_both_methods(self):
        df = pd.DataFrame({
            "entity_a_x": [10.0, 20.0],
            "entity_b_x": [5.0, 10.0],
        })
        result = generate_pairwise_derivatives(df, "x", methods=["diff", "ratio"])
        assert "diff_x" in result.columns
        assert "ratio_x" in result.columns
        np.testing.assert_array_almost_equal(result["diff_x"].values, [5.0, 10.0])
        np.testing.assert_array_almost_equal(result["ratio_x"].values, [2.0, 2.0])

    def test_zero_denominator_safety(self):
        df = pd.DataFrame({
            "entity_a_val": [10.0, 5.0],
            "entity_b_val": [0.0, 2.0],
        })
        result = generate_pairwise_derivatives(df, "val", methods=["ratio"])
        assert result["ratio_val"].iloc[0] == 0.0
        np.testing.assert_almost_equal(result["ratio_val"].iloc[1], 2.5)

    def test_custom_prefix(self):
        df = pd.DataFrame({
            "home_score": [3.0],
            "away_score": [1.0],
        })
        result = generate_pairwise_derivatives(
            df, "score", methods=["diff"],
            entity_a_prefix="home_", entity_b_prefix="away_",
        )
        assert result["diff_score"].iloc[0] == 2.0

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"unrelated": [1.0]})
        with pytest.raises(ValueError, match="Entity columns not found"):
            generate_pairwise_derivatives(df, "score", methods=["diff"])

    def test_unknown_method_raises(self):
        df = pd.DataFrame({
            "entity_a_x": [1.0],
            "entity_b_x": [1.0],
        })
        with pytest.raises(ValueError, match="Unknown pairwise method"):
            generate_pairwise_derivatives(df, "x", methods=["bogus"])
