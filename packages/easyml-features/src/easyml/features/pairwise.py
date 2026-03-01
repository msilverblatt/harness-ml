"""Pairwise feature builder — computes diff/ratio features for entity matchups."""
from __future__ import annotations

import pandas as pd


# Small constant to avoid division by zero in ratio computation.
_EPSILON = 1e-12


class PairwiseFeatureBuilder:
    """Builds pairwise (matchup-level) features from entity-level data.

    Supports ``"diff"`` (a - b) and ``"ratio"`` (a / b) methods.
    """

    def __init__(self, methods: list[str]) -> None:
        valid = {"diff", "ratio"}
        for m in methods:
            if m not in valid:
                raise ValueError(
                    f"Unknown pairwise method '{m}'. Must be one of {valid}"
                )
        self._methods = methods

    def build(
        self,
        entity_df: pd.DataFrame,
        matchups: pd.DataFrame,
        feature_columns: list[str],
    ) -> pd.DataFrame:
        """Merge entity features onto matchups and compute pairwise columns.

        Parameters
        ----------
        entity_df:
            Must contain ``entity_id``, ``period_id``, and all *feature_columns*.
        matchups:
            Must contain ``entity_a_id``, ``entity_b_id``, ``period_id``.
        feature_columns:
            Which columns from *entity_df* to create pairwise features for.

        Returns
        -------
        DataFrame with the original matchup columns plus new pairwise columns.
        """
        result = matchups.copy()

        # Merge entity_a features
        a_cols = {col: f"_a_{col}" for col in feature_columns}
        a_df = entity_df[["entity_id", "period_id"] + feature_columns].rename(
            columns={"entity_id": "entity_a_id", **a_cols}
        )
        result = result.merge(a_df, on=["entity_a_id", "period_id"], how="left")

        # Merge entity_b features
        b_cols = {col: f"_b_{col}" for col in feature_columns}
        b_df = entity_df[["entity_id", "period_id"] + feature_columns].rename(
            columns={"entity_id": "entity_b_id", **b_cols}
        )
        result = result.merge(b_df, on=["entity_b_id", "period_id"], how="left")

        # Compute pairwise features
        for col in feature_columns:
            a_name = f"_a_{col}"
            b_name = f"_b_{col}"

            if "diff" in self._methods:
                result[f"diff_{col}"] = result[a_name] - result[b_name]

            if "ratio" in self._methods:
                result[f"ratio_{col}"] = result[a_name] / (
                    result[b_name] + _EPSILON
                )

        # Drop the intermediate _a_ / _b_ columns
        drop_cols = [
            c for c in result.columns if c.startswith("_a_") or c.startswith("_b_")
        ]
        result = result.drop(columns=drop_cols)

        return result
