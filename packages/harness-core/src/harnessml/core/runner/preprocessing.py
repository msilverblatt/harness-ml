"""Leakage-safe preprocessing pipeline.

Always fit on training data, transform both train and test.
Never compute statistics on test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler


class Preprocessor:
    def __init__(
        self,
        numeric_strategy: str = "none",  # "none", "zscore", "robust", "quantile"
        categorical_strategy: str = "none",  # "none", "frequency", "ordinal"
        missing_strategy: str = "median",
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.missing_strategy = missing_strategy
        self._fitted = False
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._scaler = None
        self._fill_values: dict[str, float] = {}
        self._frequency_maps: dict[str, dict] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        self._categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

        # Fit missing value imputation
        for col in self._numeric_cols:
            if self.missing_strategy == "median":
                self._fill_values[col] = df[col].median()
            elif self.missing_strategy == "mean":
                self._fill_values[col] = df[col].mean()
            elif self.missing_strategy == "zero":
                self._fill_values[col] = 0.0

        # Fit scaler
        if self.numeric_strategy != "none" and self._numeric_cols:
            scaler_map = {
                "zscore": StandardScaler,
                "robust": RobustScaler,
                "quantile": QuantileTransformer,
            }
            self._scaler = scaler_map[self.numeric_strategy]()
            # Fill NaNs before fitting scaler
            filled = df[self._numeric_cols].copy()
            for col in self._numeric_cols:
                filled[col] = filled[col].fillna(self._fill_values.get(col, 0))
            self._scaler.fit(filled)

        # Fit categorical encoding
        if self.categorical_strategy == "frequency":
            for col in self._categorical_cols:
                counts = df[col].value_counts(normalize=True)
                self._frequency_maps[col] = counts.to_dict()

        self._fitted = True
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Must call fit_transform before transform")

        result = df.copy()

        # Impute missing
        for col in self._numeric_cols:
            if col in result.columns:
                result[col] = result[col].fillna(self._fill_values.get(col, 0))

        # Scale numeric
        if self._scaler is not None:
            cols = [c for c in self._numeric_cols if c in result.columns]
            if cols:
                result[cols] = self._scaler.transform(result[cols])

        # Encode categorical
        if self.categorical_strategy == "frequency":
            for col in self._categorical_cols:
                if col in result.columns:
                    freq_map = self._frequency_maps.get(col, {})
                    result[col] = result[col].map(freq_map).fillna(0)

        return result
