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
        categorical_strategy: str = "none",  # "none", "frequency", "ordinal", "target"
        missing_strategy: str = "median",
        knn_neighbors: int = 5,
        smoothing: float = 10.0,
    ):
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.missing_strategy = missing_strategy
        self.knn_neighbors = knn_neighbors
        self.smoothing = smoothing
        self._fitted = False
        self._numeric_cols: list[str] = []
        self._categorical_cols: list[str] = []
        self._scaler = None
        self._fill_values: dict[str, float] = {}
        self._frequency_maps: dict[str, dict] = {}
        self._target_encode_maps: dict[str, dict] = {}
        self._global_mean: float = 0.0
        self._imputer = None

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        self._numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
        self._categorical_cols = list(df.select_dtypes(exclude=[np.number]).columns)

        # Fit missing value imputation
        if self.missing_strategy == "knn":
            from sklearn.impute import KNNImputer

            self._imputer = KNNImputer(n_neighbors=self.knn_neighbors)
            self._imputer.fit(df[self._numeric_cols])
        elif self.missing_strategy == "iterative":
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            self._imputer = IterativeImputer()
            self._imputer.fit(df[self._numeric_cols])
        else:
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
            if self._imputer is not None:
                filled = pd.DataFrame(
                    self._imputer.transform(filled),
                    columns=self._numeric_cols,
                    index=filled.index,
                )
            else:
                for col in self._numeric_cols:
                    filled[col] = filled[col].fillna(self._fill_values.get(col, 0))
            self._scaler.fit(filled)

        # Fit categorical encoding
        if self.categorical_strategy == "frequency":
            for col in self._categorical_cols:
                counts = df[col].value_counts(normalize=True)
                self._frequency_maps[col] = counts.to_dict()
        elif self.categorical_strategy == "target":
            if y is None:
                raise ValueError(
                    "Target encoding requires y (target series) to be passed "
                    "to fit_transform()."
                )
            self._global_mean = float(y.mean())
            for col in self._categorical_cols:
                group = pd.DataFrame({"cat": df[col], "target": y}).groupby("cat")["target"]
                cat_sum = group.sum()
                cat_count = group.count()
                # Bayesian smoothing: (sum + global_mean * smoothing) / (count + smoothing)
                smoothed = (cat_sum + self._global_mean * self.smoothing) / (cat_count + self.smoothing)
                self._target_encode_maps[col] = smoothed.to_dict()

        self._fitted = True
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Must call fit_transform before transform")

        result = df.copy()

        # Impute missing
        if self._imputer is not None:
            num_cols = [c for c in self._numeric_cols if c in result.columns]
            if num_cols:
                result[num_cols] = self._imputer.transform(result[num_cols])
        else:
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
        elif self.categorical_strategy == "target":
            for col in self._categorical_cols:
                if col in result.columns:
                    encode_map = self._target_encode_maps.get(col, {})
                    result[col] = result[col].map(encode_map).fillna(self._global_mean)

        return result
