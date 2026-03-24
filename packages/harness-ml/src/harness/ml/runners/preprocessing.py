import pandas as pd


class Preprocessor:
    """Leakage-safe preprocessing — fit on train only."""

    def __init__(self):
        self._medians: dict[str, float] = {}
        self._fitted = False

    def fit(self, X_train: pd.DataFrame) -> "Preprocessor":
        """Compute medians from training data only."""
        for col in X_train.select_dtypes(include="number").columns:
            self._medians[col] = float(X_train[col].median())
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN with fitted medians. Never uses test data stats."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transform")
        result = X.copy()
        for col, median in self._medians.items():
            if col in result.columns:
                result[col] = result[col].fillna(median)
        return result

    @property
    def feature_medians(self) -> dict[str, float]:
        return dict(self._medians)
