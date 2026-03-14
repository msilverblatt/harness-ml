"""TabPFN wrapper (optional dependency).

TabPFN is a pre-trained transformer for tabular classification. It requires no
hyperparameter tuning and works best on small-to-medium datasets (<10k rows,
<100 features).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from harnessml.core.models.base import BaseModel

logger = logging.getLogger(__name__)

_MAX_ROWS_WARN = 10_000
_MAX_FEATURES_WARN = 100


class TabPFNModel(BaseModel):
    """Wrapper around TabPFNClassifier for tabular classification.

    TabPFN is a pre-trained transformer that performs in-context learning on
    tabular data. It supports binary and multiclass classification natively.

    Parameters
    ----------
    params : dict | None
        Forwarded to ``TabPFNClassifier``.
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._build_model()

    def _build_model(self) -> None:
        try:
            from tabpfn import TabPFNClassifier
            self._model = TabPFNClassifier(**self.params)
        except ImportError:
            raise ImportError(
                "TabPFN is required for TabPFNModel. "
                "Install with: pip install tabpfn"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        if sample_weight is not None:
            logger.warning(
                "TabPFN does not support sample_weight; ignoring."
            )
        n_rows, n_features = X.shape
        if n_rows > _MAX_ROWS_WARN:
            logger.warning(
                "TabPFN is designed for datasets with <%d rows; "
                "got %d. Performance may degrade.",
                _MAX_ROWS_WARN,
                n_rows,
            )
        if n_features > _MAX_FEATURES_WARN:
            logger.warning(
                "TabPFN is designed for datasets with <%d features; "
                "got %d. Performance may degrade.",
                _MAX_FEATURES_WARN,
                n_features,
            )
        self._model.fit(X, y)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self._model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        return probs

    @property
    def is_regression(self) -> bool:
        return False

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"params": self.params}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> TabPFNModel:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        instance = cls.__new__(cls)
        instance.params = meta.get("params", {})
        instance._fitted = False
        instance._build_model()
        return instance
