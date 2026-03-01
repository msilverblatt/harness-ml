"""XGBoost wrapper supporting both classifier and regressor modes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from easyml.models.base import BaseModel


class XGBoostModel(BaseModel):
    """Wrapper around XGBClassifier / XGBRegressor.

    Parameters
    ----------
    params : dict | None
        Forwarded to the underlying XGBoost estimator.
    mode : str
        ``"classifier"`` or ``"regressor"``.
    cdf_scale : float | None
        For regressor mode: scale parameter for the normal CDF used to convert
        raw margins to probabilities.  Required when calling ``predict_proba``
        on a regressor.
    """

    def __init__(
        self,
        params: dict | None = None,
        *,
        mode: str = "classifier",
        cdf_scale: float | None = None,
    ):
        super().__init__(params)
        if mode not in ("classifier", "regressor"):
            raise ValueError(f"mode must be 'classifier' or 'regressor', got {mode!r}")
        self._mode = mode
        self._cdf_scale = cdf_scale
        self._build_model()

    def _build_model(self) -> None:
        if self._mode == "classifier":
            from xgboost import XGBClassifier

            self._model = XGBClassifier(**self.params)
        else:
            from xgboost import XGBRegressor

            self._model = XGBRegressor(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._mode == "classifier":
            return self._model.predict_proba(X)[:, 1]
        else:
            if self._cdf_scale is None:
                raise ValueError(
                    "cdf_scale must be set to convert regressor margins to probabilities"
                )
            from scipy.stats import norm

            margins = self._model.predict(X)
            return norm.cdf(margins / self._cdf_scale)

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        if self._mode != "regressor":
            raise NotImplementedError("predict_margin is only available in regressor mode")
        return self._model.predict(X)

    @property
    def is_regression(self) -> bool:
        return self._mode == "regressor"

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path / "model.json"))
        meta = {
            "mode": self._mode,
            "cdf_scale": self._cdf_scale,
            "params": self.params,
        }
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> XGBoostModel:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        instance = cls.__new__(cls)
        instance.params = meta["params"]
        instance._mode = meta["mode"]
        instance._cdf_scale = meta["cdf_scale"]
        instance._fitted = True
        instance._build_model()
        instance._model.load_model(str(path / "model.json"))
        return instance
