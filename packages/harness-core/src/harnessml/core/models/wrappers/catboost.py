"""CatBoost wrapper supporting both classifier and regressor modes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from harnessml.core.models.base import BaseModel


class CatBoostModel(BaseModel):
    """Wrapper around CatBoostClassifier / CatBoostRegressor.

    Parameters
    ----------
    params : dict | None
        Forwarded to the underlying CatBoost estimator.
    mode : str
        ``"classifier"`` or ``"regressor"``.
    """

    def __init__(self, params: dict | None = None, *, mode: str = "classifier"):
        super().__init__(params)
        if mode not in ("classifier", "regressor"):
            raise ValueError(f"mode must be 'classifier' or 'regressor', got {mode!r}")
        self._mode = mode
        self._build_model()

    def _catboost_cls(self):
        if self._mode == "classifier":
            from catboost import CatBoostClassifier
            return CatBoostClassifier
        else:
            from catboost import CatBoostRegressor
            return CatBoostRegressor

    def _build_model(self) -> None:
        params = dict(self.params)
        params.setdefault("allow_writing_files", False)
        self._model = self._catboost_cls()(**params)

    def fit(self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None, eval_set=None, **kwargs) -> None:
        params = dict(self.params)
        early_stopping = params.pop("early_stopping_rounds", None)
        params.setdefault("verbose", 0)
        params.setdefault("allow_writing_files", False)
        self._model = self._catboost_cls()(**params)

        fit_kwargs: dict = {}
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            fit_kwargs["eval_set"] = (X_val, y_val)
            if early_stopping:
                fit_kwargs["early_stopping_rounds"] = early_stopping
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        self._model.fit(X, y, **fit_kwargs)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._mode == "regressor":
            raise ValueError(
                "predict_proba not available in regressor mode. Use predict_margin."
            )
        probs = self._model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs

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
        self._model.save_model(str(path / "model.cbm"))
        meta = {"params": self.params, "mode": self._mode}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> CatBoostModel:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        mode = meta.get("mode", "classifier")
        instance = cls.__new__(cls)
        instance.params = meta["params"]
        instance._mode = mode
        instance._fitted = True
        instance._model = instance._catboost_cls()()
        instance._model.load_model(str(path / "model.cbm"))
        return instance
