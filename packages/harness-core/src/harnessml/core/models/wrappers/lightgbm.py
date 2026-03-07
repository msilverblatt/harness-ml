"""LightGBM wrapper supporting both classifier and regressor modes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from harnessml.core.models.base import BaseModel


class LightGBMModel(BaseModel):
    """Wrapper around LGBMClassifier / LGBMRegressor.

    Parameters
    ----------
    params : dict | None
        Forwarded to the underlying LightGBM estimator.
    mode : str
        ``"classifier"`` or ``"regressor"``.
    """

    def __init__(self, params: dict | None = None, *, mode: str = "classifier"):
        super().__init__(params)
        if mode not in ("classifier", "regressor"):
            raise ValueError(f"mode must be 'classifier' or 'regressor', got {mode!r}")
        self._mode = mode
        self._build_model()

    def _build_model(self) -> None:
        # Strip early_stopping_rounds from constructor — handled via callback
        self._early_stopping = self.params.get("early_stopping_rounds")
        clean_params = {
            k: v for k, v in self.params.items()
            if k != "early_stopping_rounds"
        }
        clean_params.setdefault("verbosity", -1)
        if self._mode == "classifier":
            from lightgbm import LGBMClassifier
            self._model = LGBMClassifier(**clean_params)
        else:
            from lightgbm import LGBMRegressor
            self._model = LGBMRegressor(**clean_params)

    def fit(self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None, **kwargs) -> None:
        import lightgbm as lgb

        if "eval_set" in kwargs and self._early_stopping:
            kwargs.setdefault("callbacks", [])
            kwargs["callbacks"].append(
                lgb.early_stopping(self._early_stopping, verbose=False)
            )
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        self._model.fit(X, y, **kwargs)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._mode == "classifier":
            probs = self._model.predict_proba(X)
            if probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        else:
            raise ValueError(
                "predict_proba not available in regressor mode. Use predict_margin."
            )

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        if self._mode != "regressor":
            raise NotImplementedError("predict_margin is only available in regressor mode")
        return self._model.predict(X)

    @property
    def is_regression(self) -> bool:
        return self._mode == "regressor"

    def save(self, path: Path) -> None:
        import json

        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")
        meta = {"mode": self._mode, "params": self.params}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        import json

        import joblib

        path = Path(path)
        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            mode = meta.get("mode", "classifier")
            params = meta.get("params", {})
        else:
            mode = "classifier"
            params = {}
        instance = cls.__new__(cls)
        instance.params = params
        instance._mode = mode
        instance._early_stopping = None
        instance._model = joblib.load(path / "model.joblib")
        instance._fitted = True
        return instance
