"""TabNet classifier wrapper with multi-seed averaging."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from easyml.core.models.base import BaseModel

# Keys that belong to TabNetClassifier.fit(), not __init__()
_FIT_KEYS = frozenset({
    "max_epochs", "patience", "batch_size", "virtual_batch_size",
    "num_workers", "drop_last",
})


class TabNetModel(BaseModel):
    """Multi-seed averaged TabNet classifier.

    Parameters
    ----------
    params : dict | None
        Forwarded to TabNetClassifier.  Constructor keys (``n_d``, ``n_a``,
        ``n_steps``, ``verbose``, etc.) go to ``__init__``; training keys
        (``max_epochs``, ``patience``, ``batch_size``) go to ``fit()``.
    n_seeds : int
        Number of models to train with different random seeds.
    """

    def __init__(self, params: dict | None = None, *, n_seeds: int = 1):
        super().__init__(params)
        self._n_seeds = n_seeds
        self._models: list = []

    def _split_params(self) -> tuple[dict, dict]:
        """Split self.params into (init_kwargs, fit_kwargs)."""
        init_kwargs: dict = {}
        fit_kwargs: dict = {}
        for k, v in self.params.items():
            if k in _FIT_KEYS:
                fit_kwargs[k] = v
            else:
                init_kwargs[k] = v
        return init_kwargs, fit_kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from pytorch_tabnet.tab_model import TabNetClassifier

        init_kwargs, fit_kwargs = self._split_params()
        self._models = []
        for seed in range(self._n_seeds):
            model = TabNetClassifier(**init_kwargs, seed=seed)
            model.fit(X, y, **fit_kwargs)
            self._models.append(model)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for model in self._models:
            # predict_proba returns (n, 2) for binary classification
            p = model.predict_proba(X)[:, 1]
            preds.append(p)
        return np.mean(preds, axis=0)

    @property
    def is_regression(self) -> bool:
        return False

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "n_seeds": self._n_seeds,
            "params": self.params,
        }
        (path / "meta.json").write_text(json.dumps(meta))
        for i, model in enumerate(self._models):
            model.save_model(str(path / f"tabnet_{i}"))

    @classmethod
    def load(cls, path: Path) -> TabNetModel:
        from pytorch_tabnet.tab_model import TabNetClassifier

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        instance = cls.__new__(cls)
        instance.params = meta["params"]
        instance._n_seeds = meta["n_seeds"]
        instance._fitted = True

        instance._models = []
        for i in range(instance._n_seeds):
            model = TabNetClassifier()
            model.load_model(str(path / f"tabnet_{i}.zip"))
            instance._models.append(model)

        return instance
