"""TabNet classifier wrapper with multi-seed averaging."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harnessml.core.models.base import BaseModel

# Keys that belong to TabNetClassifier.fit(), not __init__()
_FIT_KEYS = frozenset({
    "max_epochs", "patience", "batch_size", "virtual_batch_size",
    "num_workers", "drop_last",
})

# Keys managed by this wrapper, not passed to TabNet
_WRAPPER_KEYS = frozenset({
    "normalize", "val_fraction", "learning_rate",
    "scheduler_step_size", "scheduler_gamma", "seed_stride",
    "seed", "n_seeds",
})

# Aliases: user-facing name -> TabNet parameter name
_PARAM_RENAMES = {
    "relaxation_factor": "gamma",
}


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
    normalize : bool
        If True, standardize features before training and prediction.
    val_fraction : float | None
        If set, auto-split this fraction of training data for validation.
    learning_rate : float | None
        If set, inject into optimizer_params as ``lr``.
    scheduler_step_size : int | None
        Step size for StepLR scheduler (requires scheduler_gamma).
    scheduler_gamma : float | None
        Gamma for StepLR scheduler (requires scheduler_step_size).
    seed_stride : int
        Stride between seed values for multi-seed training.
    """

    def __init__(
        self,
        params: dict | None = None,
        *,
        n_seeds: int = 1,
        normalize: bool = False,
        val_fraction: float | None = None,
        learning_rate: float | None = None,
        scheduler_step_size: int | None = None,
        scheduler_gamma: float | None = None,
        seed_stride: int = 1,
    ):
        super().__init__(params)
        self._n_seeds = n_seeds
        self._models: list = []
        self._normalize = normalize
        self._val_fraction = val_fraction
        self._learning_rate = learning_rate
        self._scheduler_step_size = scheduler_step_size
        self._scheduler_gamma = scheduler_gamma
        self._seed_stride = seed_stride
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    def _split_params(self) -> tuple[dict, dict]:
        """Split self.params into (init_kwargs, fit_kwargs).

        Wrapper-only keys are excluded. Renames are applied.
        If learning_rate is set, it is injected into optimizer_params.
        """
        init_kwargs: dict = {}
        fit_kwargs: dict = {}
        for k, v in self.params.items():
            if k in _WRAPPER_KEYS:
                continue
            out_key = _PARAM_RENAMES.get(k, k)
            if out_key in _FIT_KEYS:
                fit_kwargs[out_key] = v
            else:
                init_kwargs[out_key] = v

        # Inject learning_rate into optimizer_params
        if self._learning_rate is not None:
            opt_params = dict(init_kwargs.get("optimizer_params", {}))
            opt_params["lr"] = self._learning_rate
            init_kwargs["optimizer_params"] = opt_params

        return init_kwargs, fit_kwargs

    def fit(self, X: np.ndarray, y: np.ndarray, *, eval_set=None) -> None:
        from pytorch_tabnet.tab_model import TabNetClassifier

        # Normalization
        if self._normalize:
            self._feature_means = X.mean(axis=0)
            self._feature_stds = X.std(axis=0)
            self._feature_stds[self._feature_stds == 0] = 1.0
            X = (X - self._feature_means) / self._feature_stds

        init_kwargs, fit_kwargs = self._split_params()

        # Scheduler params (TabNet expects these in __init__, not fit)
        if self._scheduler_step_size is not None and self._scheduler_gamma is not None:
            init_kwargs["scheduler_params"] = {
                "step_size": self._scheduler_step_size,
                "gamma": self._scheduler_gamma,
            }
            init_kwargs["scheduler_fn"] = __import__("torch.optim.lr_scheduler", fromlist=["StepLR"]).StepLR

        # Validation split
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        elif self._val_fraction is not None:
            n_val = int(len(X) * self._val_fraction)
            X_train, X_val = X[:-n_val], X[-n_val:]
            y_train, y_val = y[:-n_val], y[-n_val:]
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            X, y = X_train, y_train

        self._models = []
        for seed_idx in range(self._n_seeds):
            seed_val = seed_idx * self._seed_stride
            model = TabNetClassifier(**init_kwargs, seed=seed_val)
            model.fit(X, y, **fit_kwargs)
            self._models.append(model)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._normalize and self._feature_means is not None:
            X = (X - self._feature_means) / self._feature_stds
        preds = []
        for model in self._models:
            p = model.predict_proba(X)
            if p.shape[1] == 2:
                p = p[:, 1]
            preds.append(p)
        return np.mean(preds, axis=0)

    @property
    def is_regression(self) -> bool:
        return False

    def _get_meta(self) -> dict:
        """Build metadata dict for persistence."""
        meta = {
            "n_seeds": self._n_seeds,
            "params": self.params,
            "normalize": self._normalize,
            "val_fraction": self._val_fraction,
            "learning_rate": self._learning_rate,
            "scheduler_step_size": self._scheduler_step_size,
            "scheduler_gamma": self._scheduler_gamma,
            "seed_stride": self._seed_stride,
        }
        if self._feature_means is not None:
            meta["feature_means"] = self._feature_means.tolist()
        if self._feature_stds is not None:
            meta["feature_stds"] = self._feature_stds.tolist()
        return meta

    def _load_meta(self, meta: dict) -> None:
        """Restore wrapper state from metadata dict."""
        self.params = meta["params"]
        self._n_seeds = meta["n_seeds"]
        self._normalize = meta.get("normalize", False)
        self._val_fraction = meta.get("val_fraction")
        self._learning_rate = meta.get("learning_rate")
        self._scheduler_step_size = meta.get("scheduler_step_size")
        self._scheduler_gamma = meta.get("scheduler_gamma")
        self._seed_stride = meta.get("seed_stride", 1)
        if "feature_means" in meta:
            self._feature_means = np.array(meta["feature_means"])
        else:
            self._feature_means = None
        if "feature_stds" in meta:
            self._feature_stds = np.array(meta["feature_stds"])
        else:
            self._feature_stds = None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(json.dumps(self._get_meta()))
        for i, model in enumerate(self._models):
            model.save_model(str(path / f"tabnet_{i}"))

    @classmethod
    def load(cls, path: Path) -> TabNetModel:
        from pytorch_tabnet.tab_model import TabNetClassifier

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        instance = cls.__new__(cls)
        instance._models = []
        instance._fitted = True
        instance._load_meta(meta)

        for i in range(instance._n_seeds):
            model = TabNetClassifier()
            model.load_model(str(path / f"tabnet_{i}.zip"))
            instance._models.append(model)

        return instance
