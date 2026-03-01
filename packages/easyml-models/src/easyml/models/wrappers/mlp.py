"""PyTorch MLP wrapper with multi-seed averaging."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from easyml.models.base import BaseModel


def _build_mlp(input_dim: int, hidden_dims: list[int], dropout: float, mode: str):
    """Build a simple MLP as nn.Sequential."""
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class MLPModel(BaseModel):
    """Multi-seed averaged PyTorch MLP supporting classifier and regressor modes.

    Parameters
    ----------
    params : dict | None
        ``hidden_dims`` (list[int]), ``dropout`` (float), ``lr`` (float),
        ``epochs`` (int), ``batch_size`` (int, default 32).
    mode : str
        ``"classifier"`` (BCEWithLogitsLoss) or ``"regressor"`` (MSELoss).
    n_seeds : int
        Number of models to train with different random seeds.  Predictions
        are averaged across all seeds.
    cdf_scale : float | None
        For regressor mode: scale for normal CDF margin-to-probability
        conversion.  Required when calling ``predict_proba`` on a regressor.
    """

    def __init__(
        self,
        params: dict | None = None,
        *,
        mode: str = "classifier",
        n_seeds: int = 1,
        cdf_scale: float | None = None,
    ):
        super().__init__(params)
        if mode not in ("classifier", "regressor"):
            raise ValueError(f"mode must be 'classifier' or 'regressor', got {mode!r}")
        self._mode = mode
        self._n_seeds = n_seeds
        self._cdf_scale = cdf_scale
        self._models: list = []  # populated by fit()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        import torch.nn as nn

        hidden_dims = self.params.get("hidden_dims", [128, 64])
        dropout = self.params.get("dropout", 0.0)
        lr = self.params.get("lr", 0.001)
        epochs = self.params.get("epochs", 10)
        batch_size = self.params.get("batch_size", 32)

        X_t = torch.tensor(X, dtype=torch.float32)
        if self._mode == "classifier":
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            criterion = nn.MSELoss()

        self._models = []
        for seed in range(self._n_seeds):
            torch.manual_seed(seed)
            model = _build_mlp(X.shape[1], hidden_dims, dropout, self._mode)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            model.train()
            n = len(X_t)
            for _ in range(epochs):
                perm = torch.randperm(n)
                for start in range(0, n, batch_size):
                    idx = perm[start : start + batch_size]
                    out = model(X_t[idx])
                    loss = criterion(out, y_t[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            self._models.append(model)

        self._input_dim = X.shape[1]
        self._fitted = True

    def _raw_predictions(self, X: np.ndarray) -> np.ndarray:
        """Average raw outputs across seeds."""
        import torch

        X_t = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for model in self._models:
                out = model(X_t).squeeze(1).numpy()
                preds.append(out)
        return np.mean(preds, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._raw_predictions(X)
        if self._mode == "classifier":
            # Raw outputs are logits; apply sigmoid
            return 1.0 / (1.0 + np.exp(-raw))
        else:
            if self._cdf_scale is None:
                raise ValueError(
                    "cdf_scale must be set to convert regressor margins to probabilities"
                )
            from scipy.stats import norm

            return norm.cdf(raw / self._cdf_scale)

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        if self._mode != "regressor":
            raise NotImplementedError("predict_margin is only available in regressor mode")
        return self._raw_predictions(X)

    @property
    def is_regression(self) -> bool:
        return self._mode == "regressor"

    def save(self, path: Path) -> None:
        import torch

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "mode": self._mode,
            "n_seeds": self._n_seeds,
            "cdf_scale": self._cdf_scale,
            "params": self.params,
            "input_dim": self._input_dim,
        }
        (path / "meta.json").write_text(json.dumps(meta))
        for i, model in enumerate(self._models):
            torch.save(model.state_dict(), path / f"model_{i}.pt")

    @classmethod
    def load(cls, path: Path) -> MLPModel:
        import torch

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        instance = cls.__new__(cls)
        instance.params = meta["params"]
        instance._mode = meta["mode"]
        instance._n_seeds = meta["n_seeds"]
        instance._cdf_scale = meta["cdf_scale"]
        instance._input_dim = meta["input_dim"]
        instance._fitted = True

        hidden_dims = instance.params.get("hidden_dims", [128, 64])
        dropout = instance.params.get("dropout", 0.0)

        instance._models = []
        for i in range(instance._n_seeds):
            model = _build_mlp(instance._input_dim, hidden_dims, dropout, instance._mode)
            model.load_state_dict(torch.load(path / f"model_{i}.pt", weights_only=True))
            model.eval()
            instance._models.append(model)

        return instance
