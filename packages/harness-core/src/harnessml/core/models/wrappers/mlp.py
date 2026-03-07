"""PyTorch MLP wrapper with multi-seed averaging."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harnessml.core.models.base import BaseModel


def _build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
    mode: str,
    batch_norm: bool = False,
):
    """Build a simple MLP as nn.Sequential."""
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
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
    normalize : bool
        If True, z-score standardize features during fit and apply the same
        transformation during prediction.
    batch_norm : bool
        If True, insert BatchNorm1d layers after each Linear layer.
    weight_decay : float
        L2 regularization weight for the Adam optimizer.
    early_stopping_rounds : int | None
        If set (and eval_set is passed to fit), stop training after this many
        epochs without improvement on the validation set.
    seed_stride : int
        Stride between random seeds.  Seed i uses ``i * seed_stride``.
    """

    def __init__(
        self,
        params: dict | None = None,
        *,
        mode: str = "classifier",
        n_seeds: int = 1,
        cdf_scale: float | None = None,
        normalize: bool = False,
        batch_norm: bool = False,
        weight_decay: float = 0.0,
        early_stopping_rounds: int | None = None,
        seed_stride: int = 1,
        seed: int = 0,
    ):
        super().__init__(params)
        if mode not in ("classifier", "regressor"):
            raise ValueError(f"mode must be 'classifier' or 'regressor', got {mode!r}")
        self._mode = mode
        self._n_seeds = n_seeds
        self._cdf_scale = cdf_scale
        self._normalize = normalize
        self._batch_norm = batch_norm
        self._weight_decay = weight_decay
        self._early_stopping_rounds = early_stopping_rounds
        self._seed_stride = seed_stride
        self._base_seed = seed
        self._models: list = []  # populated by fit()
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    def _normalize_features(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Z-score standardize features. If fit=True, compute and store stats."""
        if not self._normalize:
            return X
        if fit:
            self._feature_means = np.nanmean(X, axis=0)
            self._feature_stds = np.nanstd(X, axis=0)
            # Avoid division by zero for constant features
            self._feature_stds[self._feature_stds < 1e-8] = 1.0
        result = (X - self._feature_means) / self._feature_stds
        return np.nan_to_num(result, nan=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray, *, eval_set=None) -> None:
        import torch
        import torch.nn as nn

        hidden_dims = self.params.get("hidden_layers", self.params.get("hidden_dims", [128, 64]))
        dropout = self.params.get("dropout", 0.0)
        lr = self.params.get("learning_rate", self.params.get("lr", 0.001))
        epochs = self.params.get("epochs", 10)
        batch_size = self.params.get("batch_size", 32)

        X = self._normalize_features(X, fit=True)

        X_t = torch.tensor(X, dtype=torch.float32)
        if self._mode == "classifier":
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            criterion = nn.MSELoss()

        # Prepare validation data if provided
        X_val_t = None
        y_val_t = None
        if eval_set is not None:
            # Accept both (X, y) tuple and [(X, y)] list-of-tuples format
            if isinstance(eval_set, list):
                X_val, y_val = eval_set[0]
            else:
                X_val, y_val = eval_set
            X_val = self._normalize_features(X_val)
            X_val_t = torch.tensor(X_val, dtype=torch.float32)
            y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        self._models = []
        for i in range(self._n_seeds):
            seed = self._base_seed + i * self._seed_stride
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = _build_mlp(
                X.shape[1], hidden_dims, dropout, self._mode,
                batch_norm=self._batch_norm,
            )
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=self._weight_decay,
            )

            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            model.train()
            n = len(X_t)
            for _epoch in range(epochs):
                perm = torch.randperm(n)
                for start in range(0, n, batch_size):
                    idx = perm[start : start + batch_size]
                    out = model(X_t[idx])
                    loss = criterion(out, y_t[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Early stopping check
                if (
                    self._early_stopping_rounds is not None
                    and X_val_t is not None
                ):
                    model.eval()
                    with torch.no_grad():
                        val_out = model(X_val_t)
                        val_loss = criterion(val_out, y_val_t).item()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                    if patience_counter >= self._early_stopping_rounds:
                        break
                    model.train()

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            self._models.append(model)

        self._input_dim = X.shape[1]
        self._fitted = True

    def _raw_predictions(self, X: np.ndarray) -> np.ndarray:
        """Average raw outputs across seeds."""
        import torch

        X = self._normalize_features(X)
        X_t = torch.tensor(X, dtype=torch.float32)
        preds = []
        with torch.no_grad():
            for model in self._models:
                out = model(X_t).squeeze(1).numpy()
                preds.append(out)
        return np.mean(preds, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._mode == "classifier":
            raw = self._raw_predictions(X)
            # Raw outputs are logits; apply sigmoid
            return 1.0 / (1.0 + np.exp(-raw))
        else:
            if self._cdf_scale is None:
                raise ValueError(
                    "cdf_scale must be set to convert regressor margins to probabilities"
                )
            # Per-seed CDF conversion then average (Jensen's inequality matters)
            return self._per_seed_proba(X, self._cdf_scale)

    def _per_seed_proba(self, X: np.ndarray, scale: float) -> np.ndarray:
        """Convert margins to probabilities per seed, then average.

        Uses logistic sigmoid: P = 1 / (1 + exp(-margin/scale)).
        Averaging probabilities instead of margins avoids Jensen's
        inequality bias for multi-seed models.
        """
        import torch

        X = self._normalize_features(X)
        X_t = torch.tensor(X, dtype=torch.float32)
        all_probs = []
        with torch.no_grad():
            for model in self._models:
                margins = model(X_t).squeeze(1).numpy()
                z = margins / scale
                # Numerically stable logistic sigmoid
                probs = np.where(
                    z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)),
                )
                all_probs.append(probs)
        return np.mean(all_probs, axis=0)

    def set_cdf_scale(self, scale: float) -> None:
        """Set CDF scale for regressor probability conversion."""
        self._cdf_scale = scale

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
            "normalize": self._normalize,
            "batch_norm": self._batch_norm,
            "weight_decay": self._weight_decay,
            "early_stopping_rounds": self._early_stopping_rounds,
            "seed_stride": self._seed_stride,
            "base_seed": self._base_seed,
        }
        if self._normalize and self._feature_means is not None:
            meta["feature_means"] = self._feature_means.tolist()
            meta["feature_stds"] = self._feature_stds.tolist()
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
        instance._normalize = meta.get("normalize", False)
        instance._batch_norm = meta.get("batch_norm", False)
        instance._weight_decay = meta.get("weight_decay", 0.0)
        instance._early_stopping_rounds = meta.get("early_stopping_rounds")
        instance._seed_stride = meta.get("seed_stride", 1)
        instance._base_seed = meta.get("base_seed", 0)
        instance._fitted = True

        if instance._normalize and "feature_means" in meta:
            instance._feature_means = np.array(meta["feature_means"], dtype=np.float64)
            instance._feature_stds = np.array(meta["feature_stds"], dtype=np.float64)
        else:
            instance._feature_means = None
            instance._feature_stds = None

        hidden_dims = instance.params.get("hidden_layers", instance.params.get("hidden_dims", [128, 64]))
        dropout = instance.params.get("dropout", 0.0)

        instance._models = []
        for i in range(instance._n_seeds):
            model = _build_mlp(
                instance._input_dim, hidden_dims, dropout, instance._mode,
                batch_norm=instance._batch_norm,
            )
            model.load_state_dict(torch.load(path / f"model_{i}.pt", weights_only=True))
            model.eval()
            instance._models.append(model)

        return instance
