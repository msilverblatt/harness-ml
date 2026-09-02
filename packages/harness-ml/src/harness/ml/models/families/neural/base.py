"""Base class for neural network model family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from harness.ml.models.protocol import FitResult


def _get_device() -> str:
    """Detect the best available device: cuda > mps > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"


class NeuralBase(ABC):
    """Shared base for neural network model wrappers."""

    @abstractmethod
    def _build_network(self, input_dim: int, params: dict, task_type: str) -> Any:
        """Build and return the torch network."""
        ...

    @abstractmethod
    def _prepare_targets(
        self, y: pd.Series, task_type: str
    ) -> tuple[Any, Any]:
        """Return (y_tensor, loss_fn). Requires torch to be imported by caller."""
        ...

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        params: dict,
    ) -> FitResult:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required to train neural network models. "
                "Install it with: uv pip install torch"
            ) from exc

        params = dict(params)
        task_type = params.pop("_task_type", self.supports_tasks[0])
        learning_rate = params.pop("learning_rate", 1e-3)
        epochs = params.pop("epochs", 50)
        batch_size = params.pop("batch_size", 64)

        device = _get_device()

        X_np = X_train.values.astype(np.float32)
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)

        net = self._build_network(X_np.shape[1], params, task_type)
        net = net.to(device)

        y_tensor, loss_fn = self._prepare_targets(y_train, task_type, device)

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

        n = X_tensor.shape[0]
        net.train()
        for _epoch in range(epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]

                optimizer.zero_grad()
                out = net(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()

        net.eval()

        # Store task_type on the network for use during predict
        net._task_type = task_type  # type: ignore[attr-defined]

        return FitResult(model=net)

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for neural network predictions."
            ) from exc

        device = _get_device()
        model = model.to(device)
        model.eval()

        X_np = X.values.astype(np.float32)
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            out = model(X_tensor)

        task_type = getattr(model, "_task_type", "binary")
        if task_type == "binary" or task_type == "regression":
            return out.cpu().numpy().squeeze()
        else:
            # multiclass — return probability matrix
            probs = torch.softmax(out, dim=1)
            return probs.cpu().numpy()

    def save(self, model: Any, path: Path) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to save neural network models.") from exc
        torch.save(model, path)

    def load(self, path: Path) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("torch is required to load neural network models.") from exc
        return torch.load(path, weights_only=False)

    def supports_multi_seed(self) -> bool:
        return False
