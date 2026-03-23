"""MLP (Multilayer Perceptron) model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness.ml.models.families.neural.base import NeuralBase
from harness.ml.models.protocol import FitResult

NAME = "mlp"


def _build_mlp_net(input_dim: int, hidden_dims: list[int], output_dim: int) -> Any:
    """Build a simple feedforward network using torch.nn."""
    import torch.nn as nn

    layers: list[nn.Module] = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(nn.ReLU())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class MLPModel(NeuralBase):
    name = "mlp"
    supports_tasks = ["binary", "multiclass", "regression"]
    requires_packages = ["torch"]

    def _build_network(self, input_dim: int, params: dict, task_type: str) -> Any:
        hidden_dims = params.get("hidden_dims", [64, 32])
        if task_type == "multiclass":
            output_dim = params.get("num_classes", 3)
        elif task_type == "regression":
            output_dim = 1
        else:
            output_dim = 1
        return _build_mlp_net(input_dim, hidden_dims, output_dim)

    def _prepare_targets(
        self, y: pd.Series, task_type: str, device: str
    ) -> tuple[Any, Any]:
        import torch
        import torch.nn as nn

        if task_type == "binary":
            y_tensor = torch.tensor(y.values.astype(np.float32), device=device).unsqueeze(1)
            loss_fn = nn.BCEWithLogitsLoss()
        elif task_type == "regression":
            y_tensor = torch.tensor(y.values.astype(np.float32), device=device).unsqueeze(1)
            loss_fn = nn.MSELoss()
        else:
            # multiclass
            y_tensor = torch.tensor(y.values.astype(np.int64), device=device)
            loss_fn = nn.CrossEntropyLoss()
        return y_tensor, loss_fn

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for neural network predictions."
            ) from exc

        from harness.ml.models.families.neural.base import _get_device

        device = _get_device()
        model = model.to(device)
        model.eval()

        X_np = X.values.astype(np.float32)
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            out = model(X_tensor)

        task_type = getattr(model, "_task_type", "binary")
        if task_type == "binary":
            probs = torch.sigmoid(out).cpu().numpy().squeeze()
            return probs
        elif task_type == "regression":
            return out.cpu().numpy().squeeze()
        else:
            probs = torch.softmax(out, dim=1).cpu().numpy()
            return probs

    def default_params(self, task_type: str) -> dict:
        base: dict = {
            "_task_type": task_type,
            "hidden_dims": [64, 32],
            "learning_rate": 1e-3,
            "epochs": 50,
            "batch_size": 64,
        }
        if task_type == "multiclass":
            base["num_classes"] = 3
        return base

    def param_schema(self) -> dict:
        return {
            "hidden_dims": {
                "type": "list",
                "default": [64, 32],
            },
            "learning_rate": {
                "type": "float",
                "default": 1e-3,
                "min": 1e-5,
                "max": 1e-1,
            },
            "epochs": {
                "type": "int",
                "default": 50,
                "min": 1,
                "max": 1000,
            },
            "batch_size": {
                "type": "int",
                "default": 64,
                "min": 8,
                "max": 512,
            },
        }
