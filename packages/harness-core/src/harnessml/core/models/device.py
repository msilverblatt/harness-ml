"""GPU/device detection for ML model training."""
from __future__ import annotations

import os


def detect_device() -> str:
    """Detect the best available device for PyTorch training.

    Checks (in order):
    1. HARNESS_DEVICE environment variable (user override)
    2. CUDA availability
    3. MPS availability (Apple Silicon)
    4. Falls back to CPU

    Returns
    -------
    str
        One of "cuda", "mps", or "cpu".
    """
    env_device = os.environ.get("HARNESS_DEVICE")
    if env_device:
        return env_device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
