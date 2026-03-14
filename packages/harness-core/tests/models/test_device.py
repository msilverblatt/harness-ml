"""Tests for GPU/device detection."""

from harnessml.core.models.device import detect_device


def test_detect_device_returns_valid():
    """detect_device should always return a valid device string."""
    device = detect_device()
    assert device in ("cpu", "cuda", "mps")


def test_detect_device_respects_env(monkeypatch):
    """HARNESS_DEVICE env var should override auto-detection."""
    monkeypatch.setenv("HARNESS_DEVICE", "cpu")
    assert detect_device() == "cpu"


def test_detect_device_env_overrides_gpu(monkeypatch):
    """Even if GPU is available, env var takes precedence."""
    monkeypatch.setenv("HARNESS_DEVICE", "cpu")
    device = detect_device()
    assert device == "cpu"


def test_detect_device_custom_env(monkeypatch):
    """Custom device strings from env should be passed through."""
    monkeypatch.setenv("HARNESS_DEVICE", "cuda:1")
    assert detect_device() == "cuda:1"


def test_detect_device_no_torch(monkeypatch):
    """If torch is not importable, should fall back to cpu."""
    monkeypatch.delenv("HARNESS_DEVICE", raising=False)
    import sys

    # Temporarily hide torch
    real_torch = sys.modules.get("torch")
    sys.modules["torch"] = None  # type: ignore[assignment]
    try:
        # Need to reload since detect_device imports torch inside the function
        assert detect_device() == "cpu"
    finally:
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            sys.modules.pop("torch", None)
