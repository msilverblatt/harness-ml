"""Backward-compat shim — re-exports from easyml.core.config."""

from easyml.core.config import deep_merge, load_config_file, resolve_config  # noqa: F401

__all__ = ["deep_merge", "load_config_file", "resolve_config"]
