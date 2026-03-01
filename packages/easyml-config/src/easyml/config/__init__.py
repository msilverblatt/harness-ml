"""Configuration resolution and validation for EasyML."""

from easyml.config.merge import deep_merge
from easyml.config.loader import load_config_file
from easyml.config.resolver import resolve_config

__all__ = ["deep_merge", "load_config_file", "resolve_config"]
