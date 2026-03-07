"""Experiment logging utilities.

The primary logging implementation lives in ``ExperimentManager.log()``.
This module re-exports the key types for convenience.
"""

from harnessml.core.runner.experiment_manager import ExperimentError, ExperimentManager

__all__ = ["ExperimentError", "ExperimentManager"]
