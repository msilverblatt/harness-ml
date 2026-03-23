"""Transform engine — auto-discovers and runs transform steps."""
from __future__ import annotations

import importlib
import pkgutil
import pandas as pd
from typing import Any

from harness.data.transforms.protocol import StepConfig
import harness.data.transforms.steps as steps_package


class TransformEngine:
    """Discovers step modules and runs transform pipelines."""

    def __init__(self):
        self._steps: dict[str, Any] = {}
        self._discover_steps()

    def _discover_steps(self) -> None:
        """Auto-discover step modules from the steps/ package."""
        for importer, modname, ispkg in pkgutil.iter_modules(steps_package.__path__):
            module = importlib.import_module(f"harness.data.transforms.steps.{modname}")
            if hasattr(module, "NAME") and hasattr(module, "step"):
                self._steps[module.NAME] = module.step

    def available_steps(self) -> list[str]:
        """Return names of all registered steps."""
        return list(self._steps.keys())

    def apply_step(
        self,
        df: pd.DataFrame,
        config: StepConfig | None = None,
        resolver: Any = None,
        *,
        step_type: str | None = None,
        params: dict | None = None,
    ) -> pd.DataFrame:
        """Apply a single step to a DataFrame.

        Can be called with a StepConfig or with step_type + params kwargs.
        """
        if config is not None:
            op = config.op
            step_params = config.params
        elif step_type is not None:
            op = step_type
            step_params = params or {}
        else:
            raise ValueError("Must provide either config or step_type")

        if op not in self._steps:
            raise ValueError(f"Unknown step: '{op}'. Available: {self.available_steps()}")

        return self._steps[op](df, step_params)

    def run_pipeline(
        self,
        df: pd.DataFrame,
        steps: list[StepConfig],
        resolver: Any = None,
    ) -> pd.DataFrame:
        """Run a sequence of steps, chaining output to input."""
        result = df
        for step_config in steps:
            result = self.apply_step(result, step_config, resolver)
        return result
