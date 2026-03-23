"""Task type registry with lazy imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.ml.tasks.protocol import TaskType


_REGISTRY: dict[str, str] = {
    "binary": "harness.ml.tasks.binary.task",
    "regression": "harness.ml.tasks.regression.task",
    "multiclass": "harness.ml.tasks.multiclass.task",
}

_TASK_CLASS_NAMES: dict[str, str] = {
    "binary": "BinaryTask",
    "regression": "RegressionTask",
    "multiclass": "MulticlassTask",
}


class TaskRegistry:
    """Registry for task types with lazy module loading."""

    _cache: dict[str, TaskType] = {}

    @classmethod
    def get(cls, name: str) -> TaskType:
        """Get a task type instance by name.

        Parameters
        ----------
        name : str
            Task type name (case-insensitive).

        Returns
        -------
        TaskType
            The task type instance.

        Raises
        ------
        KeyError
            If the task type name is not registered.
        """
        key = name.lower()
        if key not in _REGISTRY:
            raise KeyError(
                f"Unknown task type: {name!r}. "
                f"Available: {list(_REGISTRY.keys())}"
            )

        if key not in cls._cache:
            import importlib

            module = importlib.import_module(_REGISTRY[key])
            task_class = getattr(module, _TASK_CLASS_NAMES[key])
            cls._cache[key] = task_class()

        return cls._cache[key]

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered task type names."""
        return list(_REGISTRY.keys())
