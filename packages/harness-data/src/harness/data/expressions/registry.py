"""Function registry for the expression engine."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class FunctionInfo:
    """Metadata about a registered function."""
    name: str
    fn: Callable
    description: str


class FunctionRegistry:
    """Registry of functions available to the expression engine."""

    def __init__(self):
        self._functions: dict[str, FunctionInfo] = {}

    def register(self, name: str, fn: Callable, *, description: str = "") -> None:
        """Register a function by name."""
        self._functions[name] = FunctionInfo(name=name, fn=fn, description=description)

    def load_defaults(self) -> None:
        """Load all built-in function modules."""
        from harness.data.expressions.functions import comparison, math, null, stats

        for module in (math, stats, comparison, null):
            for name, info in module.FUNCTIONS.items():
                self.register(name, info["fn"], description=info["description"])

    def list_functions(self) -> list[str]:
        """Return names of all registered functions."""
        return list(self._functions.keys())

    def get(self, name: str) -> FunctionInfo | None:
        """Get function info by name, or None if not found."""
        return self._functions.get(name)
