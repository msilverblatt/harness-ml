"""Expression engine — safe formula evaluation over DataFrames."""
from __future__ import annotations

import re
import pandas as pd
from harness.data.expressions.registry import FunctionRegistry


# Patterns that indicate unsafe operations
_DANGEROUS_PATTERNS = [
    r'__\w+__',        # dunder attributes
    r'\bimport\b',     # import statements
    r'\beval\b',       # eval calls
    r'\bexec\b',       # exec calls
    r'\bcompile\b',    # compile calls
    r'\bgetattr\b',    # getattr calls
    r'\bsetattr\b',    # setattr calls
    r'\bdelattr\b',    # delattr calls
    r'\bglobals\b',    # globals access
    r'\blocals\b',     # locals access
    r'\bopen\b',       # file open
]


class ExpressionEngine:
    """Evaluate expressions safely against a DataFrame."""

    def __init__(self, registry: FunctionRegistry | None = None):
        if registry is None:
            registry = FunctionRegistry()
            registry.load_defaults()
        self._registry = registry

    def evaluate(self, df: pd.DataFrame, expression: str) -> pd.Series:
        """Evaluate an expression against a DataFrame and return a Series.

        Uses pd.eval with engine='python' and a restricted namespace
        containing only registered functions and DataFrame columns.
        """
        self._check_safety(expression)

        # Build namespace: registered functions + DataFrame columns
        namespace = {}
        for name in self._registry.list_functions():
            info = self._registry.get(name)
            if info is not None:
                namespace[name] = info.fn

        for col in df.columns:
            namespace[col] = df[col]

        try:
            result = pd.eval(expression, local_dict=namespace, engine="python")
        except Exception as exc:
            raise ValueError(f"Expression evaluation failed: {expression!r} -- {exc}") from exc

        if isinstance(result, pd.Series):
            return result
        # Scalar result: broadcast to match df length
        return pd.Series(result, index=df.index)

    def _check_safety(self, expression: str) -> None:
        """Reject expressions that contain dangerous patterns."""
        for pattern in _DANGEROUS_PATTERNS:
            if re.search(pattern, expression):
                raise ValueError(
                    f"Expression contains disallowed pattern: {expression!r}"
                )
