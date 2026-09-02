"""Expression validator — static validation without execution."""
from __future__ import annotations

import re
from difflib import get_close_matches

from harness.data.expressions.registry import FunctionRegistry
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of expression validation."""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    suggestion: str = ""


# Tokens that are operators/literals, not identifiers
_OPERATORS = {'+', '-', '*', '/', '%', '>', '<', '>=', '<=', '==', '!=', '&', '|', '~', ','}
_KEYWORDS = {'and', 'or', 'not', 'True', 'False', 'None', 'in', 'is'}


def _tokenize(expression: str) -> list[str]:
    """Extract identifier tokens from an expression."""
    # Split on operators, parens, whitespace, commas
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', expression)
    return tokens


def _is_numeric_adjacent(expression: str, token: str) -> bool:
    """Check if a token might be part of a numeric literal (e.g., 1e10)."""
    return False


class ExpressionValidator:
    """Validate expressions against a schema without executing them."""

    def __init__(self, registry: FunctionRegistry | None = None):
        if registry is None:
            registry = FunctionRegistry()
            registry.load_defaults()
        self._registry = registry

    def validate(self, expression: str, schema: dict) -> ValidationResult:
        """Validate an expression against a schema.

        Args:
            expression: The expression string to validate.
            schema: Dict with 'columns' (list[str]) and optionally 'column_types'.

        Returns:
            ValidationResult with is_valid, errors, and suggestion.
        """
        columns = set(schema.get("columns", []))
        known_functions = set(self._registry.list_functions())

        tokens = _tokenize(expression)
        errors = []
        suggestions = []

        # Classify each token
        for token in tokens:
            if token in _KEYWORDS:
                continue
            if token in known_functions:
                continue
            if token in columns:
                continue
            # Check if it looks like a function call
            # Find if token is immediately followed by '(' in the expression
            func_pattern = re.compile(rf'\b{re.escape(token)}\s*\(')
            if func_pattern.search(expression):
                # It's being used as a function but isn't registered
                errors.append(f"Unknown function: '{token}'")
            else:
                # It's being used as a column reference but doesn't exist
                close = get_close_matches(token, list(columns), n=3, cutoff=0.4)
                errors.append(f"Unknown column: '{token}'")
                if close:
                    suggestions.extend(close)
                else:
                    # No fuzzy match — suggest all available columns
                    suggestions.extend(columns)

        suggestion = ""
        if suggestions:
            suggestion = "Did you mean: " + ", ".join(sorted(set(suggestions))) + "?"

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            suggestion=suggestion,
        )
