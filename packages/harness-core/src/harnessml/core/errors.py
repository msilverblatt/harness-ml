"""Structured error codes for HarnessML.

Provides consistent, machine-parseable error codes alongside human-readable
messages.  Used by config_writer, validation, and MCP handlers.
"""
from __future__ import annotations

from enum import Enum


class ErrorCode(Enum):
    """Enumeration of all structured error codes."""

    # Formula / feature errors
    FORMULA_SYNTAX_ERROR = "FORMULA_SYNTAX_ERROR"
    FEATURE_NOT_FOUND = "FEATURE_NOT_FOUND"
    FEATURE_DUPLICATE = "FEATURE_DUPLICATE"

    # Model errors
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_ALREADY_EXISTS = "MODEL_ALREADY_EXISTS"
    MODEL_TYPE_MISSING = "MODEL_TYPE_MISSING"

    # Config errors
    CONFIG_NOT_FOUND = "CONFIG_NOT_FOUND"
    CONFIG_INVALID = "CONFIG_INVALID"
    CONFIG_ALREADY_EXISTS = "CONFIG_ALREADY_EXISTS"

    # Data errors
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    COLUMN_NOT_FOUND = "COLUMN_NOT_FOUND"
    TARGET_COLUMN_MISSING = "TARGET_COLUMN_MISSING"

    # Validation errors
    FOLD_COLUMN_MISSING = "FOLD_COLUMN_MISSING"
    CLASS_IMBALANCE = "CLASS_IMBALANCE"
    HIGH_MISSING_RATE = "HIGH_MISSING_RATE"
    NO_ACTIVE_MODELS = "NO_ACTIVE_MODELS"

    # Runtime errors
    BACKTEST_FAILED = "BACKTEST_FAILED"
    TRAINING_FAILED = "TRAINING_FAILED"

    # Guardrail errors
    GUARDRAIL_VIOLATION = "GUARDRAIL_VIOLATION"


def format_error(code: ErrorCode, message: str, detail: str = "") -> str:
    """Format a structured error message.

    Parameters
    ----------
    code : ErrorCode
        The error code enum value.
    message : str
        Human-readable error summary.
    detail : str
        Optional additional detail or suggestion.

    Returns
    -------
    str
        Markdown-formatted error string with code prefix.
    """
    parts = [f"**Error** [{code.value}]: {message}"]
    if detail:
        parts.append(f"\n{detail}")
    return "\n".join(parts)


def format_warning(code: ErrorCode, message: str, detail: str = "") -> str:
    """Format a structured warning message.

    Parameters
    ----------
    code : ErrorCode
        The warning code enum value.
    message : str
        Human-readable warning summary.
    detail : str
        Optional additional detail or suggestion.

    Returns
    -------
    str
        Markdown-formatted warning string with code prefix.
    """
    parts = [f"**Warning** [{code.value}]: {message}"]
    if detail:
        parts.append(f"\n{detail}")
    return "\n".join(parts)
