"""Error classification for Studio API responses."""
from __future__ import annotations

from enum import Enum

from fastapi.responses import JSONResponse


class ErrorCategory(str, Enum):
    TRANSIENT = "transient"      # Network, timeout - show retry button
    NOT_FOUND = "not_found"      # Missing data - show helpful empty state
    PERMANENT = "permanent"      # Bad config - show error with guidance
    SERVER = "server"            # Internal error - show generic error


def classify_error(exc: Exception) -> ErrorCategory:
    """Classify an exception into an error category."""
    if isinstance(exc, FileNotFoundError):
        return ErrorCategory.NOT_FOUND
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return ErrorCategory.TRANSIENT
    if isinstance(exc, (ValueError, KeyError)):
        return ErrorCategory.PERMANENT
    return ErrorCategory.SERVER


def _status_code(category: ErrorCategory) -> int:
    """Map error category to HTTP status code."""
    if category == ErrorCategory.NOT_FOUND:
        return 404
    if category == ErrorCategory.TRANSIENT:
        return 503
    if category == ErrorCategory.PERMANENT:
        return 422
    return 500


def error_response(exc: Exception) -> JSONResponse:
    """Build a structured error JSONResponse from an exception."""
    category = classify_error(exc)
    return JSONResponse(
        status_code=_status_code(category),
        content={
            "error": str(exc),
            "category": category.value,
            "retryable": category == ErrorCategory.TRANSIENT,
        },
    )
