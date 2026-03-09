"""Structured logging for HarnessML using structlog."""
from __future__ import annotations

import logging
import os

import structlog

_configured = False


def configure_logging(json_output: bool | None = None) -> None:
    """Configure structlog processors and output format.

    Safe to call multiple times; only the first call takes effect.

    Parameters
    ----------
    json_output:
        If *True*, render as JSON lines.  If *False*, use the coloured
        console renderer.  When *None* (default), reads the
        ``HARNESS_LOG_FORMAT`` env-var (``"json"`` → JSON, anything else
        → console).
    """
    global _configured
    if _configured:
        return

    if json_output is None:
        json_output = os.environ.get("HARNESS_LOG_FORMAT", "").lower() == "json"

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger for *name*.

    Calls :func:`configure_logging` on first use so callers never need
    to worry about setup.
    """
    configure_logging()
    return structlog.get_logger(name)
