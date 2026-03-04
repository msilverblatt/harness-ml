"""Guardrail base classes with override support.

Two types:
- **Overridable guardrails**: Can be bypassed with ``human_override=True``.
- **Non-overridable guardrails**: NEVER bypass (e.g., feature leakage, critical path).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from easyml.core.schemas.contracts import GuardrailViolation


# ---------------------------------------------------------------------------
# Exception wrapper
# ---------------------------------------------------------------------------

class GuardrailError(Exception):
    """Exception raised when a guardrail check fails.

    Wraps a :class:`GuardrailViolation` Pydantic model with the details.
    """

    def __init__(self, violation: GuardrailViolation) -> None:
        self.violation = violation
        super().__init__(violation.message)


# ---------------------------------------------------------------------------
# Base guardrail
# ---------------------------------------------------------------------------

class Guardrail(ABC):
    """Base guardrail class. Subclasses implement :meth:`_check`.

    Parameters
    ----------
    name:
        Short identifier for this guardrail (e.g. ``"naming_convention"``).
    overridable:
        Whether ``human_override=True`` can bypass a failure.
    description:
        Human-readable description of what this guardrail checks.
    """

    def __init__(
        self,
        name: str,
        overridable: bool = True,
        description: str = "",
    ) -> None:
        self.name = name
        self.overridable = overridable
        self.description = description

    def check(self, context: dict, human_override: bool = False) -> None:
        """Check the guardrail against *context*.

        Raises :class:`GuardrailError` wrapping a
        :class:`~easyml.schemas.core.GuardrailViolation` on failure.

        If ``human_override`` is ``True`` and the guardrail is overridable,
        the error is suppressed.
        """
        try:
            self._check(context)
        except GuardrailError:
            if human_override and self.overridable:
                return  # allow override
            raise

    @abstractmethod
    def _check(self, context: dict) -> None:
        """Subclass hook — raise :class:`GuardrailError` on violation."""

    def _fail(self, message: str, source: str = "") -> None:
        """Convenience method to raise a :class:`GuardrailError`."""
        raise GuardrailError(
            GuardrailViolation(
                blocked=True,
                rule=self.name,
                message=message,
                source=source or self.name,
                overridable=self.overridable,
                override_hint=(
                    "Re-call with human_override=true to bypass."
                    if self.overridable
                    else None
                ),
            )
        )
