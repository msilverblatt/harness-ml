"""Tests for guardrail base classes — overridable and non-overridable."""

import pytest

from harnessml.core.guardrails.base import Guardrail, GuardrailError
from harnessml.core.guardrails.inventory import (
    FeatureLeakageGuardrail,
    NamingConventionGuardrail,
)


# ---------------------------------------------------------------------------
# Basic check
# ---------------------------------------------------------------------------


def test_guardrail_check_passes():
    g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
    g.check(context={"experiment_id": "exp-001-test"})  # should not raise


def test_guardrail_check_fails():
    g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
    with pytest.raises(GuardrailError):
        g.check(context={"experiment_id": "bad"})


# ---------------------------------------------------------------------------
# Overridable
# ---------------------------------------------------------------------------


def test_guardrail_human_override():
    g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
    # human_override=True should suppress the error for overridable guardrails
    g.check(context={"experiment_id": "bad"}, human_override=True)


# ---------------------------------------------------------------------------
# Non-overridable
# ---------------------------------------------------------------------------


def test_non_overridable_guardrail():
    g = FeatureLeakageGuardrail(denylist=["leaky_col"])
    with pytest.raises(GuardrailError):
        # human_override=True must NOT suppress for non-overridable guardrails
        g.check(context={"model_features": ["leaky_col"]}, human_override=True)


def test_non_overridable_flag():
    g = FeatureLeakageGuardrail(denylist=["leaky_col"])
    assert g.overridable is False


# ---------------------------------------------------------------------------
# Violation metadata
# ---------------------------------------------------------------------------


def test_violation_contains_rule():
    g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
    with pytest.raises(GuardrailError) as exc_info:
        g.check(context={"experiment_id": "bad"})
    assert exc_info.value.violation.rule == "naming_convention"
    assert exc_info.value.violation.blocked is True


def test_non_overridable_violation_has_no_override_hint():
    g = FeatureLeakageGuardrail(denylist=["leaky_col"])
    with pytest.raises(GuardrailError) as exc_info:
        g.check(context={"model_features": ["leaky_col"]})
    assert exc_info.value.violation.override_hint is None


# ---------------------------------------------------------------------------
# Custom guardrail subclass
# ---------------------------------------------------------------------------


class AlwaysFailGuardrail(Guardrail):
    def __init__(self, overridable: bool = True):
        super().__init__(
            name="always_fail",
            overridable=overridable,
            description="Always fails",
        )

    def _check(self, context: dict) -> None:
        self._fail("always fails", source="test")


def test_custom_guardrail_overridable():
    g = AlwaysFailGuardrail(overridable=True)
    with pytest.raises(GuardrailError):
        g.check({})
    # override works
    g.check({}, human_override=True)


def test_custom_guardrail_not_overridable():
    g = AlwaysFailGuardrail(overridable=False)
    with pytest.raises(GuardrailError):
        g.check({}, human_override=True)
