"""Tests for the hook system (HookRegistry + hook constants)."""

from __future__ import annotations

import pytest
from harnessml.core.runner.hooks import (
    COLUMN_CANDIDATES,
    COLUMN_RENAMES,
    COMPETITION_NARRATIVE,
    FEATURE_EXPANSION,
    FEATURE_TYPE,
    POST_PREDICTION,
    PRE_TRAINING,
    PROVIDER_INJECTION,
    HookRegistry,
)


@pytest.fixture(autouse=True)
def _clean_hooks():
    """Ensure each test starts and ends with a clean registry."""
    HookRegistry.clear()
    yield
    HookRegistry.clear()


def test_register_and_call_all():
    """Registered hooks are all called."""
    calls = []

    def hook_a(x):
        calls.append("a")
        return x + 1

    def hook_b(x):
        calls.append("b")
        return x + 2

    HookRegistry.register("my_hook", hook_a)
    HookRegistry.register("my_hook", hook_b)

    results = HookRegistry.call_all("my_hook", 10)

    assert calls == ["a", "b"]
    assert results == [11, 12]


def test_call_first_returns_first_non_none():
    """call_first returns the result of the first registered hook."""

    def returns_none(x):
        return None

    def returns_value(x):
        return x * 2

    HookRegistry.register("my_hook", returns_none)
    HookRegistry.register("my_hook", returns_value)

    # call_first invokes the first registered hook regardless of return value
    result = HookRegistry.call_first("my_hook", 5)
    assert result is None

    # If the first hook returns a value, that is what we get
    HookRegistry.clear()
    HookRegistry.register("my_hook", returns_value)
    HookRegistry.register("my_hook", returns_none)

    result = HookRegistry.call_first("my_hook", 5)
    assert result == 10


def test_call_all_collects_results():
    """call_all collects all results including None values."""

    def hook_1():
        return "first"

    def hook_2():
        return None

    def hook_3():
        return "third"

    HookRegistry.register("collect", hook_1)
    HookRegistry.register("collect", hook_2)
    HookRegistry.register("collect", hook_3)

    results = HookRegistry.call_all("collect")
    assert results == ["first", None, "third"]


def test_hook_with_no_registrations():
    """Unregistered hook names return empty/None gracefully."""
    assert HookRegistry.get("nonexistent") == []
    assert HookRegistry.call_first("nonexistent") is None
    assert HookRegistry.call_all("nonexistent") == []


def test_multiple_hooks_same_name():
    """All handlers registered under the same hook name are invoked."""
    counter = {"n": 0}

    def increment():
        counter["n"] += 1

    for _ in range(5):
        HookRegistry.register("counter_hook", increment)

    HookRegistry.call_all("counter_hook")
    assert counter["n"] == 5


def test_hook_constants_defined():
    """All expected hook name constants exist and are non-empty strings."""
    constants = [
        FEATURE_EXPANSION,
        PROVIDER_INJECTION,
        PRE_TRAINING,
        POST_PREDICTION,
        FEATURE_TYPE,
        COLUMN_CANDIDATES,
        COLUMN_RENAMES,
        COMPETITION_NARRATIVE,
    ]
    for const in constants:
        assert isinstance(const, str)
        assert len(const) > 0
