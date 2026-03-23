"""Tests for the task type registry."""

import pytest
from harness.ml.tasks.registry import TaskRegistry


class TestTaskRegistry:
    def test_get_binary(self):
        task = TaskRegistry.get("binary")
        assert task.name == "binary"

    def test_get_binary_case_insensitive(self):
        task = TaskRegistry.get("Binary")
        assert task.name == "binary"

    def test_list_available(self):
        available = TaskRegistry.list_available()
        assert "binary" in available

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            TaskRegistry.get("unknown_task_type")

    def test_get_regression(self):
        task = TaskRegistry.get("regression")
        assert task.name == "regression"

    def test_get_multiclass(self):
        task = TaskRegistry.get("multiclass")
        assert task.name == "multiclass"
