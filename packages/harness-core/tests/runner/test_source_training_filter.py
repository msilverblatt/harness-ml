"""Tests for TrainingFilterDef and SourceConfig.join_on validation."""
import pytest
from harnessml.core.runner.schema import SourceConfig, TrainingFilterDef


class TestTrainingFilterDef:
    def test_minimal(self):
        tf = TrainingFilterDef(expr="season >= 2015")
        assert tf.expr == "season >= 2015"
        assert tf.description == ""
        assert tf.apply_to == "train"

    def test_full(self):
        tf = TrainingFilterDef(
            expr="status == 'active'",
            description="Only active records",
            apply_to="both",
        )
        assert tf.apply_to == "both"
        assert tf.description == "Only active records"

    def test_apply_to_test(self):
        tf = TrainingFilterDef(expr="x > 0", apply_to="test")
        assert tf.apply_to == "test"

    def test_invalid_apply_to(self):
        with pytest.raises(Exception):
            TrainingFilterDef(expr="x > 0", apply_to="invalid")


class TestSourceConfigJoinOnValidation:
    def test_none_is_valid(self):
        sc = SourceConfig(name="src1", join_on=None)
        assert sc.join_on is None

    def test_non_empty_list_is_valid(self):
        sc = SourceConfig(name="src1", join_on=["id"])
        assert sc.join_on == ["id"]

    def test_empty_list_is_invalid(self):
        with pytest.raises(Exception, match="join_on must not be empty"):
            SourceConfig(name="src1", join_on=[])

    def test_omitted_defaults_to_none(self):
        sc = SourceConfig(name="src1")
        assert sc.join_on is None
