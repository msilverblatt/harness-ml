"""Tests for FeatureLevel and GuardrailRule enums."""
from harnessml.core.runner.schema import FeatureLevel
from harnessml.core.schemas.contracts import GuardrailRule


class TestFeatureLevel:
    def test_values(self):
        assert FeatureLevel.ENTITY == "entity"
        assert FeatureLevel.INTERACTION == "interaction"
        assert FeatureLevel.REGIME == "regime"
        assert FeatureLevel.QUERY == "query"
        assert FeatureLevel.INSTANCE == "instance"

    def test_member_count(self):
        assert len(FeatureLevel) == 5

    def test_string_comparison(self):
        assert FeatureLevel.ENTITY == "entity"
        assert FeatureLevel("entity") is FeatureLevel.ENTITY


class TestGuardrailRule:
    def test_values(self):
        assert GuardrailRule.FEATURE_LEAKAGE == "feature_leakage"
        assert GuardrailRule.TEMPORAL_ORDERING == "temporal_ordering"
        assert GuardrailRule.NAMING == "naming"
        assert GuardrailRule.CRITICAL_PATH == "critical_path"
        assert GuardrailRule.MANDATORY_LOGGING == "mandatory_logging"
        assert GuardrailRule.DO_NOT_RETRY == "do_not_retry"

    def test_member_count(self):
        assert len(GuardrailRule) == 6

    def test_string_comparison(self):
        assert GuardrailRule.FEATURE_LEAKAGE == "feature_leakage"
        assert GuardrailRule("critical_path") is GuardrailRule.CRITICAL_PATH
