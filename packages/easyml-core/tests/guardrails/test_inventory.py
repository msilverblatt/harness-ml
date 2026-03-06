"""Tests for all 11 guardrail implementations."""

import time

import pytest

from easyml.core.guardrails.base import GuardrailError
from easyml.core.guardrails.inventory import (
    ConfigProtectionGuardrail,
    CriticalPathGuardrail,
    DoNotRetryGuardrail,
    ExperimentLoggedGuardrail,
    FeatureLeakageGuardrail,
    FeatureStalenessGuardrail,
    NamingConventionGuardrail,
    RateLimitGuardrail,
    SanityCheckGuardrail,
    SingleVariableGuardrail,
    TemporalOrderingGuardrail,
)


# ---------------------------------------------------------------------------
# 1. SanityCheckGuardrail
# ---------------------------------------------------------------------------


class TestSanityCheckGuardrail:
    def test_passes_on_success(self):
        g = SanityCheckGuardrail()
        g.check(context={"sanity_result": {"status": "success"}})

    def test_fails_on_error(self):
        g = SanityCheckGuardrail()
        with pytest.raises(GuardrailError, match="Sanity check failed"):
            g.check(context={"sanity_result": {"status": "failed", "stderr": "bad config"}})

    def test_fails_when_missing(self):
        g = SanityCheckGuardrail()
        with pytest.raises(GuardrailError, match="No sanity_result"):
            g.check(context={})

    def test_includes_error_lines(self):
        g = SanityCheckGuardrail()
        with pytest.raises(GuardrailError, match="missing col"):
            g.check(context={
                "sanity_result": {
                    "status": "failed",
                    "errors": ["missing col X", "missing col Y"],
                }
            })

    def test_overridable(self):
        g = SanityCheckGuardrail()
        assert g.overridable is True
        g.check(
            context={"sanity_result": {"status": "failed"}},
            human_override=True,
        )


# ---------------------------------------------------------------------------
# 2. NamingConventionGuardrail
# ---------------------------------------------------------------------------


class TestNamingConventionGuardrail:
    def test_valid_name(self):
        g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
        g.check(context={"experiment_id": "exp-001-test"})

    def test_invalid_name(self):
        g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
        with pytest.raises(GuardrailError, match="Invalid experiment ID"):
            g.check(context={"experiment_id": "bad"})

    def test_empty_string(self):
        g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
        with pytest.raises(GuardrailError):
            g.check(context={"experiment_id": ""})

    def test_overridable(self):
        g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
        g.check(context={"experiment_id": "bad"}, human_override=True)


# ---------------------------------------------------------------------------
# 3. DoNotRetryGuardrail
# ---------------------------------------------------------------------------


class TestDoNotRetryGuardrail:
    def test_no_match(self):
        g = DoNotRetryGuardrail(patterns=[
            {"pattern": "temperature scaling", "reference": "EXP-002", "reason": "hurts"},
        ])
        g.check(context={"description": "add new model"})

    def test_match(self):
        g = DoNotRetryGuardrail(patterns=[
            {"pattern": "temperature scaling", "reference": "EXP-002", "reason": "hurts"},
        ])
        with pytest.raises(GuardrailError, match="do-not-retry"):
            g.check(context={"description": "try temperature scaling again"})

    def test_case_insensitive(self):
        g = DoNotRetryGuardrail(patterns=[
            {"pattern": "Temperature Scaling", "reference": "EXP-002", "reason": "hurts"},
        ])
        with pytest.raises(GuardrailError):
            g.check(context={"description": "temperature scaling experiment"})

    def test_empty_description(self):
        g = DoNotRetryGuardrail(patterns=[
            {"pattern": "temperature", "reference": "EXP-002", "reason": "hurts"},
        ])
        g.check(context={"description": ""})  # should not raise

    def test_no_patterns(self):
        g = DoNotRetryGuardrail()
        g.check(context={"description": "anything"})

    def test_overridable(self):
        g = DoNotRetryGuardrail(patterns=[
            {"pattern": "temperature", "reference": "EXP-002", "reason": "hurts"},
        ])
        g.check(context={"description": "temperature"}, human_override=True)


# ---------------------------------------------------------------------------
# 4. SingleVariableGuardrail
# ---------------------------------------------------------------------------


class TestSingleVariableGuardrail:
    def test_single_change_passes(self):
        g = SingleVariableGuardrail()
        g.check(context={"total_changes": 1})

    def test_multi_change_fails(self):
        g = SingleVariableGuardrail()
        with pytest.raises(GuardrailError, match="Multi-variable"):
            g.check(context={"total_changes": 3})

    def test_zero_changes_passes(self):
        g = SingleVariableGuardrail()
        g.check(context={"total_changes": 0})

    def test_custom_max(self):
        g = SingleVariableGuardrail(max_changes=2)
        g.check(context={"total_changes": 2})
        with pytest.raises(GuardrailError):
            g.check(context={"total_changes": 3})

    def test_overridable(self):
        g = SingleVariableGuardrail()
        g.check(context={"total_changes": 5}, human_override=True)


# ---------------------------------------------------------------------------
# 5. FeatureLeakageGuardrail (NON-overridable)
# ---------------------------------------------------------------------------


class TestFeatureLeakageGuardrail:
    def test_clean_features(self):
        g = FeatureLeakageGuardrail(denylist=["leaky_col"])
        g.check(context={"model_features": ["safe_col", "other_col"]})

    def test_leaky_feature(self):
        g = FeatureLeakageGuardrail(denylist=["leaky_col"])
        with pytest.raises(GuardrailError, match="leaky_col"):
            g.check(context={"model_features": ["leaky_col"]})

    def test_multiple_leaky(self):
        g = FeatureLeakageGuardrail(denylist=["a", "b"])
        with pytest.raises(GuardrailError):
            g.check(context={"model_features": ["a", "b", "c"]})

    def test_not_overridable(self):
        g = FeatureLeakageGuardrail(denylist=["leaky_col"])
        assert g.overridable is False
        with pytest.raises(GuardrailError):
            g.check(context={"model_features": ["leaky_col"]}, human_override=True)

    def test_empty_denylist(self):
        g = FeatureLeakageGuardrail(denylist=[])
        g.check(context={"model_features": ["anything"]})


# ---------------------------------------------------------------------------
# 6. ConfigProtectionGuardrail
# ---------------------------------------------------------------------------


class TestConfigProtectionGuardrail:
    def test_unprotected_change(self):
        g = ConfigProtectionGuardrail(protected_paths=["config/production.yaml"])
        g.check(context={"changed_files": ["config/experimental.yaml"]})

    def test_protected_change(self):
        g = ConfigProtectionGuardrail(protected_paths=["config/production.yaml"])
        with pytest.raises(GuardrailError, match="Protected"):
            g.check(context={"changed_files": ["config/production.yaml"]})

    def test_no_changes(self):
        g = ConfigProtectionGuardrail(protected_paths=["config/production.yaml"])
        g.check(context={"changed_files": []})

    def test_overridable(self):
        g = ConfigProtectionGuardrail(protected_paths=["config/production.yaml"])
        g.check(
            context={"changed_files": ["config/production.yaml"]},
            human_override=True,
        )


# ---------------------------------------------------------------------------
# 7. CriticalPathGuardrail (NON-overridable)
# ---------------------------------------------------------------------------


class TestCriticalPathGuardrail:
    def test_safe_operation(self):
        g = CriticalPathGuardrail(protected_dirs=["/data/models"])
        g.check(context={"operation": "write", "target_path": "/data/models/new.pkl"})

    def test_delete_blocked(self):
        g = CriticalPathGuardrail(protected_dirs=["/data/models"])
        with pytest.raises(GuardrailError, match="Cannot delete"):
            g.check(context={"operation": "delete", "target_path": "/data/models/core.pkl"})

    def test_overwrite_blocked(self):
        g = CriticalPathGuardrail(protected_dirs=["/data/models"])
        with pytest.raises(GuardrailError, match="Cannot overwrite"):
            g.check(context={"operation": "overwrite", "target_path": "/data/models/core.pkl"})

    def test_outside_protected_dir(self):
        g = CriticalPathGuardrail(protected_dirs=["/data/models"])
        g.check(context={"operation": "delete", "target_path": "/tmp/junk.pkl"})

    def test_not_overridable(self):
        g = CriticalPathGuardrail(protected_dirs=["/data/models"])
        assert g.overridable is False
        with pytest.raises(GuardrailError):
            g.check(
                context={"operation": "delete", "target_path": "/data/models/x.pkl"},
                human_override=True,
            )


# ---------------------------------------------------------------------------
# 8. RateLimitGuardrail
# ---------------------------------------------------------------------------


class TestRateLimitGuardrail:
    def test_no_previous_run(self):
        g = RateLimitGuardrail(cooldown_seconds=60)
        g.check(context={"last_run_timestamp": None})

    def test_cooldown_expired(self):
        g = RateLimitGuardrail(cooldown_seconds=1)
        g.check(context={"last_run_timestamp": time.time() - 10})

    def test_cooldown_active(self):
        g = RateLimitGuardrail(cooldown_seconds=300)
        with pytest.raises(GuardrailError, match="Rate limit"):
            g.check(context={"last_run_timestamp": time.time()})

    def test_overridable(self):
        g = RateLimitGuardrail(cooldown_seconds=300)
        g.check(context={"last_run_timestamp": time.time()}, human_override=True)


# ---------------------------------------------------------------------------
# 9. ExperimentLoggedGuardrail
# ---------------------------------------------------------------------------


class TestExperimentLoggedGuardrail:
    def test_all_logged(self):
        g = ExperimentLoggedGuardrail()
        g.check(context={"has_unlogged": False})

    def test_unlogged(self):
        g = ExperimentLoggedGuardrail()
        with pytest.raises(GuardrailError, match="not been logged"):
            g.check(context={"has_unlogged": True})

    def test_overridable(self):
        g = ExperimentLoggedGuardrail()
        g.check(context={"has_unlogged": True}, human_override=True)


# ---------------------------------------------------------------------------
# 10. FeatureStalenessGuardrail
# ---------------------------------------------------------------------------


class TestFeatureStalenessGuardrail:
    def test_no_staleness(self):
        g = FeatureStalenessGuardrail()
        g.check(context={
            "manifest_hashes": {"features.py": "abc123"},
            "source_hashes": {"features.py": "abc123"},
        })

    def test_stale(self):
        g = FeatureStalenessGuardrail()
        with pytest.raises(GuardrailError, match="Stale"):
            g.check(context={
                "manifest_hashes": {"features.py": "abc123"},
                "source_hashes": {"features.py": "def456"},
            })

    def test_empty_manifest(self):
        g = FeatureStalenessGuardrail()
        g.check(context={"manifest_hashes": {}, "source_hashes": {"a.py": "x"}})

    def test_overridable(self):
        g = FeatureStalenessGuardrail()
        g.check(
            context={
                "manifest_hashes": {"a.py": "old"},
                "source_hashes": {"a.py": "new"},
            },
            human_override=True,
        )


# ---------------------------------------------------------------------------
# 11. TemporalOrderingGuardrail (NON-overridable)
# ---------------------------------------------------------------------------


class TestTemporalOrderingGuardrail:
    def test_valid_ordering(self):
        g = TemporalOrderingGuardrail()
        g.check(context={
            "training_folds": [2015, 2016, 2017],
            "test_fold": 2018,
        })

    def test_leakage(self):
        g = TemporalOrderingGuardrail()
        with pytest.raises(GuardrailError, match="Temporal leakage"):
            g.check(context={
                "training_folds": [2015, 2016, 2018],
                "test_fold": 2017,
            })

    def test_same_season_leakage(self):
        g = TemporalOrderingGuardrail()
        with pytest.raises(GuardrailError, match="Temporal leakage"):
            g.check(context={
                "training_folds": [2015, 2016, 2017],
                "test_fold": 2017,
            })

    def test_no_test_fold(self):
        g = TemporalOrderingGuardrail()
        g.check(context={"training_folds": [2015, 2016]})

    def test_not_overridable(self):
        g = TemporalOrderingGuardrail()
        assert g.overridable is False
        with pytest.raises(GuardrailError):
            g.check(
                context={
                    "training_folds": [2018],
                    "test_fold": 2017,
                },
                human_override=True,
            )
