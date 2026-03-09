"""Tests for guard implementations — check() methods, context, and override behavior."""

import pytest
from harnessml.core.guardrails.base import GuardrailError
from harnessml.core.guardrails.inventory import (
    FeatureLeakageGuardrail,
    FeatureStalenessGuardrail,
    NamingConventionGuardrail,
    SanityCheckGuardrail,
    TemporalOrderingGuardrail,
)

# ---------------------------------------------------------------------------
# Leakage guard — non-overridable, blocks target in features
# ---------------------------------------------------------------------------


def test_leakage_guard_blocks_target_in_features():
    """FeatureLeakageGuardrail blocks when denied columns appear in model features."""
    g = FeatureLeakageGuardrail(denylist=["target_score", "result_margin"])
    # Clean features pass
    g.check(context={"model_features": ["elo_diff", "win_pct"]})
    # Denied column triggers error
    with pytest.raises(GuardrailError, match="Feature leakage detected"):
        g.check(context={"model_features": ["elo_diff", "target_score"]})
    # Multiple denied columns all reported
    with pytest.raises(GuardrailError, match="result_margin"):
        g.check(context={"model_features": ["target_score", "result_margin", "safe"]})


# ---------------------------------------------------------------------------
# Temporal guard — non-overridable, blocks future data
# ---------------------------------------------------------------------------


def test_temporal_guard_blocks_future_data():
    """TemporalOrderingGuardrail blocks when training folds >= test fold."""
    g = TemporalOrderingGuardrail()
    # Valid: all training folds before test fold
    g.check(context={"training_folds": [2018, 2019, 2020], "test_fold": 2021})
    # Future data: training fold after test fold
    with pytest.raises(GuardrailError, match="Temporal leakage"):
        g.check(context={"training_folds": [2018, 2019, 2022], "test_fold": 2021})
    # Same fold counts as leakage (>=)
    with pytest.raises(GuardrailError, match="Temporal leakage"):
        g.check(context={"training_folds": [2018, 2021], "test_fold": 2021})


# ---------------------------------------------------------------------------
# Naming guard — rejects bad names
# ---------------------------------------------------------------------------


def test_naming_guard_rejects_bad_names():
    """NamingConventionGuardrail rejects IDs that don't match the pattern."""
    g = NamingConventionGuardrail(pattern=r"^exp-\d{3}-[a-z_]+$")
    # Valid name passes
    g.check(context={"experiment_id": "exp-001-baseline"})
    # Missing prefix
    with pytest.raises(GuardrailError, match="Invalid experiment ID"):
        g.check(context={"experiment_id": "run-001-baseline"})
    # Uppercase rejected
    with pytest.raises(GuardrailError, match="Invalid experiment ID"):
        g.check(context={"experiment_id": "exp-001-Baseline"})
    # Too few digits
    with pytest.raises(GuardrailError, match="Invalid experiment ID"):
        g.check(context={"experiment_id": "exp-01-test"})
    # Empty string
    with pytest.raises(GuardrailError):
        g.check(context={"experiment_id": ""})


# ---------------------------------------------------------------------------
# Overridable guard can be overridden
# ---------------------------------------------------------------------------


def test_overridable_guard_can_be_overridden():
    """Overridable guards suppress errors when human_override=True."""
    g = NamingConventionGuardrail(pattern=r"^exp-\d{3}-[a-z]+$")
    assert g.overridable is True
    # Fails without override
    with pytest.raises(GuardrailError):
        g.check(context={"experiment_id": "bad-name"})
    # Passes with override
    g.check(context={"experiment_id": "bad-name"}, human_override=True)


# ---------------------------------------------------------------------------
# Non-overridable guard cannot be overridden
# ---------------------------------------------------------------------------


def test_non_overridable_guard_cannot_be_overridden():
    """Non-overridable guards raise even when human_override=True."""
    # FeatureLeakageGuardrail
    g1 = FeatureLeakageGuardrail(denylist=["leak"])
    assert g1.overridable is False
    with pytest.raises(GuardrailError):
        g1.check(context={"model_features": ["leak"]}, human_override=True)

    # TemporalOrderingGuardrail
    g2 = TemporalOrderingGuardrail()
    assert g2.overridable is False
    with pytest.raises(GuardrailError):
        g2.check(
            context={"training_folds": [2022], "test_fold": 2021},
            human_override=True,
        )


# ---------------------------------------------------------------------------
# Sanity check guard — detects obvious issues
# ---------------------------------------------------------------------------


def test_sanity_check_guard():
    """SanityCheckGuardrail blocks on failed sanity results."""
    g = SanityCheckGuardrail()
    # Success passes
    g.check(context={"sanity_result": {"status": "success"}})
    # Missing sanity_result
    with pytest.raises(GuardrailError, match="No sanity_result"):
        g.check(context={})
    # Failed status with stderr
    with pytest.raises(GuardrailError, match="Sanity check failed"):
        g.check(context={"sanity_result": {"status": "failed", "stderr": "bad data"}})
    # Failed status with error list
    with pytest.raises(GuardrailError, match="column missing"):
        g.check(context={
            "sanity_result": {
                "status": "error",
                "errors": ["column missing", "type mismatch"],
            }
        })
    # Overridable
    assert g.overridable is True
    g.check(
        context={"sanity_result": {"status": "failed", "stderr": "oops"}},
        human_override=True,
    )


# ---------------------------------------------------------------------------
# Feature staleness guard — warns on stale features
# ---------------------------------------------------------------------------


def test_feature_staleness_guard():
    """FeatureStalenessGuardrail detects hash mismatches between manifest and source."""
    g = FeatureStalenessGuardrail()
    # Matching hashes pass
    g.check(context={
        "manifest_hashes": {"feat_a.py": "abc", "feat_b.py": "def"},
        "source_hashes": {"feat_a.py": "abc", "feat_b.py": "def"},
    })
    # Mismatched hash triggers error
    with pytest.raises(GuardrailError, match="Stale features detected"):
        g.check(context={
            "manifest_hashes": {"feat_a.py": "abc"},
            "source_hashes": {"feat_a.py": "xyz"},
        })
    # Empty manifest or source skips check
    g.check(context={"manifest_hashes": {}, "source_hashes": {"a.py": "x"}})
    g.check(context={"manifest_hashes": {"a.py": "x"}, "source_hashes": {}})
    # New file in source but not in manifest — no error (only checks manifest keys)
    g.check(context={
        "manifest_hashes": {"feat_a.py": "abc"},
        "source_hashes": {"feat_a.py": "abc", "feat_new.py": "new_hash"},
    })
    # Overridable
    assert g.overridable is True
    g.check(
        context={
            "manifest_hashes": {"a.py": "old"},
            "source_hashes": {"a.py": "new"},
        },
        human_override=True,
    )
