"""Tests for Pydantic contract schemas in harnessml.core.schemas.contracts."""

from __future__ import annotations

import numpy as np
import pytest
from harnessml.core.schemas.contracts import (
    FeatureMeta,
    Fold,
    GuardrailViolation,
    ModelConfig,
    SourceMeta,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# SourceMeta
# ---------------------------------------------------------------------------


class TestSourceMeta:
    def test_source_meta_valid(self):
        sm = SourceMeta(
            name="game_logs",
            category="external",
            outputs=["data/games.csv"],
            temporal_safety="pre_event",
        )
        assert sm.name == "game_logs"
        assert sm.category == "external"
        assert sm.outputs == ["data/games.csv"]
        assert sm.temporal_safety == "pre_event"
        assert sm.leakage_notes == ""
        assert sm.freshness_check is None

    def test_source_meta_all_temporal_values(self):
        for val in ("pre_event", "post_event", "mixed", "unknown"):
            sm = SourceMeta(
                name="src",
                category="internal",
                outputs=[],
                temporal_safety=val,
            )
            assert sm.temporal_safety == val

    def test_source_meta_rejects_bad_temporal_safety(self):
        with pytest.raises(ValidationError, match="temporal_safety"):
            SourceMeta(
                name="src",
                category="external",
                outputs=[],
                temporal_safety="invalid_value",
            )

    def test_source_meta_rejects_missing_required(self):
        with pytest.raises(ValidationError):
            SourceMeta(name="src")  # missing category, outputs, temporal_safety


# ---------------------------------------------------------------------------
# FeatureMeta
# ---------------------------------------------------------------------------


class TestFeatureMeta:
    def test_feature_meta_defaults(self):
        fm = FeatureMeta(
            name="win_rate",
            category="performance",
            level="entity",
            output_columns=["win_rate_home", "win_rate_away"],
        )
        assert fm.nan_strategy == "median"
        assert fm.temporal_filter is None
        assert fm.tainted_columns == []

    def test_feature_meta_rejects_bad_level(self):
        with pytest.raises(ValidationError, match="level"):
            FeatureMeta(
                name="f",
                category="c",
                level="global",
                output_columns=["col"],
            )


# ---------------------------------------------------------------------------
# GuardrailViolation
# ---------------------------------------------------------------------------


class TestGuardrailViolation:
    def test_guardrail_violation_fields(self):
        gv = GuardrailViolation(
            blocked=True,
            rule="no_leakage",
            message="Column X leaks target",
            source="leakage_guard",
            override_hint="Remove column X",
            overridable=False,
        )
        assert gv.blocked is True
        assert gv.rule == "no_leakage"
        assert gv.message == "Column X leaks target"
        assert gv.source == "leakage_guard"
        assert gv.override_hint == "Remove column X"
        assert gv.overridable is False

    def test_guardrail_violation_defaults(self):
        gv = GuardrailViolation(
            blocked=False,
            rule="naming",
            message="Bad name",
            source="naming_guard",
        )
        assert gv.override_hint is None
        assert gv.overridable is True

    def test_guardrail_violation_to_dict(self):
        gv = GuardrailViolation(
            blocked=True,
            rule="r",
            message="m",
            source="s",
            override_hint="hint",
        )
        d = gv.to_dict()
        assert d["blocked"] is True
        assert d["rule"] == "r"
        assert d["override_hint"] == "hint"

    def test_guardrail_violation_to_dict_omits_none_hint(self):
        gv = GuardrailViolation(
            blocked=False,
            rule="r",
            message="m",
            source="s",
        )
        d = gv.to_dict()
        assert "override_hint" not in d


# ---------------------------------------------------------------------------
# Fold (arbitrary_types_allowed for numpy)
# ---------------------------------------------------------------------------


class TestFold:
    def test_fold_accepts_numpy(self):
        train = np.array([0, 1, 2, 3])
        test = np.array([4, 5])
        cal = np.array([6])
        fold = Fold(
            fold_id=0,
            train_idx=train,
            test_idx=test,
            calibration_idx=cal,
        )
        assert fold.fold_id == 0
        np.testing.assert_array_equal(fold.train_idx, train)
        np.testing.assert_array_equal(fold.test_idx, test)
        np.testing.assert_array_equal(fold.calibration_idx, cal)

    def test_fold_calibration_idx_defaults_none(self):
        fold = Fold(
            fold_id=1,
            train_idx=np.array([0]),
            test_idx=np.array([1]),
        )
        assert fold.calibration_idx is None


# ---------------------------------------------------------------------------
# ModelConfig serialization round-trip
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_model_config_serialization(self):
        mc = ModelConfig(
            name="xgb_v1",
            type="xgboost",
            mode="classifier",
            features=["f1", "f2"],
            params={"n_estimators": 100, "max_depth": 6},
            active=True,
            n_seeds=3,
        )
        d = mc.model_dump()
        restored = ModelConfig(**d)
        assert restored.name == mc.name
        assert restored.type == mc.type
        assert restored.mode == mc.mode
        assert restored.features == mc.features
        assert restored.params == mc.params
        assert restored.active == mc.active
        assert restored.n_seeds == mc.n_seeds
        assert restored.pre_calibrate is None
        assert restored.training_filter is None

    def test_model_config_rejects_bad_mode(self):
        with pytest.raises(ValidationError, match="mode"):
            ModelConfig(
                name="m",
                type="t",
                mode="transformer",
                features=[],
                params={},
            )
