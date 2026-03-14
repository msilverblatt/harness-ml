
import pytest
from harnessml.core.runner.experiments.schema import (
    ConfigChange,
    ExperimentRecord,
    ExperimentStatus,
    ExperimentVerdict,
    StructuredConclusion,
    TrialRecord,
)
from pydantic import ValidationError


def test_experiment_record_minimal():
    """Minimal valid record: id + hypothesis + status."""
    r = ExperimentRecord(
        experiment_id="exp-001-add-momentum",
        hypothesis="Adding momentum features will improve Brier score by >0.005",
        status=ExperimentStatus.CREATED,
    )
    assert r.experiment_id == "exp-001-add-momentum"
    assert r.status == ExperimentStatus.CREATED
    assert r.created_at is not None


def test_experiment_record_requires_hypothesis():
    with pytest.raises(ValidationError):
        ExperimentRecord(
            experiment_id="exp-bad",
            status=ExperimentStatus.CREATED,
        )


def test_experiment_record_full():
    """Full record with all fields populated."""
    r = ExperimentRecord(
        experiment_id="exp-003-tune-lr",
        hypothesis="Reducing XGBoost learning rate from 0.1 to 0.05 will reduce overfitting",
        status=ExperimentStatus.COMPLETED,
        parent_id="exp-002-xgboost-baseline",
        branching_reason="Exploring learning rate after baseline established",
        config_changes=[
            ConfigChange(
                path="models.xgb_main.params.learning_rate",
                old_value=0.1,
                new_value=0.05,
            )
        ],
        config_hash="abc123def456",
        trials=[
            TrialRecord(
                trial_id="trial-1",
                metrics={"brier": 0.1350, "accuracy": 0.82, "ece": 0.028},
                duration_seconds=45.2,
            )
        ],
        conclusion=StructuredConclusion(
            verdict=ExperimentVerdict.KEEP,
            primary_metric="brier",
            baseline_value=0.1388,
            result_value=0.1350,
            improvement=0.0038,
            improvement_pct=2.74,
            learnings="Lower learning rate with more trees reduces overfitting. Brier improved 2.7%. ECE also improved.",
            next_steps=["Try learning_rate=0.03 with n_estimators=500", "Test subsample=0.8"],
        ),
    )
    assert r.conclusion.verdict == ExperimentVerdict.KEEP
    assert r.conclusion.improvement_pct == 2.74
    assert len(r.config_changes) == 1
    assert r.parent_id == "exp-002-xgboost-baseline"


def test_experiment_record_serializes_to_jsonl():
    """Record round-trips through JSON."""
    r = ExperimentRecord(
        experiment_id="exp-001",
        hypothesis="Test hypothesis",
        status=ExperimentStatus.CREATED,
    )
    line = r.model_dump_json()
    restored = ExperimentRecord.model_validate_json(line)
    assert restored.experiment_id == r.experiment_id
    assert restored.hypothesis == r.hypothesis


def test_trial_record():
    t = TrialRecord(
        trial_id="trial-1",
        metrics={"brier": 0.14, "accuracy": 0.81},
        duration_seconds=30.0,
        notes="Used default parameters",
    )
    assert t.metrics["brier"] == 0.14


def test_structured_conclusion_requires_verdict():
    with pytest.raises(ValidationError):
        StructuredConclusion(
            primary_metric="brier",
            baseline_value=0.14,
            result_value=0.13,
        )


def test_config_change():
    c = ConfigChange(
        path="models.xgb_main.params.max_depth",
        old_value=6,
        new_value=8,
    )
    assert c.path == "models.xgb_main.params.max_depth"


def test_experiment_verdict_values():
    assert ExperimentVerdict.KEEP == "keep"
    assert ExperimentVerdict.REVERT == "revert"
    assert ExperimentVerdict.PARTIAL == "partial"
    assert ExperimentVerdict.INCONCLUSIVE == "inconclusive"


def test_experiment_status_values():
    assert ExperimentStatus.CREATED == "created"
    assert ExperimentStatus.RUNNING == "running"
    assert ExperimentStatus.COMPLETED == "completed"
    assert ExperimentStatus.FAILED == "failed"
