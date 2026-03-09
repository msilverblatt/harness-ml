"""Pydantic schemas for the JSONL experiment journal.

The journal is the single source of truth for experiment history. Each line
is a self-contained ExperimentRecord with full metadata for reproducibility.

EXPERIMENT_LOG.md is generated from this — never hand-edited.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExperimentStatus(str, Enum):
    """Lifecycle status of an experiment."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ExperimentVerdict(str, Enum):
    """Outcome decision for a completed experiment."""
    KEEP = "keep"           # Promote to production
    REVERT = "revert"       # Discard changes
    PARTIAL = "partial"     # Keep some changes, revert others
    INCONCLUSIVE = "inconclusive"  # Need more data / different approach


class ConfigChange(BaseModel):
    """A single configuration change made in this experiment."""
    path: str               # Dot-notation path: "models.xgb_main.params.max_depth"
    old_value: Any = None   # Previous value (None if new field)
    new_value: Any = None   # New value (None if removed)
    reason: str = ""        # Why this change was made


class TrialRecord(BaseModel):
    """One training run within an experiment.

    An experiment may have multiple trials (e.g., HPO iterations,
    seed sweeps, or iterative refinements).
    """
    trial_id: str
    metrics: dict[str, float] = {}
    fold_metrics: dict[str, dict[str, float]] = {}  # {fold_id: {metric: value}}
    duration_seconds: float | None = None
    config_hash: str | None = None     # Hash of the exact config used
    model_artifacts: list[str] = []    # Paths to saved model files
    notes: str = ""


class StructuredConclusion(BaseModel):
    """Machine-readable conclusion with quantitative results.

    This is the key differentiator from free-form experiment logging.
    An agent or human can programmatically compare experiments, build
    leaderboards, and make data-driven decisions about what to try next.
    """
    verdict: ExperimentVerdict
    primary_metric: str = ""           # Which metric drove the verdict
    baseline_value: float | None = None
    result_value: float | None = None
    improvement: float | None = None   # result - baseline (signed)
    improvement_pct: float | None = None
    secondary_metrics: dict[str, float] = {}  # Other metrics of interest
    learnings: str = ""                # What was learned (human-readable)
    next_steps: list[str] = []         # Suggested follow-up experiments
    do_not_retry_reason: str | None = None  # If this approach should be blocked


class ExperimentRecord(BaseModel):
    """A single experiment record in the JSONL journal.

    This is the atomic unit of experiment history. Each record is
    self-contained — it has everything needed to understand what was
    tried, why, and what happened, without looking at any other files.

    Fields are ordered by lifecycle: identity -> context -> execution -> results.
    """
    # --- Identity ---
    experiment_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    # --- Context ---
    hypothesis: str         # REQUIRED: what we expect to learn
    parent_id: str | None = None       # For multi-step research plans
    branching_reason: str = ""         # Why this branched from parent
    tags: list[str] = []               # Free-form tags for filtering
    phase: str = ""                    # Workflow phase: "eda", "feature_discovery", "tuning", etc.

    # --- Configuration ---
    config_changes: list[ConfigChange] = []
    config_hash: str | None = None     # SHA-256 of the full config at experiment time
    overlay_path: str | None = None    # Path to the overlay YAML

    # --- Execution ---
    status: ExperimentStatus
    trials: list[TrialRecord] = []
    total_duration_seconds: float | None = None

    # --- Results ---
    conclusion: StructuredConclusion | None = None

    @field_validator("hypothesis")
    @classmethod
    def _validate_hypothesis(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Hypothesis is required. A good hypothesis states what you expect "
                "to happen and why. Example: 'Adding momentum features will improve "
                "Brier score by >0.005 because recent form matters more than season averages.'"
            )
        return v.strip()
