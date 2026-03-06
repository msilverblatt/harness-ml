"""Core data schemas — shared contracts used by all easyml packages."""
from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Temporal / leakage filters
# ---------------------------------------------------------------------------

class TemporalFilter(BaseModel):
    """Controls which rows a feature function may access."""

    exclude_event_types: list[str] = []
    max_date_field: str | None = None


# ---------------------------------------------------------------------------
# Source metadata
# ---------------------------------------------------------------------------

class SourceMeta(BaseModel):
    """Describes a data source with leakage and freshness metadata."""

    name: str
    category: str  # e.g. "external", "internal", "derived"
    outputs: list[str]  # output file/dir paths
    temporal_safety: Literal["pre_event", "post_event", "mixed", "unknown"]
    leakage_notes: str = ""
    freshness_check: str | None = None  # how freshness is checked


# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

class FeatureMeta(BaseModel):
    """Registry entry describing one feature function."""

    name: str
    category: str
    level: Literal["entity", "pairwise"]
    output_columns: list[str]
    nan_strategy: str = "median"
    temporal_filter: TemporalFilter | None = None
    tainted_columns: list[str] = []


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Single model definition inside a pipeline config."""

    name: str
    type: str
    mode: Literal["classifier", "regressor"]
    features: list[str]
    params: dict[str, Any]
    active: bool = True
    pre_calibrate: str | None = None
    n_seeds: int = 1
    training_filter: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------

class EnsembleConfig(BaseModel):
    """Stacking / averaging / weighting setup."""

    method: Literal["stacked", "average", "weighted"]
    meta_learner_type: str = "logistic"
    meta_learner_params: dict[str, Any] = {}
    post_calibration: str | None = None
    temperature: float = 1.0
    clip_floor: float = 0.0
    availability_adjustment: float = 0.1


# ---------------------------------------------------------------------------
# Artifacts & pipeline stages
# ---------------------------------------------------------------------------

class ArtifactDecl(BaseModel):
    """Declares one pipeline artifact (input or output)."""

    name: str
    type: Literal["data", "features", "model", "predictions"]
    path: str


class StageConfig(BaseModel):
    """One stage in the pipeline DAG."""

    script: str
    consumes: list[str]
    produces: list[ArtifactDecl]
    params: dict[str, Any] = {}


class PipelineConfig(BaseModel):
    """Full pipeline specification — ordered stages."""

    stages: dict[str, StageConfig]


# ---------------------------------------------------------------------------
# Run tracking
# ---------------------------------------------------------------------------

class RunManifest(BaseModel):
    """Metadata for one versioned pipeline run."""

    run_id: str
    created_at: str
    labels: list[str]
    stage: str
    config_hash: str | None = None
    files: list[str] = []


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

class ExperimentResult(BaseModel):
    """Outcome of a single experiment run."""

    experiment_id: str
    baseline_metrics: dict[str, Any]
    result_metrics: dict[str, Any]
    delta: dict[str, Any]
    verdict: Literal["keep", "revert", "partial"]
    models_trained: list[str]
    notes: str = ""


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

class GuardrailViolation(BaseModel):
    """A guardrail check result — may block execution."""

    blocked: bool
    rule: str
    message: str
    source: str
    override_hint: str | None = None
    overridable: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict (matches MCP tool response shape)."""
        d: dict[str, Any] = {
            "blocked": self.blocked,
            "rule": self.rule,
            "message": self.message,
            "source": self.source,
            "overridable": self.overridable,
        }
        if self.override_hint is not None:
            d["override_hint"] = self.override_hint
        return d


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------

class Fold(BaseModel):
    """One CV fold with index arrays (e.g. LOSO fold split)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    calibration_idx: np.ndarray | None = None
