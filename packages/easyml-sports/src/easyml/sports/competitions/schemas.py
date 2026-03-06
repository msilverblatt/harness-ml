"""Pydantic v2 schemas for the competition engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CompetitionFormat(str, Enum):
    """Supported competition bracket/tournament formats."""

    single_elimination = "single_elimination"
    double_elimination = "double_elimination"
    round_robin = "round_robin"
    swiss = "swiss"
    group_knockout = "group_knockout"


class SeedingMode(str, Enum):
    """How participants are seeded into the bracket."""

    ranked = "ranked"
    random = "random"
    manual = "manual"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class ScoringConfig(BaseModel):
    """Scoring system configuration.

    Two types supported:
    - ``per_round``: a list of point values, one per round.
    - ``points``: win/draw/loss point values (e.g. for round-robin).
    """

    type: str = "per_round"
    values: list[float] = Field(default_factory=list)
    win: float = 0.0
    draw: float = 0.0
    loss: float = 0.0

    @model_validator(mode="after")
    def _validate_scoring(self) -> "ScoringConfig":
        if self.type == "per_round" and len(self.values) == 0:
            raise ValueError("per_round scoring requires a non-empty 'values' list")
        if self.type == "points" and self.win == 0.0:
            raise ValueError("points scoring requires a non-zero 'win' value")
        return self


# ---------------------------------------------------------------------------
# Group / Knockout configs
# ---------------------------------------------------------------------------


class GroupConfig(BaseModel):
    """Configuration for the group stage of a group_knockout competition."""

    n_groups: int = 1
    group_size: int = 4
    format: CompetitionFormat = CompetitionFormat.round_robin
    advance: int = 2
    scoring: ScoringConfig | None = None


class KnockoutConfig(BaseModel):
    """Configuration for the knockout stage of a group_knockout competition."""

    format: CompetitionFormat = CompetitionFormat.single_elimination
    scoring: ScoringConfig | None = None


# ---------------------------------------------------------------------------
# Top-level competition config
# ---------------------------------------------------------------------------


class CompetitionConfig(BaseModel):
    """Full configuration for a competition/bracket."""

    format: CompetitionFormat = CompetitionFormat.single_elimination
    n_participants: int = Field(default=2, ge=1)
    regions: list[str] = Field(default_factory=list)
    seeding: SeedingMode = SeedingMode.ranked
    scoring: ScoringConfig | None = None
    rounds: list[str] = Field(default_factory=list)
    n_rounds: int | None = None
    byes: list[str] = Field(default_factory=list)
    groups: GroupConfig | None = None
    knockout: KnockoutConfig | None = None
    grand_final: bool = False


# ---------------------------------------------------------------------------
# Matchup / result models
# ---------------------------------------------------------------------------


class MatchupContext(BaseModel):
    """Context for a single matchup within a competition."""

    slot: str
    round_num: int
    entity_a: str
    entity_b: str
    prob_a: float
    model_probs: dict[str, float] = Field(default_factory=dict)
    model_agreement: float = 1.0
    pick: str = ""
    strategy: str = ""
    upset: bool = False


class CompetitionResult(BaseModel):
    """Aggregate result of a competition simulation."""

    picks: dict[str, str] = Field(default_factory=dict)
    matchups: dict[str, Any] = Field(default_factory=dict)
    expected_points: float = 0.0
    win_probability: float = 0.0
    top10_probability: float = 0.0
    strategy: str = ""


class StandingsEntry(BaseModel):
    """A single row in a standings table (round-robin / group stage)."""

    entity: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0
    goal_diff: float = 0.0


# ---------------------------------------------------------------------------
# Structure / scoring
# ---------------------------------------------------------------------------


class CompetitionStructure(BaseModel):
    """Pre-computed structural layout of a competition bracket."""

    config: CompetitionConfig
    slots: list[str] = Field(default_factory=list)
    slot_matchups: dict[str, tuple[str, str]] = Field(default_factory=dict)
    slot_to_round: dict[str, int] = Field(default_factory=dict)
    round_slots: dict[int, list[str]] = Field(default_factory=dict)
    seed_to_entity: dict[str, str] = Field(default_factory=dict)
    entity_to_seed: dict[str, str] = Field(default_factory=dict)


class ScoreResult(BaseModel):
    """Scoring output after evaluating picks against actual results."""

    total_points: float
    round_points: dict[str, float] = Field(default_factory=dict)
    round_correct: dict[str, int] = Field(default_factory=dict)
    round_total: dict[str, int] = Field(default_factory=dict)
    picks_detail: list[dict[str, Any]] = Field(default_factory=list)


class AdjustmentConfig(BaseModel):
    """User-specified probability adjustments and overrides."""

    entity_multipliers: dict[str, float] = Field(default_factory=dict)
    external_weight: float = 0.0
    probability_overrides: dict[str, float] = Field(default_factory=dict)
