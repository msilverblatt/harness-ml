"""Competition engine schemas and data models for bracket/tournament simulation."""

from easyml.sports.competitions.schemas import (
    AdjustmentConfig,
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    CompetitionStructure,
    GroupConfig,
    KnockoutConfig,
    MatchupContext,
    ScoreResult,
    ScoringConfig,
    SeedingMode,
    StandingsEntry,
)
from easyml.sports.competitions.adjustments import apply_adjustments
from easyml.sports.competitions.scorer import CompetitionScorer
from easyml.sports.competitions.simulator import CompetitionSimulator
from easyml.sports.competitions.structure import build_structure

__all__ = [
    "AdjustmentConfig",
    "CompetitionConfig",
    "CompetitionFormat",
    "CompetitionResult",
    "CompetitionScorer",
    "CompetitionSimulator",
    "CompetitionStructure",
    "GroupConfig",
    "KnockoutConfig",
    "MatchupContext",
    "ScoreResult",
    "ScoringConfig",
    "SeedingMode",
    "StandingsEntry",
    "apply_adjustments",
    "build_structure",
]
