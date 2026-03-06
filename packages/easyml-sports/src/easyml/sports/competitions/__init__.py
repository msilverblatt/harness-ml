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
from easyml.sports.competitions.confidence import (
    compute_feature_outliers,
    compute_model_disagreement,
    generate_confidence_report,
)
from easyml.sports.competitions.explainer import CompetitionExplainer
from easyml.sports.competitions.optimizer import (
    BUILTIN_STRATEGIES,
    CompetitionOptimizer,
    StrategyFn,
)
from easyml.sports.competitions.export import (
    export_analysis_report,
    export_bracket_markdown,
    export_csv,
    export_json,
    export_standings_markdown,
)
from easyml.sports.competitions.scorer import CompetitionScorer
from easyml.sports.competitions.simulator import CompetitionSimulator
from easyml.sports.competitions.structure import build_structure

__all__ = [
    "AdjustmentConfig",
    "BUILTIN_STRATEGIES",
    "CompetitionConfig",
    "CompetitionExplainer",
    "CompetitionFormat",
    "CompetitionOptimizer",
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
    "StrategyFn",
    "apply_adjustments",
    "build_structure",
    "compute_feature_outliers",
    "compute_model_disagreement",
    "export_analysis_report",
    "export_bracket_markdown",
    "export_csv",
    "export_json",
    "export_standings_markdown",
    "generate_confidence_report",
]
