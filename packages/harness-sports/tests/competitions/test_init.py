"""Tests that all public symbols are importable from harnessml.sports.competitions."""

from __future__ import annotations

import harnessml.sports.competitions as comp


# Every symbol that should be re-exported from the package
EXPECTED_SYMBOLS = [
    # schemas.py
    "CompetitionFormat",
    "SeedingMode",
    "ScoringConfig",
    "GroupConfig",
    "KnockoutConfig",
    "CompetitionConfig",
    "MatchupContext",
    "CompetitionResult",
    "StandingsEntry",
    "CompetitionStructure",
    "ScoreResult",
    "AdjustmentConfig",
    # structure.py
    "build_structure",
    # simulator.py
    "CompetitionSimulator",
    # scorer.py
    "CompetitionScorer",
    # adjustments.py
    "apply_adjustments",
    # optimizer.py
    "CompetitionOptimizer",
    "BUILTIN_STRATEGIES",
    "StrategyFn",
    # explainer.py
    "CompetitionExplainer",
    # confidence.py
    "compute_feature_outliers",
    "compute_model_disagreement",
    "generate_confidence_report",
    # export.py
    "export_bracket_markdown",
    "export_standings_markdown",
    "export_json",
    "export_csv",
    "export_analysis_report",
]


class TestPublicAPI:
    """Verify every expected symbol is importable from the competitions package."""

    def test_all_symbols_importable(self):
        for name in EXPECTED_SYMBOLS:
            assert hasattr(comp, name), f"Missing symbol: {name}"

    def test_all_in__all__(self):
        for name in EXPECTED_SYMBOLS:
            assert name in comp.__all__, f"Symbol {name} not in __all__"

    def test_no_extra_in__all__(self):
        """Ensure __all__ contains only expected symbols (no stale leftovers)."""
        extra = set(comp.__all__) - set(EXPECTED_SYMBOLS)
        assert not extra, f"Unexpected symbols in __all__: {extra}"
