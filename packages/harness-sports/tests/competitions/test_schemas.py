"""Tests for competition engine schemas."""

import pytest
from pydantic import ValidationError

from harnessml.sports.competitions.schemas import (
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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestCompetitionFormat:
    def test_values(self):
        assert CompetitionFormat.single_elimination == "single_elimination"
        assert CompetitionFormat.double_elimination == "double_elimination"
        assert CompetitionFormat.round_robin == "round_robin"
        assert CompetitionFormat.swiss == "swiss"
        assert CompetitionFormat.group_knockout == "group_knockout"

    def test_invalid(self):
        with pytest.raises(ValueError):
            CompetitionFormat("invalid_format")


class TestSeedingMode:
    def test_values(self):
        assert SeedingMode.ranked == "ranked"
        assert SeedingMode.random == "random"
        assert SeedingMode.manual == "manual"

    def test_invalid(self):
        with pytest.raises(ValueError):
            SeedingMode("invalid_mode")


# ---------------------------------------------------------------------------
# ScoringConfig
# ---------------------------------------------------------------------------


class TestScoringConfig:
    def test_per_round_valid(self):
        sc = ScoringConfig(type="per_round", values=[10, 20, 40])
        assert sc.type == "per_round"
        assert sc.values == [10, 20, 40]

    def test_per_round_empty_values_fails(self):
        with pytest.raises(ValidationError, match="non-empty"):
            ScoringConfig(type="per_round", values=[])

    def test_per_round_default_values_fails(self):
        with pytest.raises(ValidationError, match="non-empty"):
            ScoringConfig(type="per_round")

    def test_points_valid(self):
        sc = ScoringConfig(type="points", win=3.0, draw=1.0, loss=0.0)
        assert sc.win == 3.0
        assert sc.draw == 1.0

    def test_points_zero_win_fails(self):
        with pytest.raises(ValidationError, match="non-zero"):
            ScoringConfig(type="points", win=0.0)


# ---------------------------------------------------------------------------
# GroupConfig / KnockoutConfig
# ---------------------------------------------------------------------------


class TestGroupConfig:
    def test_defaults(self):
        gc = GroupConfig()
        assert gc.n_groups == 1
        assert gc.group_size == 4
        assert gc.format == CompetitionFormat.round_robin
        assert gc.advance == 2
        assert gc.scoring is None

    def test_with_scoring(self):
        sc = ScoringConfig(type="points", win=3.0)
        gc = GroupConfig(n_groups=4, scoring=sc)
        assert gc.n_groups == 4
        assert gc.scoring is not None
        assert gc.scoring.win == 3.0


class TestKnockoutConfig:
    def test_defaults(self):
        kc = KnockoutConfig()
        assert kc.format == CompetitionFormat.single_elimination
        assert kc.scoring is None

    def test_custom(self):
        kc = KnockoutConfig(
            format=CompetitionFormat.double_elimination,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        assert kc.format == CompetitionFormat.double_elimination


# ---------------------------------------------------------------------------
# CompetitionConfig
# ---------------------------------------------------------------------------


class TestCompetitionConfig:
    def test_defaults(self):
        cc = CompetitionConfig()
        assert cc.format == CompetitionFormat.single_elimination
        assert cc.n_participants == 2
        assert cc.seeding == SeedingMode.ranked
        assert cc.regions == []
        assert cc.grand_final is False

    def test_n_participants_min(self):
        cc = CompetitionConfig(n_participants=1)
        assert cc.n_participants == 1

    def test_n_participants_zero_fails(self):
        with pytest.raises(ValidationError):
            CompetitionConfig(n_participants=0)

    def test_n_participants_negative_fails(self):
        with pytest.raises(ValidationError):
            CompetitionConfig(n_participants=-1)

    def test_full_config(self):
        cc = CompetitionConfig(
            format=CompetitionFormat.group_knockout,
            n_participants=16,
            regions=["East", "West"],
            seeding=SeedingMode.manual,
            rounds=["Group", "QF", "SF", "F"],
            n_rounds=4,
            byes=["team_1"],
            groups=GroupConfig(n_groups=4),
            knockout=KnockoutConfig(),
            grand_final=True,
        )
        assert cc.n_participants == 16
        assert len(cc.regions) == 2
        assert cc.groups.n_groups == 4


# ---------------------------------------------------------------------------
# MatchupContext
# ---------------------------------------------------------------------------


class TestMatchupContext:
    def test_construction(self):
        mc = MatchupContext(
            slot="R1G1",
            round_num=1,
            entity_a="team_a",
            entity_b="team_b",
            prob_a=0.65,
        )
        assert mc.slot == "R1G1"
        assert mc.entity_a == "team_a"
        assert mc.entity_b == "team_b"
        assert mc.prob_a == 0.65
        assert mc.model_probs == {}
        assert mc.model_agreement == 1.0
        assert mc.pick == ""
        assert mc.upset is False

    def test_with_model_probs(self):
        mc = MatchupContext(
            slot="R1G1",
            round_num=1,
            entity_a="a",
            entity_b="b",
            prob_a=0.7,
            model_probs={"xgb": 0.72, "lgb": 0.68},
            model_agreement=0.9,
            pick="a",
            strategy="chalk",
            upset=False,
        )
        assert mc.model_probs["xgb"] == 0.72
        assert mc.strategy == "chalk"

    def test_missing_required_fields_fails(self):
        with pytest.raises(ValidationError):
            MatchupContext(slot="R1G1", round_num=1)


# ---------------------------------------------------------------------------
# CompetitionResult
# ---------------------------------------------------------------------------


class TestCompetitionResult:
    def test_defaults(self):
        cr = CompetitionResult()
        assert cr.picks == {}
        assert cr.matchups == {}
        assert cr.expected_points == 0.0
        assert cr.win_probability == 0.0
        assert cr.top10_probability == 0.0
        assert cr.strategy == ""

    def test_with_data(self):
        cr = CompetitionResult(
            picks={"R1G1": "team_a", "R1G2": "team_b"},
            expected_points=42.5,
            strategy="upset_heavy",
        )
        assert len(cr.picks) == 2
        assert cr.expected_points == 42.5


# ---------------------------------------------------------------------------
# StandingsEntry
# ---------------------------------------------------------------------------


class TestStandingsEntry:
    def test_defaults(self):
        se = StandingsEntry(entity="team_x")
        assert se.entity == "team_x"
        assert se.wins == 0
        assert se.losses == 0
        assert se.draws == 0
        assert se.points == 0.0
        assert se.goal_diff == 0.0

    def test_with_stats(self):
        se = StandingsEntry(
            entity="team_y", wins=3, losses=1, draws=2, points=11.0, goal_diff=5.0
        )
        assert se.wins == 3
        assert se.points == 11.0

    def test_missing_entity_fails(self):
        with pytest.raises(ValidationError):
            StandingsEntry()


# ---------------------------------------------------------------------------
# CompetitionStructure
# ---------------------------------------------------------------------------


class TestCompetitionStructure:
    def test_minimal(self):
        cfg = CompetitionConfig(n_participants=4)
        cs = CompetitionStructure(config=cfg)
        assert cs.config.n_participants == 4
        assert cs.slots == []
        assert cs.slot_matchups == {}

    def test_with_data(self):
        cfg = CompetitionConfig(n_participants=4)
        cs = CompetitionStructure(
            config=cfg,
            slots=["R1G1", "R1G2", "R2G1"],
            slot_matchups={"R1G1": ("s1", "s4"), "R1G2": ("s2", "s3")},
            slot_to_round={"R1G1": 1, "R1G2": 1, "R2G1": 2},
            round_slots={1: ["R1G1", "R1G2"], 2: ["R2G1"]},
            seed_to_entity={"s1": "team_a", "s2": "team_b"},
            entity_to_seed={"team_a": "s1", "team_b": "s2"},
        )
        assert len(cs.slots) == 3
        assert cs.slot_matchups["R1G1"] == ("s1", "s4")
        assert cs.round_slots[1] == ["R1G1", "R1G2"]

    def test_missing_config_fails(self):
        with pytest.raises(ValidationError):
            CompetitionStructure()


# ---------------------------------------------------------------------------
# ScoreResult
# ---------------------------------------------------------------------------


class TestScoreResult:
    def test_construction(self):
        sr = ScoreResult(total_points=120.0)
        assert sr.total_points == 120.0
        assert sr.round_points == {}
        assert sr.round_correct == {}
        assert sr.round_total == {}
        assert sr.picks_detail == []

    def test_with_detail(self):
        sr = ScoreResult(
            total_points=80.0,
            round_points={"R1": 40.0, "R2": 40.0},
            round_correct={"R1": 4, "R2": 2},
            round_total={"R1": 4, "R2": 2},
            picks_detail=[{"slot": "R1G1", "correct": True}],
        )
        assert sr.round_points["R1"] == 40.0
        assert len(sr.picks_detail) == 1

    def test_missing_total_fails(self):
        with pytest.raises(ValidationError):
            ScoreResult()


# ---------------------------------------------------------------------------
# AdjustmentConfig
# ---------------------------------------------------------------------------


class TestAdjustmentConfig:
    def test_defaults(self):
        ac = AdjustmentConfig()
        assert ac.entity_multipliers == {}
        assert ac.external_weight == 0.0
        assert ac.probability_overrides == {}

    def test_with_data(self):
        ac = AdjustmentConfig(
            entity_multipliers={"team_a": 1.1, "team_b": 0.9},
            external_weight=0.3,
            probability_overrides={"R1G1": 0.8},
        )
        assert ac.entity_multipliers["team_a"] == 1.1
        assert ac.external_weight == 0.3


# ---------------------------------------------------------------------------
# Re-exports from __init__
# ---------------------------------------------------------------------------


class TestReExports:
    def test_all_importable_from_init(self):
        from harnessml.sports.competitions import (
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

        assert CompetitionFormat.single_elimination == "single_elimination"
        assert SeedingMode.ranked == "ranked"
