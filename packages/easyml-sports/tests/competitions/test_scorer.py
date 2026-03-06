"""Tests for CompetitionScorer."""

from __future__ import annotations

import pytest

from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionStructure,
    ScoringConfig,
    StandingsEntry,
)
from easyml.sports.competitions.scorer import CompetitionScorer
from easyml.sports.competitions.structure import build_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_4team_bracket() -> tuple[CompetitionStructure, ScoringConfig]:
    """Build a 4-team single-elimination bracket with 10/20 scoring."""
    config = CompetitionConfig(
        format=CompetitionFormat.single_elimination,
        n_participants=4,
    )
    seed_map = {"S1": "A", "S2": "B", "S3": "C", "S4": "D"}
    structure = build_structure(config, seed_map)
    scoring = ScoringConfig(type="per_round", values=[10.0, 20.0])
    return structure, scoring


def _make_8team_bracket() -> tuple[CompetitionStructure, ScoringConfig]:
    """Build an 8-team single-elimination bracket with 10/20/40 scoring."""
    config = CompetitionConfig(
        format=CompetitionFormat.single_elimination,
        n_participants=8,
    )
    seed_map = {f"S{i}": chr(64 + i) for i in range(1, 9)}  # A-H
    structure = build_structure(config, seed_map)
    scoring = ScoringConfig(type="per_round", values=[10.0, 20.0, 40.0])
    return structure, scoring


# ---------------------------------------------------------------------------
# get_round_points
# ---------------------------------------------------------------------------


class TestGetRoundPoints:
    def test_returns_correct_value_per_round(self):
        scoring = ScoringConfig(type="per_round", values=[10.0, 20.0, 40.0])
        scorer = CompetitionScorer(scoring)
        assert scorer.get_round_points(1) == 10.0
        assert scorer.get_round_points(2) == 20.0
        assert scorer.get_round_points(3) == 40.0

    def test_clamps_to_last_value_for_excess_rounds(self):
        scoring = ScoringConfig(type="per_round", values=[10.0, 20.0])
        scorer = CompetitionScorer(scoring)
        assert scorer.get_round_points(5) == 20.0

    def test_round_zero_uses_first_value(self):
        scoring = ScoringConfig(type="per_round", values=[10.0, 20.0])
        scorer = CompetitionScorer(scoring)
        assert scorer.get_round_points(0) == 10.0


# ---------------------------------------------------------------------------
# score_bracket — perfect bracket
# ---------------------------------------------------------------------------


class TestPerfectBracket:
    def test_perfect_bracket_scores_max(self):
        structure, scoring = _make_4team_bracket()
        scorer = CompetitionScorer(scoring)

        # Determine actuals by simulating a bracket:
        # 4-team: R1G1 (S1 vs S4), R1G2 (S2 vs S3), R2G1 (winner R1G1 vs winner R1G2)
        actuals = {}
        picks = {}
        # R1: top seeds win
        for slot in structure.round_slots[1]:
            a_ref, b_ref = structure.slot_matchups[slot]
            winner = structure.seed_to_entity.get(a_ref, a_ref)
            actuals[slot] = winner
            picks[slot] = winner
        # R2: pick same as actual
        for slot in structure.round_slots[2]:
            actuals[slot] = "A"
            picks[slot] = "A"

        result = scorer.score_bracket(picks, actuals, structure)
        max_pts = scorer.max_possible_points(structure)
        assert result.total_points == max_pts
        assert result.total_points > 0

    def test_perfect_8team_bracket(self):
        structure, scoring = _make_8team_bracket()
        scorer = CompetitionScorer(scoring)
        max_pts = scorer.max_possible_points(structure)

        # All top seeds win every round
        actuals = {}
        picks = {}
        # R1
        for slot in structure.round_slots[1]:
            a_ref, _ = structure.slot_matchups[slot]
            winner = structure.seed_to_entity.get(a_ref, a_ref)
            actuals[slot] = winner
            picks[slot] = winner
        # R2
        for slot in structure.round_slots[2]:
            actuals[slot] = "A"
            picks[slot] = "A"
        # R3 (final)
        for slot in structure.round_slots[3]:
            actuals[slot] = "A"
            picks[slot] = "A"

        result = scorer.score_bracket(picks, actuals, structure)
        assert result.total_points == max_pts
        # 4*10 + 2*20 + 1*40 = 120
        assert max_pts == 120.0


# ---------------------------------------------------------------------------
# score_bracket — all wrong
# ---------------------------------------------------------------------------


class TestAllWrongBracket:
    def test_all_wrong_scores_zero(self):
        structure, scoring = _make_4team_bracket()
        scorer = CompetitionScorer(scoring)

        actuals = {}
        picks = {}
        for slot in structure.round_slots[1]:
            a_ref, b_ref = structure.slot_matchups[slot]
            winner = structure.seed_to_entity.get(a_ref, a_ref)
            loser = structure.seed_to_entity.get(b_ref, b_ref)
            actuals[slot] = winner
            picks[slot] = loser  # wrong pick
        for slot in structure.round_slots[2]:
            actuals[slot] = "A"
            picks[slot] = "D"  # wrong pick

        result = scorer.score_bracket(picks, actuals, structure)
        assert result.total_points == 0.0
        for rk, count in result.round_correct.items():
            assert count == 0


# ---------------------------------------------------------------------------
# score_bracket — partial correctness
# ---------------------------------------------------------------------------


class TestPartialBracket:
    def test_partial_scores_correctly(self):
        structure, scoring = _make_4team_bracket()
        scorer = CompetitionScorer(scoring)

        actuals = {}
        picks = {}
        r1_slots = structure.round_slots[1]
        # First R1 game correct, second wrong
        a0, b0 = structure.slot_matchups[r1_slots[0]]
        actuals[r1_slots[0]] = structure.seed_to_entity.get(a0, a0)
        picks[r1_slots[0]] = structure.seed_to_entity.get(a0, a0)  # correct

        a1, b1 = structure.slot_matchups[r1_slots[1]]
        actuals[r1_slots[1]] = structure.seed_to_entity.get(a1, a1)
        picks[r1_slots[1]] = structure.seed_to_entity.get(b1, b1)  # wrong

        # R2: wrong
        for slot in structure.round_slots[2]:
            actuals[slot] = "A"
            picks[slot] = "D"

        result = scorer.score_bracket(picks, actuals, structure)
        assert result.total_points == 10.0  # only 1 correct R1 pick
        assert result.round_correct["1"] == 1
        assert result.round_correct["2"] == 0


# ---------------------------------------------------------------------------
# Custom scoring values
# ---------------------------------------------------------------------------


class TestCustomScoringValues:
    def test_custom_values(self):
        config = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=4,
        )
        seed_map = {"S1": "A", "S2": "B", "S3": "C", "S4": "D"}
        structure = build_structure(config, seed_map)
        scoring = ScoringConfig(type="per_round", values=[1.0, 5.0])
        scorer = CompetitionScorer(scoring)

        # Perfect bracket
        actuals = {}
        picks = {}
        for slot in structure.round_slots[1]:
            a_ref, _ = structure.slot_matchups[slot]
            winner = structure.seed_to_entity.get(a_ref, a_ref)
            actuals[slot] = winner
            picks[slot] = winner
        for slot in structure.round_slots[2]:
            actuals[slot] = "A"
            picks[slot] = "A"

        result = scorer.score_bracket(picks, actuals, structure)
        # 2 * 1 + 1 * 5 = 7
        assert result.total_points == 7.0

    def test_max_possible_with_custom_values(self):
        config = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=4,
        )
        seed_map = {"S1": "A", "S2": "B", "S3": "C", "S4": "D"}
        structure = build_structure(config, seed_map)
        scoring = ScoringConfig(type="per_round", values=[1.0, 5.0])
        scorer = CompetitionScorer(scoring)
        assert scorer.max_possible_points(structure) == 7.0


# ---------------------------------------------------------------------------
# Standings scoring with displacement
# ---------------------------------------------------------------------------


class TestStandingsScoring:
    def test_perfect_standings(self):
        scorer = CompetitionScorer(ScoringConfig(type="per_round", values=[10.0]))
        standings = [
            StandingsEntry(entity="A", wins=3),
            StandingsEntry(entity="B", wins=2),
            StandingsEntry(entity="C", wins=1),
        ]
        result = scorer.score_standings(standings, standings)
        assert result.total_points == 0.0  # no displacement

    def test_reversed_standings(self):
        scorer = CompetitionScorer(ScoringConfig(type="per_round", values=[10.0]))
        actual = [
            StandingsEntry(entity="A", wins=3),
            StandingsEntry(entity="B", wins=2),
            StandingsEntry(entity="C", wins=1),
        ]
        predicted = list(reversed(actual))
        result = scorer.score_standings(predicted, actual)
        # C at 0 should be at 2: disp 2, B at 1 stays: disp 0, A at 2 should be 0: disp 2
        assert result.total_points == -4.0

    def test_partial_displacement(self):
        scorer = CompetitionScorer(ScoringConfig(type="per_round", values=[10.0]))
        actual = [
            StandingsEntry(entity="A"),
            StandingsEntry(entity="B"),
            StandingsEntry(entity="C"),
            StandingsEntry(entity="D"),
        ]
        # Swap B and C
        predicted = [
            StandingsEntry(entity="A"),
            StandingsEntry(entity="C"),
            StandingsEntry(entity="B"),
            StandingsEntry(entity="D"),
        ]
        result = scorer.score_standings(predicted, actual)
        # A: 0, C: |1-2|=1, B: |2-1|=1, D: 0 => total -2
        assert result.total_points == -2.0


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_empty_picks_and_actuals(self):
        structure, scoring = _make_4team_bracket()
        scorer = CompetitionScorer(scoring)
        result = scorer.score_bracket({}, {}, structure)
        assert result.total_points == 0.0
        assert result.picks_detail == []

    def test_empty_standings(self):
        scorer = CompetitionScorer(ScoringConfig(type="per_round", values=[10.0]))
        result = scorer.score_standings([], [])
        assert result.total_points == 0.0


# ---------------------------------------------------------------------------
# picks_detail structure
# ---------------------------------------------------------------------------


class TestPicksDetail:
    def test_detail_has_required_keys(self):
        structure, scoring = _make_4team_bracket()
        scorer = CompetitionScorer(scoring)

        actuals = {}
        picks = {}
        for slot in structure.round_slots[1]:
            a_ref, _ = structure.slot_matchups[slot]
            winner = structure.seed_to_entity.get(a_ref, a_ref)
            actuals[slot] = winner
            picks[slot] = winner
        for slot in structure.round_slots[2]:
            actuals[slot] = "A"
            picks[slot] = "A"

        result = scorer.score_bracket(picks, actuals, structure)
        assert len(result.picks_detail) > 0
        for detail in result.picks_detail:
            assert "slot" in detail
            assert "round" in detail
            assert "pick" in detail
            assert "actual" in detail
            assert "correct" in detail
            assert "points" in detail

    def test_standings_detail_has_required_keys(self):
        scorer = CompetitionScorer(ScoringConfig(type="per_round", values=[10.0]))
        actual = [StandingsEntry(entity="A"), StandingsEntry(entity="B")]
        result = scorer.score_standings(actual, actual)
        for detail in result.picks_detail:
            assert "entity" in detail
            assert "predicted_rank" in detail
            assert "actual_rank" in detail
            assert "displacement" in detail


# ---------------------------------------------------------------------------
# max_possible_points
# ---------------------------------------------------------------------------


class TestMaxPossiblePoints:
    def test_4team(self):
        structure, scoring = _make_4team_bracket()
        scorer = CompetitionScorer(scoring)
        # 2 R1 games * 10 + 1 R2 game * 20 = 40
        assert scorer.max_possible_points(structure) == 40.0

    def test_8team(self):
        structure, scoring = _make_8team_bracket()
        scorer = CompetitionScorer(scoring)
        # 4*10 + 2*20 + 1*40 = 120
        assert scorer.max_possible_points(structure) == 120.0
