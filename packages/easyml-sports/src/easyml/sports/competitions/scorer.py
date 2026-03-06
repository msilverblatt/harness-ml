"""Configurable competition scorer.

Scores bracket picks and standings predictions against actual results,
using a pluggable :class:`ScoringConfig` for per-round point values.
"""

from __future__ import annotations

from typing import Any

from easyml.sports.competitions.schemas import (
    CompetitionStructure,
    ScoreResult,
    ScoringConfig,
    StandingsEntry,
)


class CompetitionScorer:
    """Score bracket picks or standings predictions against actuals."""

    def __init__(self, scoring: ScoringConfig) -> None:
        self._scoring = scoring

    # ------------------------------------------------------------------
    # Per-round point lookup
    # ------------------------------------------------------------------

    def get_round_points(self, round_num: int) -> float:
        """Points for a correct pick in the given round.

        Uses ``scoring.values`` list (0-indexed by ``round_num - 1``).
        If the round exceeds the length of the values list, the last
        value is used.

        Parameters
        ----------
        round_num:
            1-based round number.
        """
        values = self._scoring.values
        if not values:
            return 0.0
        idx = min(round_num - 1, len(values) - 1)
        return values[max(idx, 0)]

    # ------------------------------------------------------------------
    # Bracket scoring
    # ------------------------------------------------------------------

    def score_bracket(
        self,
        picks: dict[str, str],
        actuals: dict[str, str],
        structure: CompetitionStructure,
    ) -> ScoreResult:
        """Score bracket picks against actual results.

        Parameters
        ----------
        picks:
            Mapping of slot -> predicted winner entity ID.
        actuals:
            Mapping of slot -> actual winner entity ID.
        structure:
            The competition structure describing rounds and slots.

        Returns
        -------
        ScoreResult
        """
        total_points = 0.0
        round_points: dict[str, float] = {}
        round_correct: dict[str, int] = {}
        round_total: dict[str, int] = {}
        picks_detail: list[dict[str, Any]] = []

        for slot in structure.slots:
            round_num = structure.slot_to_round.get(slot, 0)
            round_key = str(round_num)

            # Only count slots present in actuals
            if slot not in actuals:
                continue

            round_total[round_key] = round_total.get(round_key, 0) + 1

            predicted = picks.get(slot, "")
            actual = actuals.get(slot, "")
            correct = predicted == actual and predicted != ""
            pts = self.get_round_points(round_num) if correct else 0.0

            total_points += pts
            round_points[round_key] = round_points.get(round_key, 0.0) + pts
            round_correct[round_key] = round_correct.get(round_key, 0) + (1 if correct else 0)

            picks_detail.append({
                "slot": slot,
                "round": round_num,
                "pick": predicted,
                "actual": actual,
                "correct": correct,
                "points": pts,
            })

        # Ensure all round keys exist even if 0
        for rk in round_total:
            round_points.setdefault(rk, 0.0)
            round_correct.setdefault(rk, 0)

        return ScoreResult(
            total_points=total_points,
            round_points=round_points,
            round_correct=round_correct,
            round_total=round_total,
            picks_detail=picks_detail,
        )

    # ------------------------------------------------------------------
    # Standings scoring (rank displacement)
    # ------------------------------------------------------------------

    def score_standings(
        self,
        predicted: list[StandingsEntry],
        actual: list[StandingsEntry],
    ) -> ScoreResult:
        """Score predicted standings against actual using rank displacement.

        Each entity's rank displacement (absolute difference between
        predicted rank and actual rank) is summed. The total_points is
        the *negative* total displacement so that higher is better
        (perfect = 0.0).

        Parameters
        ----------
        predicted:
            Predicted standings in order (index = rank).
        actual:
            Actual standings in order (index = rank).
        """
        if not predicted or not actual:
            return ScoreResult(total_points=0.0)

        actual_rank: dict[str, int] = {
            entry.entity: idx for idx, entry in enumerate(actual)
        }

        total_displacement = 0.0
        picks_detail: list[dict[str, Any]] = []

        for pred_rank, entry in enumerate(predicted):
            act_rank = actual_rank.get(entry.entity)
            if act_rank is None:
                # Entity not found in actuals; max displacement
                displacement = len(actual)
            else:
                displacement = abs(pred_rank - act_rank)
            total_displacement += displacement
            picks_detail.append({
                "entity": entry.entity,
                "predicted_rank": pred_rank,
                "actual_rank": act_rank if act_rank is not None else -1,
                "displacement": displacement,
            })

        return ScoreResult(
            total_points=-total_displacement,
            picks_detail=picks_detail,
        )

    # ------------------------------------------------------------------
    # Max possible points
    # ------------------------------------------------------------------

    def max_possible_points(self, structure: CompetitionStructure) -> float:
        """Maximum possible points for a perfect bracket.

        Sums the per-round point value multiplied by number of slots
        in each round.
        """
        total = 0.0
        for round_num, slots in structure.round_slots.items():
            pts_per_game = self.get_round_points(round_num)
            total += pts_per_game * len(slots)
        return total
