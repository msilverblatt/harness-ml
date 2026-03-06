"""Pool-size-aware bracket optimizer with pluggable strategies.

Generates diverse candidates via strategy-based generation (chalk, near_chalk,
random_sim, contrarian, late_contrarian, champion_anchor) with a continuous
pool-size-dependent strategy mix. Scores with vectorized beat-max against
pool-sized opponent field.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from easyml.sports.competitions.schemas import (
    CompetitionResult,
    ScoringConfig,
)

if TYPE_CHECKING:
    from easyml.sports.competitions.simulator import CompetitionSimulator


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StrategyFn(Protocol):
    """Protocol for bracket generation strategies.

    A strategy function takes a simulator and RNG, returning a bracket
    (dict mapping slot name to winner entity ID).
    """

    def __call__(
        self,
        simulator: CompetitionSimulator,
        rng: np.random.Generator,
        **kwargs: object,
    ) -> dict[str, str]: ...


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _simulate_with_boost(
    sim: CompetitionSimulator,
    rng: np.random.Generator,
    upset_boost: float = 0.0,
    force_winners: set[str] | None = None,
    ramp_boost: bool = False,
) -> dict[str, str]:
    """Simulate a bracket with optional probability compression and forced winners.

    Parameters
    ----------
    sim:
        Competition simulator instance.
    rng:
        Numpy random generator.
    upset_boost:
        Compress probabilities toward 0.5 (0 = raw, 1 = coin flip).
    force_winners:
        Set of entity IDs that always win their matchups.
    ramp_boost:
        If True, scale upset_boost by round (0 in round 1, full in max round).
    """
    if force_winners is None:
        force_winners = set()

    results: dict[str, str] = {}
    max_round = max(sim.structure.round_slots.keys())
    min_round = min(sim.structure.round_slots.keys())
    n_rounds = max_round - min_round

    for r in range(min_round, max_round + 1):
        if r not in sim.structure.round_slots:
            continue
        for slot in sim.structure.round_slots[r]:
            a_ref, b_ref = sim.structure.slot_matchups[slot]
            ea = sim._resolve_entity(a_ref, results)
            eb = sim._resolve_entity(b_ref, results)

            if ea == eb or ea == "BYE" or eb == "BYE":
                winner = eb if ea == "BYE" else ea
                results[slot] = winner
                continue

            if ea in force_winners:
                results[slot] = ea
                continue
            if eb in force_winners:
                results[slot] = eb
                continue

            prob_a = sim.get_win_prob(ea, eb)

            effective_boost = upset_boost
            if ramp_boost and n_rounds > 0:
                round_frac = (r - min_round) / n_rounds
                effective_boost = upset_boost * round_frac

            if effective_boost > 0:
                prob_a = 0.5 + (prob_a - 0.5) * (1.0 - effective_boost)

            if rng.random() < prob_a:
                results[slot] = ea
            else:
                results[slot] = eb

    return results


# ---------------------------------------------------------------------------
# Built-in strategies
# ---------------------------------------------------------------------------


def chalk_strategy(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    **kwargs: object,
) -> dict[str, str]:
    """Always pick the favorite in every matchup."""
    return simulator.pick_most_likely()


def near_chalk_strategy(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    **kwargs: object,
) -> dict[str, str]:
    """Chalk but flip close matchups (underdog >40%) with 30% probability."""
    results: dict[str, str] = {}
    max_round = max(simulator.structure.round_slots.keys())
    min_round = min(simulator.structure.round_slots.keys())

    for r in range(min_round, max_round + 1):
        if r not in simulator.structure.round_slots:
            continue
        for slot in simulator.structure.round_slots[r]:
            a_ref, b_ref = simulator.structure.slot_matchups[slot]
            ea = simulator._resolve_entity(a_ref, results)
            eb = simulator._resolve_entity(b_ref, results)

            if ea == eb or ea == "BYE" or eb == "BYE":
                winner = eb if ea == "BYE" else ea
                results[slot] = winner
                continue

            prob_a = simulator.get_win_prob(ea, eb)
            underdog_prob = 1.0 - prob_a if prob_a >= 0.5 else prob_a

            if underdog_prob > 0.4 and rng.random() < 0.3:
                # Flip to underdog
                winner = eb if prob_a >= 0.5 else ea
            elif prob_a >= 0.5:
                winner = ea
            else:
                winner = eb

            results[slot] = winner

    return results


def random_sim_strategy(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    **kwargs: object,
) -> dict[str, str]:
    """Run a single stochastic simulation."""
    return simulator.simulate_once(rng)


def contrarian_strategy(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    **kwargs: object,
) -> dict[str, str]:
    """Compress probabilities toward 0.5 (uniform upset boost)."""
    upset_boost = kwargs.get("upset_boost", 0.3)
    if not isinstance(upset_boost, (int, float)):
        upset_boost = 0.3
    return _simulate_with_boost(sim=simulator, rng=rng, upset_boost=float(upset_boost))


def late_contrarian_strategy(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    **kwargs: object,
) -> dict[str, str]:
    """Ramp upset boost by round (0 in R1, full in max round)."""
    upset_boost = kwargs.get("upset_boost", 0.5)
    if not isinstance(upset_boost, (int, float)):
        upset_boost = 0.5
    return _simulate_with_boost(
        sim=simulator, rng=rng, upset_boost=float(upset_boost), ramp_boost=True
    )


def champion_anchor_strategy(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    **kwargs: object,
) -> dict[str, str]:
    """Force a specific entity to win all its games."""
    champion = kwargs.get("champion")
    if champion is None:
        # Fall back to chalk if no champion specified
        return simulator.pick_most_likely()
    return _simulate_with_boost(
        sim=simulator, rng=rng, force_winners={str(champion)}
    )


BUILTIN_STRATEGIES: dict[str, StrategyFn] = {
    "chalk": chalk_strategy,
    "near_chalk": near_chalk_strategy,
    "random_sim": random_sim_strategy,
    "contrarian": contrarian_strategy,
    "late_contrarian": late_contrarian_strategy,
    "champion_anchor": champion_anchor_strategy,
}


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class CompetitionOptimizer:
    """Generate optimal brackets for different pool sizes.

    Combines strategy-based candidate generation, vectorized scoring against
    simulated outcomes and opponents, and diversity-based selection.

    Parameters
    ----------
    simulator:
        A :class:`CompetitionSimulator` with loaded probabilities.
    scoring:
        Scoring configuration (per-round point values).
    strategies:
        Optional additional strategies to merge with built-ins.
    """

    def __init__(
        self,
        simulator: CompetitionSimulator,
        scoring: ScoringConfig,
        strategies: dict[str, StrategyFn] | None = None,
    ) -> None:
        self.simulator = simulator
        self.scoring = scoring
        self.strategies: dict[str, StrategyFn] = dict(BUILTIN_STRATEGIES)
        if strategies:
            self.strategies.update(strategies)

        # Build tournament-only slot list (round >= 1) for vectorized scoring
        self._tourney_slots: list[str] = []
        for slot in simulator._slots_sorted:
            rd = simulator.structure.slot_to_round.get(slot, 0)
            if rd >= 1:
                self._tourney_slots.append(slot)

        # Points per tournament slot
        self._slot_points = np.array([
            scoring.values[
                min(
                    simulator.structure.slot_to_round[s] - 1,
                    len(scoring.values) - 1,
                )
            ]
            for s in self._tourney_slots
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_brackets(
        self,
        pool_size: int,
        n_brackets: int,
        n_sims: int = 10_000,
        seed: int = 42,
    ) -> list[CompetitionResult]:
        """Generate diverse brackets optimized for a pool of given size.

        Three-step process:
        1. Generate ~500 candidates via pool-size-dependent strategy mix.
        2. Score all via vectorized comparison against outcomes + opponents.
        3. Select top N diverse brackets (round-weighted overlap).
        """
        rng = np.random.default_rng(seed * 10_000 + pool_size)

        # Vectorized simulation for outcomes
        simulated_outcomes = self.simulator.simulate_many(n_sims, seed=seed)

        # Build outcome matrix
        outcome_matrix = self._brackets_to_matrix(simulated_outcomes)

        # Generate strategy-based candidates
        n_candidates = 500
        candidates = self._generate_candidates(
            rng, n_candidates, pool_size, simulated_outcomes
        )

        # Convert candidates to numpy matrix
        candidate_matrix = self._brackets_to_matrix(
            [c[0] for c in candidates]
        )

        # Generate opponents via simulation
        n_opponents = min(pool_size - 1, 200)
        opponent_outcomes = self.simulator.simulate_many(
            max(n_opponents, 1), seed=seed + 1_000_000
        )
        opponent_matrix = self._brackets_to_matrix(
            opponent_outcomes[:n_opponents]
        )

        # Vectorized scoring
        scored = self._score_candidates_vectorized(
            candidates, candidate_matrix, outcome_matrix, opponent_matrix,
        )

        # Select diverse set
        selected = self._select_diverse(scored, n_brackets)

        return self._enrich_brackets(selected)

    # ------------------------------------------------------------------
    # Strategy mix
    # ------------------------------------------------------------------

    def _get_strategy_mix(self, pool_size: int) -> dict[str, float]:
        """Continuous strategy mix based on log10(pool_size).

        Interpolates from near_chalk-heavy (small pools) to
        contrarian/champion_anchor-heavy (large pools).
        """
        t = max(0.0, min(1.0, (math.log10(max(pool_size, 2)) - 1.0) / 3.0))

        def lerp(a: float, b: float) -> float:
            return a + (b - a) * t

        return {
            "near_chalk": lerp(0.35, 0.00),
            "random_sim": lerp(0.25, 0.10),
            "late_contrarian": lerp(0.25, 0.15),
            "contrarian": lerp(0.05, 0.35),
            "champion_anchor": lerp(0.10, 0.40),
        }

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _generate_candidates(
        self,
        rng: np.random.Generator,
        n_candidates: int,
        pool_size: int,
        simulated_outcomes: list[dict[str, str]],
    ) -> list[tuple[dict[str, str], str]]:
        """Generate candidate brackets using pool-size-dependent strategy mix."""
        candidates: list[tuple[dict[str, str], str]] = []

        # Always include chalk
        candidates.append((self.strategies["chalk"](self.simulator, rng), "chalk"))

        # Compute strategy counts
        mix = self._get_strategy_mix(pool_size)
        remaining = n_candidates - 1
        strategy_counts: dict[str, int] = {}
        allocated = 0
        for strategy, fraction in mix.items():
            count = int(round(fraction * remaining))
            strategy_counts[strategy] = count
            allocated += count
        diff = remaining - allocated
        largest = max(strategy_counts, key=lambda k: strategy_counts[k])
        strategy_counts[largest] += diff

        # near_chalk
        for _ in range(strategy_counts.get("near_chalk", 0)):
            candidates.append((
                self.strategies["near_chalk"](self.simulator, rng),
                "near_chalk",
            ))

        # random_sim
        for _ in range(strategy_counts.get("random_sim", 0)):
            candidates.append((
                self.strategies["random_sim"](self.simulator, rng),
                "random_sim",
            ))

        # late_contrarian: varying boost values
        n_late = strategy_counts.get("late_contrarian", 0)
        late_boost_values = [0.3, 0.5, 0.7]
        for i in range(n_late):
            boost = late_boost_values[i % len(late_boost_values)]
            picks = self.strategies["late_contrarian"](
                self.simulator, rng, upset_boost=boost
            )
            candidates.append((picks, f"late_contrarian_{boost}"))

        # contrarian with varying upset_boost
        n_contrarian = strategy_counts.get("contrarian", 0)
        boost_values = [0.15, 0.3, 0.5]
        for i in range(n_contrarian):
            boost = boost_values[i % len(boost_values)]
            picks = self.strategies["contrarian"](
                self.simulator, rng, upset_boost=boost
            )
            candidates.append((picks, f"contrarian_{boost}"))

        # champion_anchor: sample from simulation champion frequency
        n_anchor = strategy_counts.get("champion_anchor", 0)
        if n_anchor > 0:
            # Find the final slot (last round)
            max_round = max(self.simulator.structure.round_slots.keys())
            final_slots = self.simulator.structure.round_slots[max_round]

            champ_counts: dict[str, int] = {}
            for outcome in simulated_outcomes:
                for slot in final_slots:
                    if slot in outcome:
                        champ = outcome[slot]
                        champ_counts[champ] = champ_counts.get(champ, 0) + 1

            if champ_counts:
                entities = list(champ_counts.keys())
                weights = np.array(
                    [champ_counts[e] for e in entities], dtype=np.float64
                )
                weights /= weights.sum()
                for _ in range(n_anchor):
                    champion = str(rng.choice(entities, p=weights))
                    picks = self.strategies["champion_anchor"](
                        self.simulator, rng, champion=champion
                    )
                    candidates.append((picks, f"anchor_{champion}"))

        return candidates

    # ------------------------------------------------------------------
    # Vectorized scoring
    # ------------------------------------------------------------------

    def _brackets_to_matrix(
        self, bracket_dicts: list[dict[str, str]]
    ) -> np.ndarray:
        """Convert list of bracket dicts to numpy matrix of entity indices.

        Returns array of shape ``(n_brackets, n_tourney_slots)``.
        """
        entity_to_idx = self.simulator._entity_to_idx
        n = len(bracket_dicts)
        n_slots = len(self._tourney_slots)
        matrix = np.zeros((n, n_slots), dtype=np.int32)
        for i, picks in enumerate(bracket_dicts):
            for j, slot in enumerate(self._tourney_slots):
                entity = picks.get(slot, "")
                matrix[i, j] = entity_to_idx.get(entity, -1)
        return matrix

    def _score_candidates_vectorized(
        self,
        candidates: list[tuple[dict[str, str], str]],
        candidate_matrix: np.ndarray,
        outcome_matrix: np.ndarray,
        opponent_matrix: np.ndarray,
    ) -> list[dict]:
        """Score candidates using vectorized numpy operations.

        Computes expected_points, win_probability, and top10_probability
        for each candidate against simulated outcomes and opponents.
        """
        n_sims = outcome_matrix.shape[0]
        n_opponents = opponent_matrix.shape[0]
        slot_points = self._slot_points

        # Pre-score all opponents: shape (n_sims, n_opponents)
        opp_scores = np.zeros((n_sims, n_opponents), dtype=np.float64)
        for j in range(n_opponents):
            matches = (outcome_matrix == opponent_matrix[j])
            opp_scores[:, j] = (matches * slot_points).sum(axis=1)

        # Per-sim max opponent score and top-10% threshold
        if n_opponents > 0:
            max_opp_scores = opp_scores.max(axis=1)
            top10_idx = max(0, n_opponents // 10)
            sorted_opp = np.sort(opp_scores, axis=1)[:, ::-1]
            top10_threshold = sorted_opp[:, min(top10_idx, sorted_opp.shape[1] - 1)]
        else:
            max_opp_scores = np.zeros(n_sims, dtype=np.float64)
            top10_threshold = np.zeros(n_sims, dtype=np.float64)

        # Score each candidate
        scored = []
        for i, (picks, strategy) in enumerate(candidates):
            bracket = candidate_matrix[i]
            matches = (outcome_matrix == bracket)
            our_scores = (matches * slot_points).sum(axis=1)

            total_points = float(our_scores.sum())
            wins = int((our_scores >= max_opp_scores).sum())
            top10_finishes = int((our_scores >= top10_threshold).sum())

            scored.append({
                "picks": picks,
                "expected_points": total_points / n_sims,
                "win_probability": wins / n_sims,
                "top10_probability": top10_finishes / n_sims,
                "strategy": strategy,
            })

        scored.sort(key=lambda x: x["win_probability"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_diverse(
        self, scored: list[dict], n_brackets: int
    ) -> list[dict]:
        """Select N diverse brackets using round-weighted overlap (greedy).

        Default overlap threshold is 0.90.
        """
        if len(scored) <= n_brackets:
            return scored

        selected = [scored[0]]

        for candidate in scored[1:]:
            if len(selected) >= n_brackets:
                break
            is_diverse = True
            for existing in selected:
                overlap = self._bracket_overlap(
                    candidate["picks"], existing["picks"]
                )
                if overlap > 0.90:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(candidate)

        # Fill remaining if diversity filtering was too aggressive
        if len(selected) < n_brackets:
            for candidate in scored:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= n_brackets:
                    break

        return selected[:n_brackets]

    def _bracket_overlap(
        self, picks_a: dict[str, str], picks_b: dict[str, str]
    ) -> float:
        """Round-weighted overlap where slot weight = round points.

        Late-round disagreements count more than early-round ones.
        """
        common_slots = set(picks_a) & set(picks_b)
        if not common_slots:
            return 0.0

        weighted_agree = 0.0
        total_weight = 0.0
        scoring_values = self.scoring.values

        for s in common_slots:
            rd = self.simulator.structure.slot_to_round.get(s, 0)
            if rd < 1:
                continue
            idx = min(rd - 1, len(scoring_values) - 1)
            weight = scoring_values[max(idx, 0)]
            total_weight += weight
            if picks_a[s] == picks_b[s]:
                weighted_agree += weight

        if total_weight == 0:
            return 0.0
        return weighted_agree / total_weight

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _enrich_brackets(self, scored: list[dict]) -> list[CompetitionResult]:
        """Convert scored bracket dicts to CompetitionResult with full MatchupContext."""
        results = []

        for bracket in scored:
            picks = bracket["picks"]
            matchups = {}

            max_round = max(self.simulator.structure.round_slots.keys())
            min_round = min(self.simulator.structure.round_slots.keys())

            for r in range(min_round, max_round + 1):
                if r not in self.simulator.structure.round_slots:
                    continue
                for slot in self.simulator.structure.round_slots[r]:
                    if r < 1 or slot not in picks:
                        continue

                    a_ref, b_ref = self.simulator.structure.slot_matchups[slot]
                    ea = self.simulator._resolve_entity(a_ref, picks)
                    eb = self.simulator._resolve_entity(b_ref, picks)

                    if ea == "BYE" or eb == "BYE" or ea == eb:
                        continue

                    ctx = self.simulator.get_matchup_context(
                        slot=slot,
                        entity_a=ea,
                        entity_b=eb,
                        round_num=r,
                    )
                    # Override pick and strategy from bracket
                    ctx = ctx.model_copy(update={
                        "pick": picks[slot],
                        "strategy": bracket["strategy"],
                    })
                    matchups[slot] = ctx

            results.append(CompetitionResult(
                picks=picks,
                matchups=matchups,
                expected_points=bracket["expected_points"],
                win_probability=bracket["win_probability"],
                top10_probability=bracket["top10_probability"],
                strategy=bracket["strategy"],
            ))

        return results
