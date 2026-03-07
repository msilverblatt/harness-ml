"""Full pipeline integration tests for the competition engine.

End-to-end tests covering single elimination, round robin, cross-format
consistency, and the adjustments pipeline.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from harnessml.sports.competitions.adjustments import apply_adjustments
from harnessml.sports.competitions.confidence import generate_confidence_report
from harnessml.sports.competitions.explainer import CompetitionExplainer
from harnessml.sports.competitions.export import (
    export_analysis_report,
    export_bracket_markdown,
    export_json,
    export_standings_markdown,
)
from harnessml.sports.competitions.optimizer import CompetitionOptimizer
from harnessml.sports.competitions.schemas import (
    AdjustmentConfig,
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    ScoringConfig,
    StandingsEntry,
)
from harnessml.sports.competitions.scorer import CompetitionScorer
from harnessml.sports.competitions.simulator import CompetitionSimulator
from harnessml.sports.competitions.structure import build_structure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed_map(n: int) -> dict[str, str]:
    """Create a seed-to-entity mapping for *n* participants."""
    return {f"S{i}": f"entity_{i}" for i in range(1, n + 1)}


def _make_prob_df(
    seed_map: dict[str, str],
    dominant: str | None = None,
    dominant_prob: float = 0.95,
    default_prob: float = 0.5,
    n_models: int = 2,
) -> pd.DataFrame:
    """Build a probabilities DataFrame for all entity pairs.

    If *dominant* is given, that entity beats everyone with *dominant_prob*.
    Otherwise all matchups use *default_prob*.
    """
    entities = sorted(seed_map.values())
    rows = []
    for i, a in enumerate(entities):
        for b in entities[i + 1 :]:
            if dominant == a:
                prob = dominant_prob
            elif dominant == b:
                prob = 1.0 - dominant_prob
            else:
                prob = default_prob
            row = {"entity_a": a, "entity_b": b, "prob_ensemble": prob}
            for m in range(1, n_models + 1):
                row[f"prob_model_{m}"] = prob + (m - 1) * 0.01
            rows.append(row)
    return pd.DataFrame(rows)


def _make_entity_features(seed_map: dict[str, str]) -> pd.DataFrame:
    """Build a synthetic entity features DataFrame."""
    rng = np.random.default_rng(99)
    entities = sorted(seed_map.values())
    return pd.DataFrame(
        {
            "entity": entities,
            "strength": rng.uniform(0.3, 0.9, len(entities)),
            "consistency": rng.uniform(0.4, 1.0, len(entities)),
            "experience": rng.uniform(1.0, 10.0, len(entities)),
        }
    )


# ---------------------------------------------------------------------------
# Test 1: Single elimination end-to-end (8 entities, 3 rounds)
# ---------------------------------------------------------------------------


class TestSingleEliminationEndToEnd:
    """Full pipeline: config -> structure -> simulate -> optimize -> score
    -> explain -> confidence -> export."""

    def test_full_pipeline(self, tmp_path):
        # 1. Create config
        scoring = ScoringConfig(type="per_round", values=[10, 20, 40])
        config = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=8,
            scoring=scoring,
        )

        # 2. Build entities and structure
        seed_map = _make_seed_map(8)
        structure = build_structure(config, seed_map)

        # Verify structure basics
        assert len(structure.slots) > 0
        assert len(structure.round_slots) == 3  # 8 entities = 3 rounds
        assert len(structure.round_slots[1]) == 4  # 4 first-round games
        assert len(structure.round_slots[2]) == 2  # 2 second-round games
        assert len(structure.round_slots[3]) == 1  # 1 final

        # 3. Create synthetic probabilities (entity_1 dominant)
        prob_df = _make_prob_df(seed_map, dominant="entity_1", dominant_prob=0.95)
        assert len(prob_df) > 0

        # 4. Create simulator
        simulator = CompetitionSimulator(config, structure, prob_df)

        # 5. Generate brackets with optimizer
        optimizer = CompetitionOptimizer(simulator, scoring)
        brackets = optimizer.generate_brackets(
            pool_size=10,
            n_brackets=3,
            n_sims=200,
            seed=42,
        )

        # 6. Assert brackets have expected structure
        assert len(brackets) == 3
        for bracket in brackets:
            assert isinstance(bracket, CompetitionResult)
            assert len(bracket.picks) > 0
            assert bracket.strategy != ""
            assert bracket.expected_points >= 0
            assert 0.0 <= bracket.win_probability <= 1.0
            assert 0.0 <= bracket.top10_probability <= 1.0
            # Every slot should have a pick
            for slot in structure.slots:
                assert slot in bracket.picks, f"Missing pick for slot {slot}"

        # 7. Simulate actuals
        rng = np.random.default_rng(123)
        actuals = simulator.simulate_once(rng)
        assert len(actuals) == len(structure.slots)

        # 8. Score each bracket
        scorer = CompetitionScorer(scoring)
        for bracket in brackets:
            score = scorer.score_bracket(bracket.picks, actuals, structure)
            # 9. Assert scores are valid
            assert score.total_points >= 0
            # Round breakdowns should sum to total
            round_sum = sum(score.round_points.values())
            assert abs(round_sum - score.total_points) < 1e-9
            # Each round's correct count should not exceed total slots in that round
            for rk, correct_count in score.round_correct.items():
                assert correct_count <= score.round_total[rk]

        # 10. Create explainer
        entity_features = _make_entity_features(seed_map)
        explainer = CompetitionExplainer(entity_features)

        # 11. Generate pick stories and entity profiles
        pick_stories = explainer.generate_pick_stories(brackets[0])
        assert isinstance(pick_stories, list)
        # Should have stories for matchups in the bracket
        if brackets[0].matchups:
            assert len(pick_stories) > 0
            for story in pick_stories:
                assert "slot" in story
                assert "narrative" in story
                assert "pick" in story

        round_probs = simulator.entity_round_probabilities(n_sims=200, seed=42)
        profiles = explainer.generate_entity_profiles(simulator, round_probs)
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        for profile in profiles:
            assert "entity" in profile
            assert "champion_prob" in profile

        # 12. Generate confidence report
        training_features = _make_entity_features(_make_seed_map(16))
        confidence_report = generate_confidence_report(
            simulator=simulator,
            entity_features=entity_features,
            training_features=training_features,
        )
        assert "feature_outliers" in confidence_report
        assert "high_disagreement" in confidence_report
        assert isinstance(confidence_report["feature_outliers"], list)
        assert isinstance(confidence_report["high_disagreement"], list)

        # 13. Export bracket markdown, JSON, and analysis report
        entity_names = {eid: eid.replace("_", " ").title() for eid in seed_map.values()}

        md_path = export_bracket_markdown(
            result=brackets[0],
            entity_names=entity_names,
            structure=structure,
            output_dir=tmp_path / "bracket_out",
        )
        json_path = export_json(
            results=brackets,
            entity_names=entity_names,
            output_dir=tmp_path / "json_out",
            filename="brackets.json",
        )
        analysis_path = export_analysis_report(
            result=brackets[0],
            pick_stories=pick_stories,
            entity_profiles=profiles,
            confidence_report=confidence_report,
            entity_names=entity_names,
            output_dir=tmp_path / "analysis_out",
        )

        # 14. Assert all output files exist
        assert md_path.exists()
        assert md_path.stat().st_size > 0
        assert json_path.exists()
        assert json_path.stat().st_size > 0
        assert analysis_path.exists()
        assert analysis_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 2: Round-robin end-to-end (6 entities, 1 round)
# ---------------------------------------------------------------------------


class TestRoundRobinEndToEnd:
    """Round-robin: config -> structure -> simulate -> standings -> export."""

    def test_full_pipeline(self, tmp_path):
        # 1. Create round_robin config
        scoring = ScoringConfig(type="points", win=3.0, draw=1.0, loss=0.0)
        config = CompetitionConfig(
            format=CompetitionFormat.round_robin,
            n_participants=6,
            scoring=scoring,
        )

        # 2. Build structure, verify 15 matchups (6 choose 2)
        seed_map = _make_seed_map(6)
        structure = build_structure(config, seed_map)
        expected_matchups = len(list(combinations(range(6), 2)))  # 15
        assert len(structure.slots) == expected_matchups
        assert len(structure.slot_matchups) == expected_matchups

        # 3. Create simulator with synthetic probabilities
        prob_df = _make_prob_df(seed_map, dominant="entity_1", dominant_prob=0.85)
        simulator = CompetitionSimulator(config, structure, prob_df)

        # 4. simulate_many (200 sims)
        results = simulator.simulate_many(200, seed=42)
        assert len(results) == 200

        # 5. Verify results contain all matchup slots
        for sim_result in results:
            for slot in structure.slots:
                assert slot in sim_result, f"Missing slot {slot} in simulation result"

        # 6. Export standings markdown
        # Build standings from simulation frequency
        entity_wins: dict[str, int] = {eid: 0 for eid in seed_map.values()}
        entity_losses: dict[str, int] = {eid: 0 for eid in seed_map.values()}

        # Use a single sim result to build example standings
        single_result = results[0]
        for slot in structure.slots:
            a_ref, b_ref = structure.slot_matchups[slot]
            ea = structure.seed_to_entity.get(a_ref, a_ref)
            eb = structure.seed_to_entity.get(b_ref, b_ref)
            winner = single_result[slot]
            loser = eb if winner == ea else ea
            entity_wins[winner] = entity_wins.get(winner, 0) + 1
            entity_losses[loser] = entity_losses.get(loser, 0) + 1

        standings = [
            StandingsEntry(
                entity=eid,
                wins=entity_wins[eid],
                losses=entity_losses[eid],
                points=entity_wins[eid] * scoring.win,
            )
            for eid in seed_map.values()
        ]
        standings.sort(key=lambda s: s.points, reverse=True)

        entity_names = {eid: eid.replace("_", " ").title() for eid in seed_map.values()}
        standings_path = export_standings_markdown(
            standings=standings,
            entity_names=entity_names,
            output_dir=tmp_path / "rr_out",
        )
        assert standings_path.exists()
        assert standings_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 3: Cross-format consistency
# ---------------------------------------------------------------------------


class TestCrossFormatConsistency:
    """Same entities + probabilities yield same lookup values in both formats."""

    def test_same_probabilities_across_formats(self):
        # 1. Create both configs with same 4 entities
        se_config = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        rr_config = CompetitionConfig(
            format=CompetitionFormat.round_robin,
            n_participants=4,
            scoring=ScoringConfig(type="points", win=3.0, draw=1.0, loss=0.0),
        )

        # 2. Build both structures
        seed_map = _make_seed_map(4)
        se_structure = build_structure(se_config, seed_map)
        rr_structure = build_structure(rr_config, seed_map)

        assert len(se_structure.slots) > 0
        assert len(rr_structure.slots) > 0

        # 3. Create simulator for each with same probability DataFrame
        prob_df = _make_prob_df(seed_map, dominant="entity_1", dominant_prob=0.80)
        se_sim = CompetitionSimulator(se_config, se_structure, prob_df)
        rr_sim = CompetitionSimulator(rr_config, rr_structure, prob_df)

        # 4. Verify get_win_prob returns same values regardless of format
        entities = sorted(seed_map.values())
        for i, a in enumerate(entities):
            for b in entities[i + 1 :]:
                se_prob = se_sim.get_win_prob(a, b)
                rr_prob = rr_sim.get_win_prob(a, b)
                assert abs(se_prob - rr_prob) < 1e-9, (
                    f"Win prob mismatch for {a} vs {b}: SE={se_prob}, RR={rr_prob}"
                )

        # 5. Verify model_agreement returns same values
        for i, a in enumerate(entities):
            for b in entities[i + 1 :]:
                se_agree = se_sim.get_model_agreement(a, b)
                rr_agree = rr_sim.get_model_agreement(a, b)
                assert abs(se_agree - rr_agree) < 1e-9, (
                    f"Agreement mismatch for {a} vs {b}: SE={se_agree}, RR={rr_agree}"
                )


# ---------------------------------------------------------------------------
# Test 4: Adjustments pipeline
# ---------------------------------------------------------------------------


class TestAdjustmentsPipeline:
    """Adjustments modify probabilities and affect simulation outcomes."""

    def test_entity_multiplier_changes_probs_and_outcomes(self):
        seed_map = _make_seed_map(8)

        # 1. Create probabilities DataFrame
        prob_df = _make_prob_df(seed_map, dominant="entity_1", dominant_prob=0.90)
        # The adjustments module uses "prob" column, not "prob_ensemble"
        adj_df = prob_df.rename(columns={"prob_ensemble": "prob"})

        # 2. Apply entity multiplier adjustment (weaken entity_1)
        adjustments = AdjustmentConfig(
            entity_multipliers={"entity_1": 0.5},
        )
        adjusted_df, log = apply_adjustments(adj_df, adjustments)

        # 3. Verify probabilities changed
        assert len(log) > 0
        # entity_1's probabilities should have decreased
        for entry in log:
            if entry["entity"] == "entity_1" and entry["side"] == "entity_a":
                assert entry["new_prob"] < entry["old_prob"]

        # 4. Create simulator with adjusted probabilities
        # Rename back for the simulator
        sim_df = adjusted_df.rename(columns={"prob": "prob_ensemble"})
        config = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=8,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
        )
        structure = build_structure(config, seed_map)
        simulator = CompetitionSimulator(config, structure, sim_df)

        # 5. Verify entity_1 win prob is lower after adjustment
        # entity_1 vs entity_2: originally 0.90, after 0.5x multiplier should be ~0.45
        weakened_prob = simulator.get_win_prob("entity_1", "entity_2")
        assert weakened_prob < 0.90, (
            f"Expected weakened prob < 0.90, got {weakened_prob}"
        )

        # Run simulations and verify entity_1 wins less often as champion
        results = simulator.simulate_many(500, seed=42)
        max_round = max(structure.round_slots.keys())
        final_slot = structure.round_slots[max_round][0]

        entity_1_wins = sum(1 for r in results if r[final_slot] == "entity_1")
        win_rate = entity_1_wins / len(results)
        # With 0.5x multiplier on a 0.90 dominant entity, win rate should be
        # noticeably lower than it would be with raw 0.90 probs
        assert win_rate < 0.85, (
            f"Expected reduced win rate for weakened entity_1, got {win_rate}"
        )
