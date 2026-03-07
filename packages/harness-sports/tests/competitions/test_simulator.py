"""Tests for the Monte Carlo competition simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from harnessml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    MatchupContext,
)
from harnessml.sports.competitions.simulator import CompetitionSimulator
from harnessml.sports.competitions.structure import build_structure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed_map(n: int) -> dict[str, str]:
    return {f"S{i}": f"entity_{i}" for i in range(1, n + 1)}


def _make_prob_df(
    seed_map: dict[str, str],
    dominant: str | None = None,
    dominant_prob: float = 0.95,
    default_prob: float = 0.5,
    n_models: int = 2,
) -> pd.DataFrame:
    """Build a probabilities DataFrame for all pairs.

    If *dominant* is given, that entity beats everyone with *dominant_prob*.
    Otherwise all matchups get *default_prob*.
    """
    entities = sorted(seed_map.values())
    rows = []
    for i, a in enumerate(entities):
        for b in entities[i + 1:]:
            if dominant == a:
                prob = dominant_prob
            elif dominant == b:
                prob = 1.0 - dominant_prob
            else:
                prob = default_prob
            row = {"entity_a": a, "entity_b": b, "prob_ensemble": prob}
            for m in range(1, n_models + 1):
                row[f"prob_model_{m}"] = prob
            rows.append(row)
    return pd.DataFrame(rows)


def _make_4team_setup(dominant: str | None = None, dominant_prob: float = 0.95):
    """Return (config, structure, probabilities, simulator) for a 4-entity bracket."""
    cfg = CompetitionConfig(
        format=CompetitionFormat.single_elimination,
        n_participants=4,
    )
    seed_map = _make_seed_map(4)
    structure = build_structure(cfg, seed_map)
    probs = _make_prob_df(seed_map, dominant=dominant, dominant_prob=dominant_prob)
    sim = CompetitionSimulator(cfg, structure, probs)
    return cfg, structure, probs, sim


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Tests for simulator construction and probability lookup."""

    def test_construction(self):
        _, _, _, sim = _make_4team_setup()
        assert sim is not None
        assert len(sim._prob_lookup) > 0
        assert len(sim._entities_sorted) == 4

    def test_prob_lookup_forward(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        # entity_1 < entity_2 lexicographically, so forward lookup
        prob = sim.get_win_prob("entity_1", "entity_2")
        assert prob == pytest.approx(0.95)

    def test_prob_lookup_reverse(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        # Reversed: entity_2 vs entity_1
        prob = sim.get_win_prob("entity_2", "entity_1")
        assert prob == pytest.approx(0.05)

    def test_prob_lookup_unknown(self):
        _, _, _, sim = _make_4team_setup()
        prob = sim.get_win_prob("entity_1", "unknown_entity")
        assert prob == pytest.approx(0.5)

    def test_prob_lookup_self(self):
        _, _, _, sim = _make_4team_setup()
        assert sim.get_win_prob("entity_1", "entity_1") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TestModelAgreement
# ---------------------------------------------------------------------------


class TestModelAgreement:
    """Tests for model agreement calculation."""

    def test_perfect_agreement(self):
        """All models agree => agreement = 1.0."""
        _, _, _, sim = _make_4team_setup()
        agreement = sim.get_model_agreement("entity_1", "entity_2")
        # All model probs are the same => std = 0 => agreement = 1.0
        assert agreement == pytest.approx(1.0)

    def test_disagreement(self):
        """Models that disagree should have agreement < 1.0."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=4,
        )
        seed_map = _make_seed_map(4)
        structure = build_structure(cfg, seed_map)
        entities = sorted(seed_map.values())
        rows = []
        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                rows.append({
                    "entity_a": a,
                    "entity_b": b,
                    "prob_ensemble": 0.6,
                    "prob_model_1": 0.8,
                    "prob_model_2": 0.4,
                })
        probs = pd.DataFrame(rows)
        sim = CompetitionSimulator(cfg, structure, probs)
        agreement = sim.get_model_agreement("entity_1", "entity_2")
        assert agreement < 1.0

    def test_agreement_symmetric(self):
        _, _, _, sim = _make_4team_setup()
        a1 = sim.get_model_agreement("entity_1", "entity_2")
        a2 = sim.get_model_agreement("entity_2", "entity_1")
        assert a1 == pytest.approx(a2)

    def test_agreement_unknown_pair(self):
        _, _, _, sim = _make_4team_setup()
        assert sim.get_model_agreement("entity_1", "unknown") == pytest.approx(1.0)

    def test_agreement_self(self):
        _, _, _, sim = _make_4team_setup()
        assert sim.get_model_agreement("entity_1", "entity_1") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestMatchupContext
# ---------------------------------------------------------------------------


class TestMatchupContext:
    """Tests for get_matchup_context."""

    def test_returns_matchup_context(self):
        _, structure, _, sim = _make_4team_setup(dominant="entity_1")
        slot = structure.round_slots[1][0]
        a_ref, b_ref = structure.slot_matchups[slot]
        ea = structure.seed_to_entity[a_ref]
        eb = structure.seed_to_entity[b_ref]
        ctx = sim.get_matchup_context(slot, ea, eb)
        assert isinstance(ctx, MatchupContext)
        assert ctx.slot == slot
        assert ctx.round_num == 1

    def test_upset_detection(self):
        """When entity_a is the underdog, upset should be True."""
        _, _, _, sim = _make_4team_setup(dominant="entity_2")
        # entity_1 vs entity_2: entity_1 is underdog (prob < 0.5)
        ctx = sim.get_matchup_context("R1G1", "entity_1", "entity_2")
        assert ctx.upset is True
        assert ctx.pick == "entity_2"

    def test_no_upset_when_favorite(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        ctx = sim.get_matchup_context("R1G1", "entity_1", "entity_2")
        assert ctx.upset is False
        assert ctx.pick == "entity_1"

    def test_model_probs_populated(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        ctx = sim.get_matchup_context("R1G1", "entity_1", "entity_2")
        assert len(ctx.model_probs) == 2  # 2 model columns


# ---------------------------------------------------------------------------
# TestSimulateOnce
# ---------------------------------------------------------------------------


class TestSimulateOnce:
    """Tests for simulate_once."""

    def test_returns_all_slots(self):
        _, structure, _, sim = _make_4team_setup()
        rng = np.random.default_rng(42)
        result = sim.simulate_once(rng)
        assert set(result.keys()) == set(structure.slots)

    def test_deterministic_with_same_seed(self):
        _, _, _, sim = _make_4team_setup()
        r1 = sim.simulate_once(np.random.default_rng(99))
        r2 = sim.simulate_once(np.random.default_rng(99))
        assert r1 == r2

    def test_dominant_entity_wins(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1", dominant_prob=0.999)
        # With very high prob, entity_1 should almost always win the final
        wins = 0
        for seed in range(100):
            result = sim.simulate_once(np.random.default_rng(seed))
            final_slot = list(result.keys())[-1]
            if result[final_slot] == "entity_1":
                wins += 1
        assert wins > 90  # Should win nearly all


# ---------------------------------------------------------------------------
# TestSimulateMany
# ---------------------------------------------------------------------------


class TestSimulateMany:
    """Tests for simulate_many (vectorized)."""

    def test_correct_shape(self):
        _, structure, _, sim = _make_4team_setup()
        results = sim.simulate_many(100, seed=42)
        assert len(results) == 100
        for r in results:
            assert set(r.keys()) == set(structure.slots)

    def test_deterministic(self):
        _, _, _, sim = _make_4team_setup()
        r1 = sim.simulate_many(50, seed=123)
        r2 = sim.simulate_many(50, seed=123)
        assert r1 == r2

    def test_dominant_entity_rate(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1", dominant_prob=0.99)
        results = sim.simulate_many(1000, seed=42)
        final_slot = sim._slots_sorted[-1]
        wins = sum(1 for r in results if r[final_slot] == "entity_1")
        assert wins / 1000 > 0.85

    def test_distribution_matches_scalar(self):
        """Vectorized results should match scalar sim distribution."""
        _, _, _, sim = _make_4team_setup(dominant="entity_1", dominant_prob=0.8)

        # Vectorized
        vec_results = sim.simulate_many(500, seed=42)
        final_slot = sim._slots_sorted[-1]
        vec_rate = sum(1 for r in vec_results if r[final_slot] == "entity_1") / 500

        # Scalar (same seed)
        rng = np.random.default_rng(42)
        scalar_wins = 0
        for _ in range(500):
            r = sim.simulate_once(rng)
            if r[final_slot] == "entity_1":
                scalar_wins += 1
        scalar_rate = scalar_wins / 500

        # They use different RNG streams, so compare distribution loosely
        assert abs(vec_rate - scalar_rate) < 0.15


# ---------------------------------------------------------------------------
# TestPickMostLikely
# ---------------------------------------------------------------------------


class TestPickMostLikely:
    """Tests for pick_most_likely (chalk bracket)."""

    def test_returns_all_slots(self):
        _, structure, _, sim = _make_4team_setup()
        picks = sim.pick_most_likely()
        assert set(picks.keys()) == set(structure.slots)

    def test_dominant_wins(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        picks = sim.pick_most_likely()
        final_slot = sim._slots_sorted[-1]
        assert picks[final_slot] == "entity_1"

    def test_always_picks_favorite(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1", dominant_prob=0.9)
        picks = sim.pick_most_likely()
        # entity_1 should win every slot it appears in
        for slot in sim._slots_sorted:
            winner = picks[slot]
            a_ref, b_ref = sim.structure.slot_matchups[slot]
            ea = sim._resolve_entity(a_ref, picks)
            eb = sim._resolve_entity(b_ref, picks)
            if ea != eb and ea != "BYE" and eb != "BYE":
                prob_a = sim.get_win_prob(ea, eb)
                expected = ea if prob_a >= 0.5 else eb
                assert winner == expected


# ---------------------------------------------------------------------------
# TestEntityRoundProbabilities
# ---------------------------------------------------------------------------


class TestEntityRoundProbabilities:
    """Tests for entity_round_probabilities."""

    def test_returns_dataframe(self):
        _, _, _, sim = _make_4team_setup()
        df = sim.entity_round_probabilities(n_sims=200, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert "entity" in df.columns

    def test_round_1_sums_correctly(self):
        """Round 1 win counts across all entities should sum correctly."""
        _, _, _, sim = _make_4team_setup()
        df = sim.entity_round_probabilities(n_sims=500, seed=42)
        # In round 1 of a 4-team bracket, 2 games produce 2 winners
        # So sum of round_1 probs should be ~2.0 (2 winners * prob)
        if "round_1" in df.columns:
            total = df["round_1"].sum()
            # Each sim has 2 round-1 winners => expected sum = 2.0
            assert total == pytest.approx(2.0, abs=0.1)

    def test_champion_sums_to_1(self):
        """Champion probabilities should sum to 1.0."""
        _, _, _, sim = _make_4team_setup()
        df = sim.entity_round_probabilities(n_sims=1000, seed=42)
        assert "champion" in df.columns
        assert df["champion"].sum() == pytest.approx(1.0, abs=0.01)

    def test_dominant_entity_high(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1", dominant_prob=0.95)
        df = sim.entity_round_probabilities(n_sims=1000, seed=42)
        e1_row = df[df["entity"] == "entity_1"].iloc[0]
        assert e1_row["champion"] > 0.5


# ---------------------------------------------------------------------------
# Test8EntityTournament
# ---------------------------------------------------------------------------


class Test8EntityTournament:
    """Full 8-entity single-elimination tournament tests."""

    def _setup(self):
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=8,
        )
        seed_map = _make_seed_map(8)
        structure = build_structure(cfg, seed_map)
        probs = _make_prob_df(seed_map, dominant="entity_1", dominant_prob=0.9)
        sim = CompetitionSimulator(cfg, structure, probs)
        return cfg, structure, probs, sim

    def test_slot_count(self):
        _, structure, _, sim = self._setup()
        assert len(structure.slots) == 7  # 4 + 2 + 1

    def test_simulate_once_all_slots(self):
        _, structure, _, sim = self._setup()
        result = sim.simulate_once(np.random.default_rng(42))
        assert len(result) == 7

    def test_simulate_many_shape(self):
        _, _, _, sim = self._setup()
        results = sim.simulate_many(200, seed=42)
        assert len(results) == 200
        assert len(results[0]) == 7

    def test_dominant_wins_tournament(self):
        _, _, _, sim = self._setup()
        results = sim.simulate_many(500, seed=42)
        final_slot = sim._slots_sorted[-1]
        wins = sum(1 for r in results if r[final_slot] == "entity_1")
        assert wins / 500 > 0.5

    def test_chalk_bracket(self):
        _, _, _, sim = self._setup()
        picks = sim.pick_most_likely()
        final_slot = sim._slots_sorted[-1]
        assert picks[final_slot] == "entity_1"

    def test_round_probabilities(self):
        _, _, _, sim = self._setup()
        df = sim.entity_round_probabilities(n_sims=500, seed=42)
        assert len(df) == 8
        assert "champion" in df.columns
        assert df["champion"].sum() == pytest.approx(1.0, abs=0.05)

    def test_bye_handling(self):
        """6-entity bracket has byes; should still work."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=6,
        )
        seed_map = _make_seed_map(6)
        structure = build_structure(cfg, seed_map)
        probs = _make_prob_df(seed_map, dominant="entity_1", dominant_prob=0.9)
        sim = CompetitionSimulator(cfg, structure, probs)
        result = sim.simulate_once(np.random.default_rng(42))
        assert len(result) == len(structure.slots)
        # entity_1 or entity_2 should get byes (top seeds)
        picks = sim.pick_most_likely()
        assert picks[sim._slots_sorted[-1]] == "entity_1"
