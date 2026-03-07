"""Tests for the pool-aware bracket optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harnessml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    ScoringConfig,
)
from harnessml.sports.competitions.simulator import CompetitionSimulator
from harnessml.sports.competitions.structure import build_structure
from harnessml.sports.competitions.optimizer import (
    BUILTIN_STRATEGIES,
    CompetitionOptimizer,
    StrategyFn,
    chalk_strategy,
    near_chalk_strategy,
    random_sim_strategy,
    contrarian_strategy,
    late_contrarian_strategy,
    champion_anchor_strategy,
)


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
) -> pd.DataFrame:
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
            rows.append({
                "entity_a": a,
                "entity_b": b,
                "prob_ensemble": prob,
                "prob_model_1": prob,
                "prob_model_2": prob,
            })
    return pd.DataFrame(rows)


def _make_8team_setup(
    dominant: str | None = "entity_1",
    dominant_prob: float = 0.90,
):
    """Return (config, structure, probs, simulator, scoring) for 8-entity bracket."""
    cfg = CompetitionConfig(
        format=CompetitionFormat.single_elimination,
        n_participants=8,
    )
    seed_map = _make_seed_map(8)
    structure = build_structure(cfg, seed_map)
    probs = _make_prob_df(seed_map, dominant=dominant, dominant_prob=dominant_prob)
    sim = CompetitionSimulator(cfg, structure, probs)
    scoring = ScoringConfig(type="per_round", values=[10, 20, 40])
    return cfg, structure, probs, sim, scoring


# ---------------------------------------------------------------------------
# TestBuiltinStrategies
# ---------------------------------------------------------------------------


class TestBuiltinStrategies:
    """Tests for BUILTIN_STRATEGIES dict and StrategyFn protocol."""

    def test_contains_all_six(self):
        assert len(BUILTIN_STRATEGIES) == 6
        expected = {
            "chalk", "near_chalk", "random_sim",
            "contrarian", "late_contrarian", "champion_anchor",
        }
        assert set(BUILTIN_STRATEGIES.keys()) == expected

    def test_strategy_fn_protocol(self):
        for name, fn in BUILTIN_STRATEGIES.items():
            assert isinstance(fn, StrategyFn), f"{name} does not satisfy StrategyFn"

    def test_chalk_matches_pick_most_likely(self):
        _, _, _, sim, _ = _make_8team_setup()
        rng = np.random.default_rng(42)
        chalk = chalk_strategy(sim, rng)
        most_likely = sim.pick_most_likely()
        assert chalk == most_likely


# ---------------------------------------------------------------------------
# TestGenerateBrackets
# ---------------------------------------------------------------------------


class TestGenerateBrackets:
    """Tests for CompetitionOptimizer.generate_brackets."""

    def test_returns_correct_count(self):
        _, _, _, sim, scoring = _make_8team_setup()
        opt = CompetitionOptimizer(sim, scoring)
        results = opt.generate_brackets(
            pool_size=10, n_brackets=5, n_sims=200, seed=42
        )
        assert len(results) == 5

    def test_returns_competition_results(self):
        _, _, _, sim, scoring = _make_8team_setup()
        opt = CompetitionOptimizer(sim, scoring)
        results = opt.generate_brackets(
            pool_size=10, n_brackets=3, n_sims=200, seed=42
        )
        for r in results:
            assert isinstance(r, CompetitionResult)

    def test_results_have_expected_fields(self):
        _, _, _, sim, scoring = _make_8team_setup()
        opt = CompetitionOptimizer(sim, scoring)
        results = opt.generate_brackets(
            pool_size=10, n_brackets=3, n_sims=200, seed=42
        )
        for r in results:
            assert r.expected_points > 0
            assert 0.0 <= r.win_probability <= 1.0
            assert r.strategy != ""

    def test_results_are_diverse(self):
        """Championship picks should differ across brackets."""
        _, _, _, sim, scoring = _make_8team_setup(dominant_prob=0.6)
        opt = CompetitionOptimizer(sim, scoring)
        results = opt.generate_brackets(
            pool_size=100, n_brackets=10, n_sims=200, seed=42
        )
        final_slot = sim._slots_sorted[-1]
        champions = {r.picks[final_slot] for r in results}
        # With moderate probabilities and 10 brackets, expect some diversity
        assert len(champions) >= 2

    def test_dominant_entity_champion_in_most(self):
        """A strongly dominant entity should be champion in most brackets."""
        _, _, _, sim, scoring = _make_8team_setup(
            dominant="entity_1", dominant_prob=0.95
        )
        opt = CompetitionOptimizer(sim, scoring)
        results = opt.generate_brackets(
            pool_size=10, n_brackets=5, n_sims=200, seed=42
        )
        final_slot = sim._slots_sorted[-1]
        e1_champ = sum(1 for r in results if r.picks[final_slot] == "entity_1")
        assert e1_champ >= 2  # Most should have entity_1 as champion

    def test_small_pool_returns_n_brackets(self):
        _, _, _, sim, scoring = _make_8team_setup()
        opt = CompetitionOptimizer(sim, scoring)
        results = opt.generate_brackets(
            pool_size=3, n_brackets=2, n_sims=200, seed=42
        )
        assert len(results) == 2
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TestCustomStrategy
# ---------------------------------------------------------------------------


class TestCustomStrategy:
    """Test that custom strategies can be registered."""

    def test_custom_strategy_registered(self):
        _, _, _, sim, scoring = _make_8team_setup()

        def my_strategy(
            simulator: CompetitionSimulator,
            rng: np.random.Generator,
            **kwargs: object,
        ) -> dict[str, str]:
            return simulator.pick_most_likely()

        opt = CompetitionOptimizer(
            sim, scoring, strategies={"my_custom": my_strategy}
        )
        assert "my_custom" in opt.strategies
        # Built-ins should still be there
        assert "chalk" in opt.strategies


# ---------------------------------------------------------------------------
# TestStrategyMix
# ---------------------------------------------------------------------------


class TestStrategyMix:
    """Test that strategy mix changes with pool size."""

    def test_mix_changes_with_pool_size(self):
        _, _, _, sim, scoring = _make_8team_setup()
        opt = CompetitionOptimizer(sim, scoring)

        small = opt._get_strategy_mix(10)
        large = opt._get_strategy_mix(10_000)

        # Small pools should favor near_chalk more
        assert small["near_chalk"] > large["near_chalk"]
        # Large pools should favor contrarian/champion_anchor more
        assert large["contrarian"] > small["contrarian"]
        assert large["champion_anchor"] > small["champion_anchor"]

    def test_mix_values_sum_to_one(self):
        _, _, _, sim, scoring = _make_8team_setup()
        opt = CompetitionOptimizer(sim, scoring)
        for pool_size in [5, 50, 500, 5000]:
            mix = opt._get_strategy_mix(pool_size)
            assert sum(mix.values()) == pytest.approx(1.0, abs=0.01)
