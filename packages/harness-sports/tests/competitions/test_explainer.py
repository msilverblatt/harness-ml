"""Tests for the generic explanation engine with hook-based narratives."""

from __future__ import annotations

import pandas as pd
from harnessml.sports.competitions.explainer import CompetitionExplainer
from harnessml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    MatchupContext,
)
from harnessml.sports.competitions.simulator import CompetitionSimulator
from harnessml.sports.competitions.structure import build_structure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity_features(entities: list[str], n_features: int = 4) -> pd.DataFrame:
    """Build a synthetic entity features DataFrame."""
    rows = []
    for i, entity in enumerate(entities):
        row = {"entity": entity}
        for f in range(1, n_features + 1):
            row[f"feat_{f}"] = float((i + 1) * f)
        rows.append(row)
    return pd.DataFrame(rows)


def _make_seed_map(n: int) -> dict[str, str]:
    return {f"S{i}": f"entity_{i}" for i in range(1, n + 1)}


def _make_prob_df(
    seed_map: dict[str, str],
    dominant: str | None = None,
    dominant_prob: float = 0.95,
    default_prob: float = 0.5,
    n_models: int = 2,
) -> pd.DataFrame:
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
                row[f"prob_model_{m}"] = prob
            rows.append(row)
    return pd.DataFrame(rows)


def _make_4team_setup(dominant: str | None = None, dominant_prob: float = 0.95):
    cfg = CompetitionConfig(
        format=CompetitionFormat.single_elimination,
        n_participants=4,
        rounds=["Semis", "Final"],
    )
    seed_map = _make_seed_map(4)
    structure = build_structure(cfg, seed_map)
    probs = _make_prob_df(seed_map, dominant=dominant, dominant_prob=dominant_prob)
    sim = CompetitionSimulator(cfg, structure, probs)
    return cfg, structure, probs, sim


def _make_competition_result(sim: CompetitionSimulator) -> CompetitionResult:
    """Build a CompetitionResult with MatchupContext objects in matchups."""
    picks = sim.pick_most_likely()
    matchups: dict = {}
    for slot in sim.structure.slots:
        a_ref, b_ref = sim.structure.slot_matchups[slot]
        ea = sim._resolve_entity(a_ref, picks)
        eb = sim._resolve_entity(b_ref, picks)
        if ea != eb and ea != "BYE" and eb != "BYE":
            ctx = sim.get_matchup_context(slot, ea, eb)
            ctx.pick = picks[slot]
            matchups[slot] = ctx
    return CompetitionResult(picks=picks, matchups=matchups)


# ---------------------------------------------------------------------------
# Tests: compute_differentials
# ---------------------------------------------------------------------------


class TestComputeDifferentials:
    def test_sorted_by_magnitude(self):
        entities = ["A", "B"]
        df = pd.DataFrame(
            [
                {"entity": "A", "small": 1.0, "medium": 5.0, "large": 10.0},
                {"entity": "B", "small": 0.5, "medium": 2.0, "large": 3.0},
            ]
        )
        explainer = CompetitionExplainer(df)
        diffs = explainer.compute_differentials("A", "B", top_n=3)

        assert len(diffs) == 3
        # Should be sorted by |difference| descending
        magnitudes = [abs(d["difference"]) for d in diffs]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert diffs[0]["feature"] == "large"  # diff=7.0

    def test_uses_display_names(self):
        df = pd.DataFrame(
            [
                {"entity": "A", "off_rtg": 110.0, "def_rtg": 95.0},
                {"entity": "B", "off_rtg": 105.0, "def_rtg": 100.0},
            ]
        )
        display = {"off_rtg": "Offensive Rating", "def_rtg": "Defensive Rating"}
        explainer = CompetitionExplainer(df, feature_display_names=display)
        diffs = explainer.compute_differentials("A", "B")

        for d in diffs:
            assert d["display_name"] == display[d["feature"]]

    def test_display_name_falls_back_to_feature_name(self):
        df = pd.DataFrame(
            [
                {"entity": "A", "x": 1.0},
                {"entity": "B", "x": 2.0},
            ]
        )
        explainer = CompetitionExplainer(df)
        diffs = explainer.compute_differentials("A", "B")
        assert diffs[0]["display_name"] == "x"

    def test_works_with_any_feature_set(self):
        df = pd.DataFrame(
            [
                {"entity": "A", "alpha": 1.0, "beta": 2.0, "gamma": 3.0, "delta": 4.0, "epsilon": 5.0, "zeta": 6.0},
                {"entity": "B", "alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "epsilon": 0.0, "zeta": 0.0},
            ]
        )
        explainer = CompetitionExplainer(df)
        diffs = explainer.compute_differentials("A", "B", top_n=3)
        assert len(diffs) == 3
        # Top 3 by magnitude should be zeta(6), epsilon(5), delta(4)
        assert [d["feature"] for d in diffs] == ["zeta", "epsilon", "delta"]

    def test_favors_field(self):
        df = pd.DataFrame(
            [
                {"entity": "A", "strength": 10.0},
                {"entity": "B", "strength": 15.0},
            ]
        )
        explainer = CompetitionExplainer(df)
        diffs = explainer.compute_differentials("A", "B")
        assert diffs[0]["favors"] == "B"
        assert diffs[0]["difference"] == -5.0

    def test_unknown_entity_returns_empty(self):
        df = pd.DataFrame([{"entity": "A", "x": 1.0}])
        explainer = CompetitionExplainer(df)
        assert explainer.compute_differentials("A", "UNKNOWN") == []

    def test_empty_features_returns_empty(self):
        df = pd.DataFrame(columns=["entity"])
        explainer = CompetitionExplainer(df)
        assert explainer.compute_differentials("A", "B") == []

    def test_top_n_limits_results(self):
        df = pd.DataFrame(
            [
                {"entity": "A", "a": 10.0, "b": 5.0, "c": 3.0},
                {"entity": "B", "a": 0.0, "b": 0.0, "c": 0.0},
            ]
        )
        explainer = CompetitionExplainer(df)
        diffs = explainer.compute_differentials("A", "B", top_n=2)
        assert len(diffs) == 2


# ---------------------------------------------------------------------------
# Tests: generate_pick_stories
# ---------------------------------------------------------------------------


class TestGeneratePickStories:
    def test_one_story_per_matchup(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        entities = [f"entity_{i}" for i in range(1, 5)]
        features = _make_entity_features(entities)
        explainer = CompetitionExplainer(features)

        result = _make_competition_result(sim)
        stories = explainer.generate_pick_stories(result)

        assert len(stories) == len(result.matchups)
        for story in stories:
            assert "slot" in story
            assert "round" in story
            assert "pick" in story
            assert "opponent" in story
            assert "probability" in story
            assert "model_agreement" in story
            assert "upset" in story
            assert "strategy" in story
            assert "key_differentials" in story
            assert "narrative" in story
            assert "model_probs" in story

    def test_uses_narrative_hook(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        entities = [f"entity_{i}" for i in range(1, 5)]
        features = _make_entity_features(entities)

        custom_narrative = "Custom hook narrative here"

        def hook(ctx: MatchupContext, diffs: list[dict]) -> str:
            return custom_narrative

        explainer = CompetitionExplainer(features, narrative_hook=hook)
        result = _make_competition_result(sim)
        stories = explainer.generate_pick_stories(result)

        for story in stories:
            assert story["narrative"] == custom_narrative

    def test_hook_returning_none_falls_back_to_generic(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        entities = [f"entity_{i}" for i in range(1, 5)]
        features = _make_entity_features(entities)

        def hook(ctx: MatchupContext, diffs: list[dict]) -> str | None:
            return None

        explainer = CompetitionExplainer(features, narrative_hook=hook)
        result = _make_competition_result(sim)
        stories = explainer.generate_pick_stories(result)

        for story in stories:
            # Generic narrative contains "over" and "confidence"
            assert "over" in story["narrative"]
            assert "confidence" in story["narrative"]

    def test_generic_narrative_format(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        entities = [f"entity_{i}" for i in range(1, 5)]
        features = _make_entity_features(entities)
        explainer = CompetitionExplainer(features)

        result = _make_competition_result(sim)
        stories = explainer.generate_pick_stories(result)

        for story in stories:
            narr = story["narrative"]
            assert "over" in narr
            assert "confidence" in narr
            assert "models favor" in narr

    def test_matchup_as_dict(self):
        """matchups can contain plain dicts instead of MatchupContext."""
        ctx = MatchupContext(
            slot="G1",
            round_num=1,
            entity_a="A",
            entity_b="B",
            prob_a=0.7,
            pick="A",
            model_probs={},
        )
        result = CompetitionResult(
            picks={"G1": "A"},
            matchups={"G1": ctx.model_dump()},
        )
        df = pd.DataFrame(
            [
                {"entity": "A", "x": 5.0},
                {"entity": "B", "x": 3.0},
            ]
        )
        explainer = CompetitionExplainer(df)
        stories = explainer.generate_pick_stories(result)
        assert len(stories) == 1
        assert stories[0]["pick"] == "A"


# ---------------------------------------------------------------------------
# Tests: generate_entity_profiles
# ---------------------------------------------------------------------------


class TestGenerateEntityProfiles:
    def test_top_n_sorted_by_champion_prob(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        round_probs = sim.entity_round_probabilities(n_sims=500, seed=0)
        entities = [f"entity_{i}" for i in range(1, 5)]
        features = _make_entity_features(entities)
        explainer = CompetitionExplainer(features)

        profiles = explainer.generate_entity_profiles(sim, round_probs, top_n=2)

        assert len(profiles) == 2
        # Should be sorted descending by champion_prob
        assert profiles[0]["champion_prob"] >= profiles[1]["champion_prob"]
        # entity_1 is dominant, should be first
        assert profiles[0]["entity"] == "entity_1"

    def test_profile_contains_expected_keys(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        round_probs = sim.entity_round_probabilities(n_sims=100, seed=0)
        entities = [f"entity_{i}" for i in range(1, 5)]
        features = _make_entity_features(entities)
        explainer = CompetitionExplainer(features)

        profiles = explainer.generate_entity_profiles(sim, round_probs, top_n=4)

        for p in profiles:
            assert "entity" in p
            assert "seed" in p
            assert "champion_prob" in p
            assert "round_probs" in p
            assert isinstance(p["round_probs"], dict)

    def test_round_probs_includes_all_rounds(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        round_probs = sim.entity_round_probabilities(n_sims=100, seed=0)
        entities = [f"entity_{i}" for i in range(1, 5)]
        features = _make_entity_features(entities)
        explainer = CompetitionExplainer(features)

        profiles = explainer.generate_entity_profiles(sim, round_probs, top_n=1)
        rp = profiles[0]["round_probs"]
        # 4-team bracket has 2 rounds: round_1 and champion
        assert "round_1" in rp
        assert "champion" in rp

    def test_seed_lookup(self):
        _, _, _, sim = _make_4team_setup(dominant="entity_1")
        round_probs = sim.entity_round_probabilities(n_sims=100, seed=0)
        entities = [f"entity_{i}" for i in range(1, 5)]
        features = _make_entity_features(entities)
        explainer = CompetitionExplainer(features)

        profiles = explainer.generate_entity_profiles(sim, round_probs, top_n=4)
        seeds_found = {p["entity"]: p["seed"] for p in profiles}
        assert seeds_found["entity_1"] == "S1"

    def test_empty_round_probs(self):
        _, _, _, sim = _make_4team_setup()
        df = pd.DataFrame(columns=["entity", "champion"])
        features = _make_entity_features(["entity_1"])
        explainer = CompetitionExplainer(features)

        profiles = explainer.generate_entity_profiles(sim, df, top_n=5)
        assert profiles == []


# ---------------------------------------------------------------------------
# Tests: empty / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_numeric_features(self):
        df = pd.DataFrame(
            [
                {"entity": "A", "name": "Alpha"},
                {"entity": "B", "name": "Beta"},
            ]
        )
        explainer = CompetitionExplainer(df)
        assert explainer.compute_differentials("A", "B") == []

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        explainer = CompetitionExplainer(df)
        assert explainer.compute_differentials("A", "B") == []

    def test_empty_matchups_result(self):
        df = pd.DataFrame([{"entity": "A", "x": 1.0}])
        explainer = CompetitionExplainer(df)
        result = CompetitionResult()
        stories = explainer.generate_pick_stories(result)
        assert stories == []
