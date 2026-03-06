"""Tests for pre-competition confidence diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.sports.competitions.confidence import (
    compute_feature_outliers,
    compute_model_disagreement,
    generate_confidence_report,
)
from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
)
from easyml.sports.competitions.simulator import CompetitionSimulator
from easyml.sports.competitions.structure import build_structure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed_map(n: int) -> dict[str, str]:
    return {f"S{i}": f"entity_{i}" for i in range(1, n + 1)}


def _make_4team_simulator(
    model_probs: dict[tuple[str, str], dict[str, float]] | None = None,
) -> CompetitionSimulator:
    """Build a 4-entity simulator with optional per-model disagreement."""
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
            prob = 0.6
            m1 = 0.6
            m2 = 0.6
            if model_probs and (a, b) in model_probs:
                mp = model_probs[(a, b)]
                prob = mp.get("ensemble", 0.6)
                m1 = mp.get("model_1", 0.6)
                m2 = mp.get("model_2", 0.6)
            rows.append({
                "entity_a": a,
                "entity_b": b,
                "prob_ensemble": prob,
                "prob_model_1": m1,
                "prob_model_2": m2,
            })
    probs = pd.DataFrame(rows)
    return CompetitionSimulator(cfg, structure, probs)


def _make_training_features(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create synthetic training features with known distribution."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "entity": [f"train_{i}" for i in range(n_rows)],
        "feat_a": rng.normal(50.0, 5.0, n_rows),
        "feat_b": rng.normal(100.0, 10.0, n_rows),
        "feat_c": rng.normal(0.0, 1.0, n_rows),
        "label": [f"cat_{i % 3}" for i in range(n_rows)],  # non-numeric
    })


def _make_entity_features(
    entities: list[str],
    overrides: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Create entity features with optional extreme values."""
    rows = []
    for entity in entities:
        row = {
            "entity": entity,
            "feat_a": 50.0,  # normal
            "feat_b": 100.0,  # normal
            "feat_c": 0.0,  # normal
            "label": "cat_0",  # non-numeric
        }
        if overrides and entity in overrides:
            row.update(overrides[entity])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TestComputeFeatureOutliers
# ---------------------------------------------------------------------------


class TestComputeFeatureOutliers:
    """Tests for compute_feature_outliers."""

    def test_flags_extreme_values(self):
        training = _make_training_features()
        # entity_1 has feat_a way above training mean (~50), z > 2
        entities = _make_entity_features(
            ["entity_1", "entity_2"],
            overrides={"entity_1": {"feat_a": 80.0}},
        )
        outliers = compute_feature_outliers(
            entities, ["entity_1", "entity_2"], training,
        )
        assert len(outliers) > 0
        # entity_1, feat_a should be flagged
        flagged = [(o["entity"], o["feature"]) for o in outliers]
        assert ("entity_1", "feat_a") in flagged

    def test_auto_discovers_numeric_columns(self):
        """Should find feat_a, feat_b, feat_c but not 'label' or 'entity'."""
        training = _make_training_features()
        entities = _make_entity_features(
            ["entity_1"],
            overrides={"entity_1": {"feat_a": 200.0, "feat_b": 300.0, "feat_c": 20.0}},
        )
        outliers = compute_feature_outliers(
            entities, ["entity_1"], training,
        )
        features_flagged = {o["feature"] for o in outliers}
        # All three numeric features should be flagged with extreme values
        assert "feat_a" in features_flagged
        assert "feat_b" in features_flagged
        assert "feat_c" in features_flagged
        # Non-numeric columns should never appear
        assert "label" not in features_flagged
        assert "entity" not in features_flagged

    def test_sorted_by_abs_z_score(self):
        training = _make_training_features()
        # Give entity_1 two outliers with different magnitudes
        entities = _make_entity_features(
            ["entity_1"],
            overrides={"entity_1": {"feat_a": 200.0, "feat_c": 5.0}},
        )
        outliers = compute_feature_outliers(
            entities, ["entity_1"], training,
        )
        # Should be sorted descending by |z_score|
        z_scores = [abs(o["z_score"]) for o in outliers]
        assert z_scores == sorted(z_scores, reverse=True)

    def test_stricter_threshold(self):
        training = _make_training_features()
        # feat_a at 65 is ~3 std devs from mean=50, std=5
        entities = _make_entity_features(
            ["entity_1"],
            overrides={"entity_1": {"feat_a": 65.0}},
        )
        outliers_z2 = compute_feature_outliers(
            entities, ["entity_1"], training, z_threshold=2.0,
        )
        outliers_z3 = compute_feature_outliers(
            entities, ["entity_1"], training, z_threshold=3.0,
        )
        # z=2 should flag it, z=3 should be stricter (fewer or no outliers)
        assert len(outliers_z3) <= len(outliers_z2)

    def test_no_outliers_returns_empty(self):
        training = _make_training_features()
        # All values are at training mean — no outliers
        entities = _make_entity_features(["entity_1"])
        outliers = compute_feature_outliers(
            entities, ["entity_1"], training,
        )
        assert outliers == []

    def test_empty_target_entities(self):
        training = _make_training_features()
        entities = _make_entity_features(["entity_1"])
        outliers = compute_feature_outliers(
            entities, [], training,
        )
        assert outliers == []

    def test_outlier_dict_keys(self):
        training = _make_training_features()
        entities = _make_entity_features(
            ["entity_1"],
            overrides={"entity_1": {"feat_a": 200.0}},
        )
        outliers = compute_feature_outliers(
            entities, ["entity_1"], training,
        )
        assert len(outliers) > 0
        expected_keys = {"entity", "feature", "value", "z_score", "training_mean", "training_std"}
        assert set(outliers[0].keys()) == expected_keys


# ---------------------------------------------------------------------------
# TestComputeModelDisagreement
# ---------------------------------------------------------------------------


class TestComputeModelDisagreement:
    """Tests for compute_model_disagreement."""

    def test_returns_first_round_matchups(self):
        sim = _make_4team_simulator()
        results = compute_model_disagreement(sim)
        # 4-team bracket has 2 first-round matchups (both refs are seed codes)
        assert len(results) == 2
        for r in results:
            assert r["round"] == 1

    def test_sorted_by_agreement_ascending(self):
        # Create disagreement on one matchup
        model_probs = {
            ("entity_1", "entity_4"): {"ensemble": 0.6, "model_1": 0.9, "model_2": 0.3},
            ("entity_2", "entity_3"): {"ensemble": 0.6, "model_1": 0.6, "model_2": 0.6},
        }
        sim = _make_4team_simulator(model_probs=model_probs)
        results = compute_model_disagreement(sim)
        agreements = [r["agreement"] for r in results]
        assert agreements == sorted(agreements)

    def test_excludes_non_seed_refs(self):
        """Round 2 matchups use slot refs, not seed codes — should be excluded."""
        sim = _make_4team_simulator()
        results = compute_model_disagreement(sim)
        slots = [r["slot"] for r in results]
        # No round-2 slots should appear
        for slot in slots:
            assert not slot.startswith("R2")

    def test_result_dict_keys(self):
        sim = _make_4team_simulator()
        results = compute_model_disagreement(sim)
        assert len(results) > 0
        expected_keys = {"slot", "round", "entity_a", "entity_b", "agreement", "prob_ensemble"}
        assert set(results[0].keys()) == expected_keys

    def test_top_n_limits_results(self):
        sim = _make_4team_simulator()
        results = compute_model_disagreement(sim, top_n=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# TestGenerateConfidenceReport
# ---------------------------------------------------------------------------


class TestGenerateConfidenceReport:
    """Tests for generate_confidence_report."""

    def test_has_both_keys(self):
        sim = _make_4team_simulator()
        training = _make_training_features()
        entities = _make_entity_features(
            [f"entity_{i}" for i in range(1, 5)],
            overrides={"entity_1": {"feat_a": 200.0}},
        )
        report = generate_confidence_report(sim, entities, training)
        assert "feature_outliers" in report
        assert "high_disagreement" in report

    def test_report_with_outliers(self):
        sim = _make_4team_simulator()
        training = _make_training_features()
        entities = _make_entity_features(
            [f"entity_{i}" for i in range(1, 5)],
            overrides={"entity_1": {"feat_a": 200.0}},
        )
        report = generate_confidence_report(sim, entities, training)
        assert len(report["feature_outliers"]) > 0

    def test_report_with_no_outliers(self):
        sim = _make_4team_simulator()
        training = _make_training_features()
        entities = _make_entity_features(
            [f"entity_{i}" for i in range(1, 5)],
        )
        report = generate_confidence_report(sim, entities, training)
        assert report["feature_outliers"] == []
        assert isinstance(report["high_disagreement"], list)
