"""Tests for post-model probability adjustments."""

from __future__ import annotations

import pandas as pd
import pytest

from easyml.sports.competitions.adjustments import apply_adjustments
from easyml.sports.competitions.schemas import AdjustmentConfig


def _make_probs(**overrides) -> pd.DataFrame:
    """Build a simple probabilities DataFrame."""
    data = {
        "entity_a": ["TeamA", "TeamC"],
        "entity_b": ["TeamB", "TeamD"],
        "prob": [0.6, 0.5],
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestNoAdjustments:
    def test_returns_copy(self):
        probs = _make_probs()
        adj = AdjustmentConfig()
        result, log = apply_adjustments(probs, adj)
        pd.testing.assert_frame_equal(result, probs)
        assert result is not probs
        assert log == []


class TestEntityMultipliers:
    def test_increases_prob_when_entity_is_a(self):
        probs = _make_probs()
        adj = AdjustmentConfig(entity_multipliers={"TeamA": 1.2})
        result, log = apply_adjustments(probs, adj)
        # TeamA is entity_a in row 0: prob *= 1.2 => 0.6 * 1.2 = 0.72
        assert result.loc[0, "prob"] == pytest.approx(0.72)
        # Row 1 (TeamC vs TeamD) unchanged
        assert result.loc[1, "prob"] == pytest.approx(0.5)
        assert len(log) == 1
        assert log[0]["type"] == "entity_multiplier"
        assert log[0]["side"] == "entity_a"

    def test_decreases_prob_when_entity_is_a(self):
        probs = _make_probs()
        adj = AdjustmentConfig(entity_multipliers={"TeamA": 0.8})
        result, _ = apply_adjustments(probs, adj)
        # 0.6 * 0.8 = 0.48
        assert result.loc[0, "prob"] == pytest.approx(0.48)

    def test_multiplier_when_entity_is_b(self):
        probs = _make_probs()
        # TeamB is entity_b in row 0: prob = 1 - (1 - 0.6) * 1.5 = 1 - 0.6 = 0.4
        adj = AdjustmentConfig(entity_multipliers={"TeamB": 1.5})
        result, log = apply_adjustments(probs, adj)
        expected = 1 - (1 - 0.6) * 1.5  # = 0.4
        assert result.loc[0, "prob"] == pytest.approx(expected)
        assert log[0]["side"] == "entity_b"

    def test_multiplier_both_sides(self):
        """Entity appears as entity_a in one row and entity_b in another."""
        probs = pd.DataFrame(
            {
                "entity_a": ["TeamX", "TeamY"],
                "entity_b": ["TeamY", "TeamZ"],
                "prob": [0.7, 0.4],
            }
        )
        adj = AdjustmentConfig(entity_multipliers={"TeamY": 1.3})
        result, log = apply_adjustments(probs, adj)
        # Row 0: TeamY is entity_b => prob = 1 - (1-0.7)*1.3 = 1 - 0.39 = 0.61
        assert result.loc[0, "prob"] == pytest.approx(0.61)
        # Row 1: TeamY is entity_a => prob = 0.4 * 1.3 = 0.52
        assert result.loc[1, "prob"] == pytest.approx(0.52)
        assert len(log) == 2


class TestExternalBlending:
    def test_blends_with_external(self):
        probs = _make_probs()
        ext = pd.DataFrame(
            {
                "entity_a": ["TeamA"],
                "entity_b": ["TeamB"],
                "prob": [0.8],
            }
        )
        adj = AdjustmentConfig(external_weight=0.5)
        result, log = apply_adjustments(probs, adj, external_probabilities=ext)
        # (1-0.5)*0.6 + 0.5*0.8 = 0.3 + 0.4 = 0.7
        assert result.loc[0, "prob"] == pytest.approx(0.7)
        # Row 1 has no external match, unchanged
        assert result.loc[1, "prob"] == pytest.approx(0.5)
        assert len(log) == 1
        assert log[0]["type"] == "external_blend"

    def test_no_blend_when_weight_zero(self):
        probs = _make_probs()
        ext = pd.DataFrame(
            {
                "entity_a": ["TeamA"],
                "entity_b": ["TeamB"],
                "prob": [0.8],
            }
        )
        adj = AdjustmentConfig(external_weight=0.0)
        result, log = apply_adjustments(probs, adj, external_probabilities=ext)
        assert result.loc[0, "prob"] == pytest.approx(0.6)
        assert log == []


class TestHardOverrides:
    def test_override_sets_exact_prob(self):
        probs = _make_probs()
        adj = AdjustmentConfig(probability_overrides={"TeamA_TeamB": 0.85})
        result, log = apply_adjustments(probs, adj)
        assert result.loc[0, "prob"] == pytest.approx(0.85)
        assert len(log) == 1
        assert log[0]["type"] == "override"
        assert log[0]["old_prob"] == pytest.approx(0.6)

    def test_override_no_match(self):
        probs = _make_probs()
        adj = AdjustmentConfig(probability_overrides={"Foo_Bar": 0.9})
        result, log = apply_adjustments(probs, adj)
        pd.testing.assert_frame_equal(result, probs)
        assert log == []


class TestClamping:
    def test_clamp_to_max(self):
        probs = pd.DataFrame(
            {
                "entity_a": ["TeamA"],
                "entity_b": ["TeamB"],
                "prob": [0.9],
            }
        )
        adj = AdjustmentConfig(entity_multipliers={"TeamA": 2.0})
        result, _ = apply_adjustments(probs, adj)
        # 0.9 * 2.0 = 1.8, clamped to 0.99
        assert result.loc[0, "prob"] == pytest.approx(0.99)

    def test_clamp_to_min(self):
        probs = pd.DataFrame(
            {
                "entity_a": ["TeamA"],
                "entity_b": ["TeamB"],
                "prob": [0.05],
            }
        )
        adj = AdjustmentConfig(entity_multipliers={"TeamA": 0.1})
        result, _ = apply_adjustments(probs, adj)
        # 0.05 * 0.1 = 0.005, clamped to 0.01
        assert result.loc[0, "prob"] == pytest.approx(0.01)

    def test_override_clamped(self):
        probs = _make_probs()
        adj = AdjustmentConfig(probability_overrides={"TeamA_TeamB": 1.5})
        result, log = apply_adjustments(probs, adj)
        assert result.loc[0, "prob"] == pytest.approx(0.99)

    def test_override_clamped_low(self):
        probs = _make_probs()
        adj = AdjustmentConfig(probability_overrides={"TeamA_TeamB": -0.5})
        result, _ = apply_adjustments(probs, adj)
        assert result.loc[0, "prob"] == pytest.approx(0.01)


class TestCombinedAdjustments:
    def test_all_three_applied_in_order(self):
        """Multiplier -> blend -> override, each builds on prior result."""
        probs = pd.DataFrame(
            {
                "entity_a": ["TeamA", "TeamC"],
                "entity_b": ["TeamB", "TeamD"],
                "prob": [0.6, 0.5],
            }
        )
        ext = pd.DataFrame(
            {
                "entity_a": ["TeamA"],
                "entity_b": ["TeamB"],
                "prob": [0.9],
            }
        )
        adj = AdjustmentConfig(
            entity_multipliers={"TeamA": 1.1},
            external_weight=0.4,
            probability_overrides={"TeamC_TeamD": 0.75},
        )
        result, log = apply_adjustments(probs, adj, external_probabilities=ext)

        # Row 0 (TeamA vs TeamB):
        #   Step 1: 0.6 * 1.1 = 0.66
        #   Step 2: (1-0.4)*0.66 + 0.4*0.9 = 0.396 + 0.36 = 0.756
        assert result.loc[0, "prob"] == pytest.approx(0.756)

        # Row 1 (TeamC vs TeamD):
        #   Step 1: no multiplier, stays 0.5
        #   Step 2: no external match, stays 0.5
        #   Step 3: override to 0.75
        assert result.loc[1, "prob"] == pytest.approx(0.75)

        # Should have 3 log entries: 1 multiplier + 1 blend + 1 override
        types = [entry["type"] for entry in log]
        assert types.count("entity_multiplier") == 1
        assert types.count("external_blend") == 1
        assert types.count("override") == 1
