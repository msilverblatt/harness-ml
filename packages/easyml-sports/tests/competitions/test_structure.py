"""Tests for competition structure building."""

from __future__ import annotations

import math

import pytest

from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    GroupConfig,
)
from easyml.sports.competitions.structure import (
    _standard_bracket_order,
    build_structure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seed_map(n: int) -> dict[str, str]:
    """Create a seed-to-entity mapping for *n* participants."""
    return {f"S{i}": f"entity_{i}" for i in range(1, n + 1)}


# ---------------------------------------------------------------------------
# TestSingleElimination
# ---------------------------------------------------------------------------


class TestSingleElimination:
    """Single-elimination bracket tests."""

    def test_4_participants(self):
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=4,
        )
        s = build_structure(cfg, _make_seed_map(4))
        # 4 participants = 2 rounds, 3 games total (2 + 1)
        assert len(s.slots) == 3
        assert len(s.round_slots[1]) == 2
        assert len(s.round_slots[2]) == 1

    def test_8_participants(self):
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=8,
        )
        s = build_structure(cfg, _make_seed_map(8))
        # 8 participants = 3 rounds, 7 games (4+2+1)
        assert len(s.slots) == 7
        assert len(s.round_slots[1]) == 4
        assert len(s.round_slots[2]) == 2
        assert len(s.round_slots[3]) == 1

    def test_16_participants(self):
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=16,
        )
        s = build_structure(cfg, _make_seed_map(16))
        # 16 participants = 4 rounds, 15 games
        assert len(s.slots) == 15
        assert len(s.round_slots[1]) == 8
        assert len(s.round_slots[4]) == 1

    def test_64_participants(self):
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=64,
        )
        s = build_structure(cfg, _make_seed_map(64))
        # 64 participants = 6 rounds, 63 games
        assert len(s.slots) == 63
        assert len(s.round_slots[1]) == 32
        assert len(s.round_slots[6]) == 1

    def test_seeding_1_vs_n(self):
        """Seed 1 should play seed N in round 1."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=8,
        )
        s = build_structure(cfg, _make_seed_map(8))
        # Find the R1 matchup containing S1
        r1_matchups = {slot: s.slot_matchups[slot] for slot in s.round_slots[1]}
        s1_matchup = [m for m in r1_matchups.values() if "S1" in m]
        assert len(s1_matchup) == 1
        assert s1_matchup[0] == ("S1", "S8")

    def test_seed_entity_mapping(self):
        """seed_to_entity and entity_to_seed should be populated."""
        seed_map = _make_seed_map(4)
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=4,
        )
        s = build_structure(cfg, seed_map)
        assert s.seed_to_entity == seed_map
        assert s.entity_to_seed == {v: k for k, v in seed_map.items()}

    def test_r1_refs_are_seeds(self):
        """Round 1 matchup refs should be seed codes (S1, S2, ...)."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=8,
        )
        s = build_structure(cfg, _make_seed_map(8))
        for slot in s.round_slots[1]:
            a, b = s.slot_matchups[slot]
            assert a.startswith("S"), f"R1 ref {a} should be a seed code"
            assert b.startswith("S"), f"R1 ref {b} should be a seed code"

    def test_later_rounds_ref_slots(self):
        """Round 2+ matchup refs should be prior slot names."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=8,
        )
        s = build_structure(cfg, _make_seed_map(8))
        for r in range(2, 4):
            for slot in s.round_slots[r]:
                a, b = s.slot_matchups[slot]
                assert a.startswith("R"), f"R{r} ref {a} should be a slot name"
                assert b.startswith("R"), f"R{r} ref {b} should be a slot name"

    def test_byes_for_non_power_of_2(self):
        """Non-power-of-2 participants should produce BYE matchups."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=6,
        )
        s = build_structure(cfg, _make_seed_map(6))
        # Bracket size = 8, so 2 byes
        bye_slots = [
            slot for slot in s.round_slots[1]
            if "BYE" in s.slot_matchups[slot]
        ]
        assert len(bye_slots) == 2
        # Total slots: 8-1 = 7 (full bracket)
        assert len(s.slots) == 7

    def test_bracket_order_8(self):
        """Standard bracket order for 8."""
        order = _standard_bracket_order(8)
        assert order == [1, 8, 4, 5, 2, 7, 3, 6]

    def test_bracket_order_4(self):
        order = _standard_bracket_order(4)
        assert order == [1, 4, 2, 3]

    def test_bracket_order_ensures_1v2_in_final(self):
        """Seeds 1 and 2 should be in opposite halves of the bracket."""
        order = _standard_bracket_order(16)
        # Seeds 1 and 2 should be in different halves
        first_half = set(order[:8])
        second_half = set(order[8:])
        assert 1 in first_half
        assert 2 in second_half


# ---------------------------------------------------------------------------
# TestRoundRobin
# ---------------------------------------------------------------------------


class TestRoundRobin:
    """Round-robin tests."""

    def test_4_participants_single(self):
        """4 participants, single round-robin = C(4,2) = 6 games."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.round_robin,
            n_participants=4,
        )
        s = build_structure(cfg, _make_seed_map(4))
        assert len(s.slots) == 6
        # All matchups should reference seeds
        for slot in s.slots:
            a, b = s.slot_matchups[slot]
            assert a.startswith("S")
            assert b.startswith("S")

    def test_4_participants_double(self):
        """4 participants, double round-robin = 12 games."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.round_robin,
            n_participants=4,
            n_rounds=2,
        )
        s = build_structure(cfg, _make_seed_map(4))
        assert len(s.slots) == 12

    def test_all_matchups_ref_seeds(self):
        """All round-robin matchups should reference seed codes."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.round_robin,
            n_participants=5,
        )
        s = build_structure(cfg, _make_seed_map(5))
        for slot in s.slots:
            a, b = s.slot_matchups[slot]
            assert a.startswith("S")
            assert b.startswith("S")

    def test_default_rounds_is_1(self):
        """Default n_rounds should be 1."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.round_robin,
            n_participants=3,
        )
        s = build_structure(cfg, _make_seed_map(3))
        # C(3,2) = 3 games for 1 round
        assert len(s.slots) == 3

    def test_slot_naming(self):
        """Slots should be named RR{round}G{game}."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.round_robin,
            n_participants=3,
        )
        s = build_structure(cfg, _make_seed_map(3))
        for slot in s.slots:
            assert slot.startswith("RR")


# ---------------------------------------------------------------------------
# TestSwiss
# ---------------------------------------------------------------------------


class TestSwiss:
    """Swiss-system tests."""

    def test_8_participants_3_rounds(self):
        """8 participants, 3 rounds = 4 games per round = 12 total."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.swiss,
            n_participants=8,
            n_rounds=3,
        )
        s = build_structure(cfg, _make_seed_map(8))
        assert len(s.slots) == 12

    def test_round_slot_counts(self):
        """Each round should have n/2 games."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.swiss,
            n_participants=8,
            n_rounds=3,
        )
        s = build_structure(cfg, _make_seed_map(8))
        for r in range(1, 4):
            assert len(s.round_slots[r]) == 4

    def test_first_round_seeded(self):
        """Round 1 should pair top half vs bottom half by seed."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.swiss,
            n_participants=8,
            n_rounds=3,
        )
        s = build_structure(cfg, _make_seed_map(8))
        r1 = s.round_slots[1]
        expected = [("S1", "S5"), ("S2", "S6"), ("S3", "S7"), ("S4", "S8")]
        actual = [s.slot_matchups[slot] for slot in r1]
        assert actual == expected

    def test_later_rounds_tbd(self):
        """Rounds 2+ should have TBD placeholders."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.swiss,
            n_participants=8,
            n_rounds=3,
        )
        s = build_structure(cfg, _make_seed_map(8))
        for r in range(2, 4):
            for slot in s.round_slots[r]:
                a, b = s.slot_matchups[slot]
                assert a == "TBD"
                assert b == "TBD"

    def test_slot_naming(self):
        """Slots should be named SW{round}G{game}."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.swiss,
            n_participants=8,
            n_rounds=3,
        )
        s = build_structure(cfg, _make_seed_map(8))
        for slot in s.slots:
            assert slot.startswith("SW")


# ---------------------------------------------------------------------------
# TestGroupKnockout
# ---------------------------------------------------------------------------


class TestGroupKnockout:
    """Group + knockout tests."""

    def test_16_participants_4_groups(self):
        """16 participants, 4 groups of 4, advance 2."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.group_knockout,
            n_participants=16,
            groups=GroupConfig(n_groups=4, group_size=4, advance=2),
        )
        s = build_structure(cfg, _make_seed_map(16))
        # Group stage: 4 groups * C(4,2) = 4 * 6 = 24 games
        group_slots = s.round_slots[0]
        assert len(group_slots) == 24

    def test_knockout_slot_count(self):
        """8 qualifiers = 3 KO rounds = 7 KO games."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.group_knockout,
            n_participants=16,
            groups=GroupConfig(n_groups=4, group_size=4, advance=2),
        )
        s = build_structure(cfg, _make_seed_map(16))
        # 8 qualifiers -> bracket_size=8, 3 rounds, 7 games
        ko_slots = [sl for sl in s.slots if sl.startswith("R")]
        assert len(ko_slots) == 7

    def test_group_prefix_naming(self):
        """Group stage slots should start with G{group}M."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.group_knockout,
            n_participants=16,
            groups=GroupConfig(n_groups=4, group_size=4, advance=2),
        )
        s = build_structure(cfg, _make_seed_map(16))
        group_slots = s.round_slots[0]
        for slot in group_slots:
            assert slot[0] == "G"
            assert "M" in slot

    def test_total_slot_count(self):
        """Total slots = group stage + knockout stage."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.group_knockout,
            n_participants=16,
            groups=GroupConfig(n_groups=4, group_size=4, advance=2),
        )
        s = build_structure(cfg, _make_seed_map(16))
        # 24 group + 7 KO = 31
        assert len(s.slots) == 31

    def test_snake_seeding(self):
        """Snake seeding distributes top seeds across groups."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.group_knockout,
            n_participants=8,
            groups=GroupConfig(n_groups=2, group_size=4, advance=2),
        )
        s = build_structure(cfg, _make_seed_map(8))
        # Group 1 should have seeds 1,4,5,8 (snake: [1,2] then [4,3] then [5,6] then [8,7])
        # Actually: forward [1,2], reverse [4,3], forward [5,6], reverse [8,7]
        # Group 1: S1, S4, S5, S8; Group 2: S2, S3, S6, S7
        g1_slots = [sl for sl in s.round_slots[0] if sl.startswith("G1")]
        g1_seeds = set()
        for sl in g1_slots:
            a, b = s.slot_matchups[sl]
            g1_seeds.add(a)
            g1_seeds.add(b)
        assert g1_seeds == {"S1", "S4", "S5", "S8"}


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_2_participants(self):
        """Minimal bracket: 2 participants = 1 game."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=2,
        )
        s = build_structure(cfg, _make_seed_map(2))
        assert len(s.slots) == 1
        assert s.slot_matchups["R1G1"] == ("S1", "S2")

    def test_config_preserved(self):
        """The original config should be stored in the structure."""
        cfg = CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=8,
        )
        s = build_structure(cfg, _make_seed_map(8))
        assert s.config is cfg
        assert s.config.format == CompetitionFormat.single_elimination
        assert s.config.n_participants == 8

    def test_double_elimination_not_implemented(self):
        cfg = CompetitionConfig(
            format=CompetitionFormat.double_elimination,
            n_participants=8,
        )
        with pytest.raises(NotImplementedError):
            build_structure(cfg, _make_seed_map(8))
