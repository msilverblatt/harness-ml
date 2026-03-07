"""Build competition structures from config.

Dispatches by ``CompetitionFormat`` to produce a fully populated
``CompetitionStructure`` with slots, matchups, and round mappings.
"""

from __future__ import annotations

import math
from itertools import combinations

from harnessml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionStructure,
    GroupConfig,
)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_structure(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> CompetitionStructure:
    """Build a :class:`CompetitionStructure` from *config* and seed mapping.

    Parameters
    ----------
    config:
        The competition configuration describing format, participants, etc.
    seed_to_entity:
        Mapping of seed codes (``"S1"``, ``"S2"``, ...) to entity IDs.

    Returns
    -------
    CompetitionStructure
        Fully populated structure with slots and matchups.
    """
    entity_to_seed = {v: k for k, v in seed_to_entity.items()}

    builders = {
        CompetitionFormat.single_elimination: _build_single_elimination,
        CompetitionFormat.double_elimination: _build_double_elimination,
        CompetitionFormat.round_robin: _build_round_robin,
        CompetitionFormat.swiss: _build_swiss,
        CompetitionFormat.group_knockout: _build_group_knockout,
    }

    builder = builders[config.format]
    slots, slot_matchups, slot_to_round, round_slots = builder(config, seed_to_entity)

    return CompetitionStructure(
        config=config,
        slots=slots,
        slot_matchups=slot_matchups,
        slot_to_round=slot_to_round,
        round_slots=round_slots,
        seed_to_entity=seed_to_entity,
        entity_to_seed=entity_to_seed,
    )


# ---------------------------------------------------------------------------
# Bracket seed ordering
# ---------------------------------------------------------------------------


def _standard_bracket_order(n: int) -> list[int]:
    """Return the standard bracket seed ordering for *n* participants.

    For *n* = 8 the result is ``[1, 8, 4, 5, 2, 7, 3, 6]`` which ensures
    that seeds 1 and 2 can only meet in the final.

    *n* must be a power of 2.
    """
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2]
    half = _standard_bracket_order(n // 2)
    result: list[int] = []
    for seed in half:
        result.append(seed)
        result.append(n + 1 - seed)
    return result


# ---------------------------------------------------------------------------
# Single elimination
# ---------------------------------------------------------------------------


def _build_single_elimination(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    n = config.n_participants
    # Next power of 2
    bracket_size = 1 << (n - 1).bit_length()
    n_rounds = int(math.log2(bracket_size))

    order = _standard_bracket_order(bracket_size)

    slots: list[str] = []
    slot_matchups: dict[str, tuple[str, str]] = {}
    slot_to_round: dict[str, int] = {}
    round_slots: dict[int, list[str]] = {}

    # Round 1
    r1_slots: list[str] = []
    game = 0
    for i in range(0, bracket_size, 2):
        game += 1
        slot = f"R1G{game}"
        seed_a = order[i]
        seed_b = order[i + 1]
        ref_a = f"S{seed_a}"
        ref_b = f"S{seed_b}"

        # Handle byes: if a seed > n, the other seed gets a bye
        if seed_a > n and seed_b > n:
            # Both are byes — shouldn't happen with proper bracket but skip
            continue
        if seed_b > n:
            # seed_a gets a bye — this slot is a bye, mark with BYE
            slot_matchups[slot] = (ref_a, "BYE")
        elif seed_a > n:
            slot_matchups[slot] = ("BYE", ref_b)
        else:
            slot_matchups[slot] = (ref_a, ref_b)

        slots.append(slot)
        slot_to_round[slot] = 1
        r1_slots.append(slot)

    round_slots[1] = r1_slots

    # Subsequent rounds
    prev_slots = r1_slots
    for r in range(2, n_rounds + 1):
        r_slots: list[str] = []
        game = 0
        for i in range(0, len(prev_slots), 2):
            game += 1
            slot = f"R{r}G{game}"
            ref_a = prev_slots[i]
            ref_b = prev_slots[i + 1]
            slot_matchups[slot] = (ref_a, ref_b)
            slots.append(slot)
            slot_to_round[slot] = r
            r_slots.append(slot)
        round_slots[r] = r_slots
        prev_slots = r_slots

    return slots, slot_matchups, slot_to_round, round_slots


# ---------------------------------------------------------------------------
# Round robin
# ---------------------------------------------------------------------------


def _build_round_robin(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    n = config.n_participants
    n_rounds = config.n_rounds if config.n_rounds is not None else 1

    seeds = [f"S{i}" for i in range(1, n + 1)]
    matchup_pairs = list(combinations(seeds, 2))

    slots: list[str] = []
    slot_matchups: dict[str, tuple[str, str]] = {}
    slot_to_round: dict[str, int] = {}
    round_slots: dict[int, list[str]] = {}

    for rr_round in range(1, n_rounds + 1):
        r_slots: list[str] = []
        for game_idx, (a, b) in enumerate(matchup_pairs, 1):
            slot = f"RR{rr_round}G{game_idx}"
            slot_matchups[slot] = (a, b)
            slots.append(slot)
            slot_to_round[slot] = 0  # round-robin uses round 0
            r_slots.append(slot)
        round_slots[rr_round] = r_slots

    return slots, slot_matchups, slot_to_round, round_slots


# ---------------------------------------------------------------------------
# Swiss
# ---------------------------------------------------------------------------


def _build_swiss(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    n = config.n_participants
    n_rounds = config.n_rounds if config.n_rounds is not None else 3

    slots: list[str] = []
    slot_matchups: dict[str, tuple[str, str]] = {}
    slot_to_round: dict[str, int] = {}
    round_slots: dict[int, list[str]] = {}

    # Round 1: seeded pairing (top half vs bottom half)
    half = n // 2
    r1_slots: list[str] = []
    for game in range(1, half + 1):
        slot = f"SW1G{game}"
        ref_a = f"S{game}"
        ref_b = f"S{game + half}"
        slot_matchups[slot] = (ref_a, ref_b)
        slots.append(slot)
        slot_to_round[slot] = 1
        r1_slots.append(slot)
    round_slots[1] = r1_slots

    # Later rounds: TBD placeholders
    for r in range(2, n_rounds + 1):
        r_slots: list[str] = []
        for game in range(1, half + 1):
            slot = f"SW{r}G{game}"
            slot_matchups[slot] = ("TBD", "TBD")
            slots.append(slot)
            slot_to_round[slot] = r
            r_slots.append(slot)
        round_slots[r] = r_slots

    return slots, slot_matchups, slot_to_round, round_slots


# ---------------------------------------------------------------------------
# Group + Knockout
# ---------------------------------------------------------------------------


def _build_group_knockout(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    groups_cfg: GroupConfig = config.groups or GroupConfig(
        n_groups=config.n_participants // 4,
        group_size=4,
    )

    n_groups = groups_cfg.n_groups
    group_size = groups_cfg.group_size
    advance = groups_cfg.advance

    # Snake seeding into groups
    # E.g. 16 participants, 4 groups: seeds go 1-4 across groups, then 5-8
    # reversed, etc.
    group_seeds: list[list[str]] = [[] for _ in range(n_groups)]
    seed_idx = 1
    forward = True
    while seed_idx <= n_groups * group_size:
        order = range(n_groups) if forward else range(n_groups - 1, -1, -1)
        for g in order:
            if seed_idx > n_groups * group_size:
                break
            group_seeds[g].append(f"S{seed_idx}")
            seed_idx += 1
        forward = not forward

    slots: list[str] = []
    slot_matchups: dict[str, tuple[str, str]] = {}
    slot_to_round: dict[str, int] = {}
    round_slots: dict[int, list[str]] = {}

    # Group stage: round-robin within each group
    group_stage_slots: list[str] = []
    for g_idx in range(n_groups):
        g_label = g_idx + 1
        matchup_pairs = list(combinations(group_seeds[g_idx], 2))
        for game_idx, (a, b) in enumerate(matchup_pairs, 1):
            slot = f"G{g_label}M{game_idx}"
            slot_matchups[slot] = (a, b)
            slots.append(slot)
            slot_to_round[slot] = 0
            group_stage_slots.append(slot)
    round_slots[0] = group_stage_slots

    # Knockout stage: single elimination for qualifiers
    n_qualifiers = n_groups * advance
    bracket_size = 1 << (n_qualifiers - 1).bit_length()
    n_ko_rounds = int(math.log2(bracket_size))

    # Qualifiers are referenced as group position placeholders
    # e.g. G1_1 = group 1 first place, G1_2 = group 1 second place
    qualifier_refs: list[str] = []
    for pos in range(1, advance + 1):
        for g_idx in range(n_groups):
            qualifier_refs.append(f"G{g_idx + 1}_{pos}")

    order = _standard_bracket_order(bracket_size)

    # KO Round 1
    prev_slots_ko: list[str] = []
    game = 0
    for i in range(0, bracket_size, 2):
        game += 1
        slot = f"R1G{game}"
        seed_a_idx = order[i] - 1
        seed_b_idx = order[i + 1] - 1
        ref_a = qualifier_refs[seed_a_idx] if seed_a_idx < len(qualifier_refs) else "BYE"
        ref_b = qualifier_refs[seed_b_idx] if seed_b_idx < len(qualifier_refs) else "BYE"
        slot_matchups[slot] = (ref_a, ref_b)
        slots.append(slot)
        slot_to_round[slot] = 1
        prev_slots_ko.append(slot)
    round_slots[1] = prev_slots_ko

    # Subsequent KO rounds
    for r in range(2, n_ko_rounds + 1):
        r_slots: list[str] = []
        game = 0
        for i in range(0, len(prev_slots_ko), 2):
            game += 1
            slot = f"R{r}G{game}"
            slot_matchups[slot] = (prev_slots_ko[i], prev_slots_ko[i + 1])
            slots.append(slot)
            slot_to_round[slot] = r
            r_slots.append(slot)
        round_slots[r] = r_slots
        prev_slots_ko = r_slots

    return slots, slot_matchups, slot_to_round, round_slots


# ---------------------------------------------------------------------------
# Double elimination (placeholder)
# ---------------------------------------------------------------------------


def _build_double_elimination(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    raise NotImplementedError("Double elimination is not yet implemented")
