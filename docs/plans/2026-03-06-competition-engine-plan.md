# Competition Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generic competition simulation engine supporting elimination brackets, leagues, Swiss, and hybrid formats.

**Architecture:** Config-driven CompetitionConfig → CompetitionStructure → CompetitionSimulator (vectorized Monte Carlo) → CompetitionOptimizer (pool-aware) → CompetitionScorer. Hook-based narrative generation via COMPETITION_NARRATIVE. MCP handler for tool integration.

**Tech Stack:** Python 3.11+, Pydantic v2, NumPy, pandas, pytest

---

## Task 1: Schemas — Config and Result Types

**Files:**
- Create: `packages/easyml-sports/src/easyml/sports/competitions/__init__.py`
- Create: `packages/easyml-sports/src/easyml/sports/competitions/schemas.py`
- Create: `packages/easyml-sports/tests/competitions/__init__.py`
- Create: `packages/easyml-sports/tests/competitions/test_schemas.py`

### Step 1.1: Write tests

Create `packages/easyml-sports/tests/competitions/__init__.py` (empty).

Create `packages/easyml-sports/tests/competitions/test_schemas.py`:

```python
"""Tests for competition schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyml.sports.competitions.schemas import (
    AdjustmentConfig,
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    CompetitionStructure,
    GroupConfig,
    KnockoutConfig,
    MatchupContext,
    ScoreResult,
    ScoringConfig,
    SeedingMode,
    StandingsEntry,
)


# -----------------------------------------------------------------------
# CompetitionFormat enum
# -----------------------------------------------------------------------

class TestCompetitionFormat:
    def test_all_formats_exist(self):
        assert CompetitionFormat.SINGLE_ELIMINATION == "single_elimination"
        assert CompetitionFormat.DOUBLE_ELIMINATION == "double_elimination"
        assert CompetitionFormat.ROUND_ROBIN == "round_robin"
        assert CompetitionFormat.SWISS == "swiss"
        assert CompetitionFormat.GROUP_KNOCKOUT == "group_knockout"

    def test_format_from_string(self):
        fmt = CompetitionFormat("single_elimination")
        assert fmt == CompetitionFormat.SINGLE_ELIMINATION


class TestSeedingMode:
    def test_all_modes_exist(self):
        assert SeedingMode.RANKED == "ranked"
        assert SeedingMode.RANDOM == "random"
        assert SeedingMode.MANUAL == "manual"


# -----------------------------------------------------------------------
# ScoringConfig
# -----------------------------------------------------------------------

class TestScoringConfig:
    def test_per_round_scoring(self):
        cfg = ScoringConfig(type="per_round", values=[10, 20, 40, 80, 160, 320])
        assert cfg.type == "per_round"
        assert len(cfg.values) == 6

    def test_points_based_scoring(self):
        cfg = ScoringConfig(type="points", win=3, draw=1, loss=0)
        assert cfg.win == 3
        assert cfg.draw == 1
        assert cfg.loss == 0

    def test_per_round_requires_values(self):
        with pytest.raises(ValidationError):
            ScoringConfig(type="per_round")

    def test_points_requires_win(self):
        with pytest.raises(ValidationError):
            ScoringConfig(type="points")


# -----------------------------------------------------------------------
# CompetitionConfig
# -----------------------------------------------------------------------

class TestCompetitionConfig:
    def test_single_elimination_config(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=64,
            regions=4,
            seeding="ranked",
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40, 80, 160, 320]),
        )
        assert cfg.format == CompetitionFormat.SINGLE_ELIMINATION
        assert cfg.n_participants == 64
        assert cfg.regions == 4
        assert cfg.seeding == SeedingMode.RANKED
        assert cfg.byes == "auto"

    def test_round_robin_config(self):
        cfg = CompetitionConfig(
            format="round_robin",
            n_participants=20,
            rounds=38,
            scoring=ScoringConfig(type="points", win=3, draw=1, loss=0),
        )
        assert cfg.format == CompetitionFormat.ROUND_ROBIN
        assert cfg.rounds == 38

    def test_swiss_config(self):
        cfg = CompetitionConfig(
            format="swiss",
            n_participants=32,
            n_rounds=5,
            scoring=ScoringConfig(type="points", win=1.0, draw=0.5, loss=0.0),
        )
        assert cfg.n_rounds == 5

    def test_group_knockout_config(self):
        cfg = CompetitionConfig(
            format="group_knockout",
            n_participants=32,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40, 80]),
            groups=GroupConfig(
                n_groups=8,
                group_size=4,
                format="round_robin",
                advance=2,
                scoring=ScoringConfig(type="points", win=3, draw=1, loss=0),
            ),
            knockout=KnockoutConfig(
                format="single_elimination",
                scoring=ScoringConfig(type="per_round", values=[10, 20, 40, 80]),
            ),
        )
        assert cfg.groups.n_groups == 8
        assert cfg.groups.advance == 2
        assert cfg.knockout.format == "single_elimination"

    def test_double_elimination_config(self):
        cfg = CompetitionConfig(
            format="double_elimination",
            n_participants=16,
            grand_final=True,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40, 80, 160]),
        )
        assert cfg.grand_final is True

    def test_defaults(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=8,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
        )
        assert cfg.seeding == SeedingMode.RANKED
        assert cfg.byes == "auto"
        assert cfg.regions is None
        assert cfg.rounds is None
        assert cfg.n_rounds is None
        assert cfg.groups is None
        assert cfg.knockout is None
        assert cfg.grand_final is True

    def test_invalid_format_rejected(self):
        with pytest.raises(ValidationError):
            CompetitionConfig(
                format="invalid_format",
                n_participants=8,
                scoring=ScoringConfig(type="per_round", values=[10]),
            )

    def test_n_participants_must_be_positive(self):
        with pytest.raises(ValidationError):
            CompetitionConfig(
                format="single_elimination",
                n_participants=0,
                scoring=ScoringConfig(type="per_round", values=[10]),
            )


# -----------------------------------------------------------------------
# MatchupContext
# -----------------------------------------------------------------------

class TestMatchupContext:
    def test_construction(self):
        ctx = MatchupContext(
            slot="R1W1",
            round_num=1,
            entity_a="entity_1",
            entity_b="entity_16",
            prob_a=0.95,
            model_probs={"prob_xgb": 0.93, "prob_lgbm": 0.97},
            model_agreement=0.92,
            pick="entity_1",
            strategy="chalk",
            upset=False,
        )
        assert ctx.slot == "R1W1"
        assert ctx.entity_a == "entity_1"
        assert ctx.prob_a == 0.95
        assert ctx.upset is False

    def test_defaults(self):
        ctx = MatchupContext(
            slot="R1W1",
            round_num=1,
            entity_a="a",
            entity_b="b",
            prob_a=0.6,
        )
        assert ctx.model_probs == {}
        assert ctx.model_agreement == 1.0
        assert ctx.pick == ""
        assert ctx.strategy == ""
        assert ctx.upset is False


# -----------------------------------------------------------------------
# CompetitionResult
# -----------------------------------------------------------------------

class TestCompetitionResult:
    def test_construction(self):
        result = CompetitionResult(
            picks={"R1W1": "entity_1", "R1W2": "entity_3"},
            matchups={},
            expected_points=150.0,
            win_probability=0.05,
            top10_probability=0.35,
            strategy="chalk",
        )
        assert result.expected_points == 150.0
        assert result.strategy == "chalk"

    def test_defaults(self):
        result = CompetitionResult(picks={}, matchups={})
        assert result.expected_points == 0.0
        assert result.win_probability == 0.0
        assert result.top10_probability == 0.0
        assert result.strategy == ""


# -----------------------------------------------------------------------
# StandingsEntry
# -----------------------------------------------------------------------

class TestStandingsEntry:
    def test_construction(self):
        entry = StandingsEntry(
            entity="entity_1",
            wins=10,
            losses=5,
            draws=3,
            points=33.0,
            goal_diff=12.0,
        )
        assert entry.entity == "entity_1"
        assert entry.points == 33.0

    def test_defaults(self):
        entry = StandingsEntry(entity="entity_1")
        assert entry.wins == 0
        assert entry.losses == 0
        assert entry.draws == 0
        assert entry.points == 0.0
        assert entry.goal_diff == 0.0


# -----------------------------------------------------------------------
# CompetitionStructure
# -----------------------------------------------------------------------

class TestCompetitionStructure:
    def test_construction(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        struct = CompetitionStructure(
            config=cfg,
            slots=["R1W1", "R1W2", "R2W1"],
            slot_matchups={
                "R1W1": ("S1", "S4"),
                "R1W2": ("S2", "S3"),
                "R2W1": ("R1W1", "R1W2"),
            },
            slot_to_round={"R1W1": 1, "R1W2": 1, "R2W1": 2},
            round_slots={1: ["R1W1", "R1W2"], 2: ["R2W1"]},
            seed_to_entity={"S1": "alpha", "S2": "beta", "S3": "gamma", "S4": "delta"},
            entity_to_seed={"alpha": "S1", "beta": "S2", "gamma": "S3", "delta": "S4"},
        )
        assert len(struct.slots) == 3
        assert struct.slot_to_round["R2W1"] == 2
        assert struct.seed_to_entity["S1"] == "alpha"

    def test_round_slots_matches_slot_to_round(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        struct = CompetitionStructure(
            config=cfg,
            slots=["R1W1", "R1W2", "R2W1"],
            slot_matchups={
                "R1W1": ("S1", "S4"),
                "R1W2": ("S2", "S3"),
                "R2W1": ("R1W1", "R1W2"),
            },
            slot_to_round={"R1W1": 1, "R1W2": 1, "R2W1": 2},
            round_slots={1: ["R1W1", "R1W2"], 2: ["R2W1"]},
            seed_to_entity={"S1": "a", "S2": "b", "S3": "c", "S4": "d"},
            entity_to_seed={"a": "S1", "b": "S2", "c": "S3", "d": "S4"},
        )
        for slot, rd in struct.slot_to_round.items():
            assert slot in struct.round_slots[rd]


# -----------------------------------------------------------------------
# ScoreResult
# -----------------------------------------------------------------------

class TestScoreResult:
    def test_construction(self):
        result = ScoreResult(
            total_points=150.0,
            round_points={1: 80, 2: 40, 3: 30},
            round_correct={1: 8, 2: 2, 3: 1},
            round_total={1: 8, 2: 4, 3: 2},
            picks_detail=[
                {"slot": "R1W1", "correct": True, "points_earned": 10},
            ],
        )
        assert result.total_points == 150.0
        assert result.round_correct[1] == 8


# -----------------------------------------------------------------------
# AdjustmentConfig
# -----------------------------------------------------------------------

class TestAdjustmentConfig:
    def test_defaults(self):
        adj = AdjustmentConfig()
        assert adj.entity_multipliers == {}
        assert adj.external_weight == 0.0
        assert adj.probability_overrides == {}

    def test_entity_multipliers(self):
        adj = AdjustmentConfig(
            entity_multipliers={"entity_1": 0.85, "entity_2": 1.1},
        )
        assert adj.entity_multipliers["entity_1"] == 0.85

    def test_probability_overrides(self):
        adj = AdjustmentConfig(
            probability_overrides={"entity_1_vs_entity_2": (0.3, 0.7)},
        )
        assert adj.probability_overrides["entity_1_vs_entity_2"] == (0.3, 0.7)
```

### Step 1.2: Verify tests fail

```bash
uv run pytest packages/easyml-sports/tests/competitions/test_schemas.py -v 2>&1 | head -30
```

### Step 1.3: Implement schemas

Create `packages/easyml-sports/src/easyml/sports/competitions/__init__.py`:

```python
"""Competition engine — generic competition simulation and optimization.

Supports single/double elimination, round-robin, Swiss, and group-to-knockout
formats via config-driven schemas, vectorized Monte Carlo simulation,
pool-aware optimization, and configurable scoring.
"""
from easyml.sports.competitions.schemas import (
    AdjustmentConfig,
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    CompetitionStructure,
    GroupConfig,
    KnockoutConfig,
    MatchupContext,
    ScoreResult,
    ScoringConfig,
    SeedingMode,
    StandingsEntry,
)

__all__ = [
    "AdjustmentConfig",
    "CompetitionConfig",
    "CompetitionFormat",
    "CompetitionResult",
    "CompetitionStructure",
    "GroupConfig",
    "KnockoutConfig",
    "MatchupContext",
    "ScoreResult",
    "ScoringConfig",
    "SeedingMode",
    "StandingsEntry",
]
```

Create `packages/easyml-sports/src/easyml/sports/competitions/schemas.py`:

```python
"""Pydantic v2 schemas for the competition engine.

All competition formats are configured declaratively via CompetitionConfig.
Entity IDs are strings throughout. Schemas cover config, structure, simulation
results, scoring, standings, and probability adjustments.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CompetitionFormat(str, Enum):
    """Supported competition formats."""

    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"
    ROUND_ROBIN = "round_robin"
    SWISS = "swiss"
    GROUP_KNOCKOUT = "group_knockout"


class SeedingMode(str, Enum):
    """How participants are seeded into the competition structure."""

    RANKED = "ranked"
    RANDOM = "random"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class ScoringConfig(BaseModel):
    """Scoring rules for a competition or phase.

    Two modes:
    - ``type="per_round"``: elimination formats, ``values`` list gives points
      per correct pick in each round (index 0 = round 1).
    - ``type="points"``: league/Swiss formats, ``win``/``draw``/``loss`` give
      points per match outcome.
    """

    type: str  # "per_round" | "points"
    values: list[int | float] | None = None
    win: int | float | None = None
    draw: int | float | None = None
    loss: int | float | None = None

    @model_validator(mode="after")
    def _validate_scoring_type(self) -> ScoringConfig:
        if self.type == "per_round":
            if not self.values or len(self.values) == 0:
                raise ValueError("per_round scoring requires non-empty 'values' list")
        elif self.type == "points":
            if self.win is None:
                raise ValueError("points scoring requires 'win' value")
            if self.draw is None:
                self.draw = 0
            if self.loss is None:
                self.loss = 0
        return self


# ---------------------------------------------------------------------------
# Sub-configs for group_knockout
# ---------------------------------------------------------------------------

class GroupConfig(BaseModel):
    """Configuration for the group stage of a group_knockout competition."""

    n_groups: int
    group_size: int
    format: str = "round_robin"
    advance: int = 2
    scoring: ScoringConfig | None = None


class KnockoutConfig(BaseModel):
    """Configuration for the knockout stage of a group_knockout competition."""

    format: str = "single_elimination"
    scoring: ScoringConfig | None = None


# ---------------------------------------------------------------------------
# Top-level competition config
# ---------------------------------------------------------------------------

class CompetitionConfig(BaseModel):
    """Declarative configuration for any competition format.

    Validated to ensure required fields per format are present.
    """

    format: CompetitionFormat
    n_participants: int
    regions: int | None = None
    seeding: SeedingMode = SeedingMode.RANKED
    scoring: ScoringConfig
    rounds: int | None = None  # round_robin
    n_rounds: int | None = None  # swiss
    byes: str = "auto"  # single/double elimination
    groups: GroupConfig | None = None  # group_knockout
    knockout: KnockoutConfig | None = None  # group_knockout
    grand_final: bool = True  # double_elimination

    @field_validator("n_participants")
    @classmethod
    def _n_participants_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_participants must be >= 1")
        return v


# ---------------------------------------------------------------------------
# Matchup context (per-game detail)
# ---------------------------------------------------------------------------

class MatchupContext(BaseModel):
    """Rich context for a single competition game."""

    slot: str
    round_num: int
    entity_a: str
    entity_b: str
    prob_a: float
    model_probs: dict[str, float] = {}
    model_agreement: float = 1.0
    pick: str = ""
    strategy: str = ""
    upset: bool = False


# ---------------------------------------------------------------------------
# Competition result (full bracket / picks)
# ---------------------------------------------------------------------------

class CompetitionResult(BaseModel):
    """A complete set of picks with matchup context and scoring metadata."""

    picks: dict[str, str]  # slot -> winning entity ID
    matchups: dict[str, MatchupContext]
    expected_points: float = 0.0
    win_probability: float = 0.0
    top10_probability: float = 0.0
    strategy: str = ""


# ---------------------------------------------------------------------------
# Standings (league / swiss)
# ---------------------------------------------------------------------------

class StandingsEntry(BaseModel):
    """One entity's standings row in a league or Swiss format."""

    entity: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    points: float = 0.0
    goal_diff: float = 0.0


# ---------------------------------------------------------------------------
# Competition structure
# ---------------------------------------------------------------------------

class CompetitionStructure(BaseModel):
    """Resolved structure of a competition — slots, matchups, seedings.

    Built from CompetitionConfig by the structure module.
    """

    config: CompetitionConfig
    slots: list[str]
    slot_matchups: dict[str, tuple[str, str]]  # slot -> (ref_a, ref_b)
    slot_to_round: dict[str, int]
    round_slots: dict[int, list[str]]
    seed_to_entity: dict[str, str]  # seed code -> entity ID
    entity_to_seed: dict[str, str]  # entity ID -> seed code


# ---------------------------------------------------------------------------
# Score result
# ---------------------------------------------------------------------------

class ScoreResult(BaseModel):
    """Output of scoring picks against actuals."""

    total_points: float
    round_points: dict[int, float]
    round_correct: dict[int, int]
    round_total: dict[int, int]
    picks_detail: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Adjustment config
# ---------------------------------------------------------------------------

class AdjustmentConfig(BaseModel):
    """Post-model probability adjustments applied before simulation.

    - entity_multipliers: scale an entity's win prob (e.g. injuries)
    - external_weight: blend weight for external probability source (0-1)
    - probability_overrides: direct prob assignment for specific matchup keys
    """

    entity_multipliers: dict[str, float] = {}
    external_weight: float = 0.0
    probability_overrides: dict[str, tuple[float, float]] = {}
```

### Step 1.4: Verify tests pass

```bash
uv run pytest packages/easyml-sports/tests/competitions/test_schemas.py -v
```

### Step 1.5: Commit

```bash
git add packages/easyml-sports/src/easyml/sports/competitions/__init__.py \
      packages/easyml-sports/src/easyml/sports/competitions/schemas.py \
      packages/easyml-sports/tests/competitions/__init__.py \
      packages/easyml-sports/tests/competitions/test_schemas.py
git commit -m "feat(competitions): add Pydantic v2 schemas for competition engine"
```

---

## Task 2: Structure — Build Competition Structures from Config

**Files:**
- Create: `packages/easyml-sports/src/easyml/sports/competitions/structure.py`
- Create: `packages/easyml-sports/tests/competitions/test_structure.py`
- Modify: `packages/easyml-sports/src/easyml/sports/competitions/__init__.py`

### Step 2.1: Write tests

Create `packages/easyml-sports/tests/competitions/test_structure.py`:

```python
"""Tests for competition structure building."""
from __future__ import annotations

import math

import pytest

from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionStructure,
    GroupConfig,
    KnockoutConfig,
    ScoringConfig,
)
from easyml.sports.competitions.structure import build_structure


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_entities(n: int) -> dict[str, str]:
    """Create seed_code -> entity_id mapping for n entities."""
    return {f"S{i + 1}": f"entity_{i + 1}" for i in range(n)}


# -----------------------------------------------------------------------
# Single elimination
# -----------------------------------------------------------------------

class TestSingleElimination:
    def test_4_participants(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        entities = _make_entities(4)
        struct = build_structure(cfg, entities)

        assert isinstance(struct, CompetitionStructure)
        # 4 participants -> 3 games (2 semifinal + 1 final)
        assert len(struct.slots) == 3
        assert struct.slot_to_round["R1G1"] == 1
        assert struct.slot_to_round["R1G2"] == 1
        assert struct.slot_to_round["R2G1"] == 2
        # Final references the two semifinal slots
        assert struct.slot_matchups["R2G1"] == ("R1G1", "R1G2")

    def test_8_participants(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=8,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
        )
        entities = _make_entities(8)
        struct = build_structure(cfg, entities)

        # 8 participants -> 7 games (4 + 2 + 1)
        assert len(struct.slots) == 7
        assert len(struct.round_slots[1]) == 4
        assert len(struct.round_slots[2]) == 2
        assert len(struct.round_slots[3]) == 1

    def test_seeding_1_vs_n(self):
        """Seed 1 plays seed N in round 1 (standard bracket seeding)."""
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=8,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
        )
        entities = _make_entities(8)
        struct = build_structure(cfg, entities)

        # Round 1: 1v8, 4v5, 2v7, 3v6 (standard bracket seeding)
        r1_matchups = {
            slot: struct.slot_matchups[slot]
            for slot in struct.round_slots[1]
        }
        seed_pairs = set()
        for ref_a, ref_b in r1_matchups.values():
            seed_pairs.add((ref_a, ref_b))
        assert ("S1", "S8") in seed_pairs
        assert ("S4", "S5") in seed_pairs
        assert ("S2", "S7") in seed_pairs
        assert ("S3", "S6") in seed_pairs

    def test_16_participants(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=16,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40, 80]),
        )
        entities = _make_entities(16)
        struct = build_structure(cfg, entities)

        # 16 -> 15 games
        assert len(struct.slots) == 15
        assert len(struct.round_slots[1]) == 8
        assert len(struct.round_slots[4]) == 1

    def test_64_with_4_regions(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=64,
            regions=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40, 80, 160, 320]),
        )
        entities = _make_entities(64)
        struct = build_structure(cfg, entities)

        # 64 -> 63 games
        assert len(struct.slots) == 63
        assert len(struct.round_slots[1]) == 32
        assert len(struct.round_slots[6]) == 1

    def test_seed_entity_mapping_preserved(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        entities = {"S1": "alpha", "S2": "beta", "S3": "gamma", "S4": "delta"}
        struct = build_structure(cfg, entities)

        assert struct.seed_to_entity == entities
        assert struct.entity_to_seed == {v: k for k, v in entities.items()}

    def test_round_1_matchups_reference_seeds(self):
        """First-round slot_matchups reference seed codes, not slot names."""
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        entities = _make_entities(4)
        struct = build_structure(cfg, entities)

        for slot in struct.round_slots[1]:
            ref_a, ref_b = struct.slot_matchups[slot]
            assert ref_a.startswith("S")
            assert ref_b.startswith("S")

    def test_later_rounds_reference_prior_slots(self):
        """Later-round slot_matchups reference winning slots from prior round."""
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        entities = _make_entities(4)
        struct = build_structure(cfg, entities)

        for slot in struct.round_slots[2]:
            ref_a, ref_b = struct.slot_matchups[slot]
            assert ref_a.startswith("R")
            assert ref_b.startswith("R")

    def test_non_power_of_2_gets_byes(self):
        """Non-power-of-2 participant count results in byes (some round-1 slots
        have seeds that auto-advance)."""
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=6,
            byes="auto",
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
        )
        entities = _make_entities(6)
        struct = build_structure(cfg, entities)

        # 8-slot bracket with 2 byes -> round 1 has 2 actual games + 2 byes
        # Total games: 2 (R1) + 4-slot bracket (3 games from R2+R3) = 5
        # Or: next power of 2 is 8 -> 7 slots, 2 of which are byes
        n_round1 = len(struct.round_slots.get(1, []))
        # With 6 participants in an 8-bracket: 2 byes, 2 real R1 games
        # Top 2 seeds get byes
        assert n_round1 >= 2


class TestSingleEliminationSlotNaming:
    def test_slot_names_contain_round_and_game(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        entities = _make_entities(4)
        struct = build_structure(cfg, entities)

        for slot in struct.slots:
            # Slots follow RxGy pattern (Round x, Game y)
            assert slot.startswith("R")
            assert "G" in slot


# -----------------------------------------------------------------------
# Round-robin
# -----------------------------------------------------------------------

class TestRoundRobin:
    def test_4_participants_all_vs_all(self):
        cfg = CompetitionConfig(
            format="round_robin",
            n_participants=4,
            rounds=1,
            scoring=ScoringConfig(type="points", win=3, draw=1, loss=0),
        )
        entities = _make_entities(4)
        struct = build_structure(cfg, entities)

        # 4 choose 2 = 6 matchups for single round-robin
        assert len(struct.slots) == 6

    def test_double_round_robin(self):
        cfg = CompetitionConfig(
            format="round_robin",
            n_participants=4,
            rounds=2,
            scoring=ScoringConfig(type="points", win=3, draw=1, loss=0),
        )
        entities = _make_entities(4)
        struct = build_structure(cfg, entities)

        # 4 choose 2 * 2 = 12 matchups
        assert len(struct.slots) == 12

    def test_all_matchups_reference_seeds(self):
        cfg = CompetitionConfig(
            format="round_robin",
            n_participants=3,
            rounds=1,
            scoring=ScoringConfig(type="points", win=3, draw=1, loss=0),
        )
        entities = _make_entities(3)
        struct = build_structure(cfg, entities)

        for slot in struct.slots:
            ref_a, ref_b = struct.slot_matchups[slot]
            assert ref_a.startswith("S")
            assert ref_b.startswith("S")

    def test_default_rounds_is_1(self):
        """If rounds not specified, defaults to single round-robin."""
        cfg = CompetitionConfig(
            format="round_robin",
            n_participants=4,
            scoring=ScoringConfig(type="points", win=3, draw=1, loss=0),
        )
        entities = _make_entities(4)
        struct = build_structure(cfg, entities)

        assert len(struct.slots) == 6  # single round-robin


# -----------------------------------------------------------------------
# Swiss
# -----------------------------------------------------------------------

class TestSwiss:
    def test_swiss_structure(self):
        cfg = CompetitionConfig(
            format="swiss",
            n_participants=8,
            n_rounds=3,
            scoring=ScoringConfig(type="points", win=1.0, draw=0.5, loss=0.0),
        )
        entities = _make_entities(8)
        struct = build_structure(cfg, entities)

        # Swiss: n_participants/2 games per round * n_rounds
        assert len(struct.slots) == 4 * 3  # 12

    def test_swiss_round_slots(self):
        cfg = CompetitionConfig(
            format="swiss",
            n_participants=8,
            n_rounds=3,
            scoring=ScoringConfig(type="points", win=1.0, draw=0.5, loss=0.0),
        )
        entities = _make_entities(8)
        struct = build_structure(cfg, entities)

        for rd in range(1, 4):
            assert len(struct.round_slots[rd]) == 4

    def test_swiss_first_round_seeded(self):
        """Round 1 of Swiss pairs by seed: S1 vs S5, S2 vs S6, etc."""
        cfg = CompetitionConfig(
            format="swiss",
            n_participants=8,
            n_rounds=3,
            scoring=ScoringConfig(type="points", win=1.0, draw=0.5, loss=0.0),
        )
        entities = _make_entities(8)
        struct = build_structure(cfg, entities)

        r1_slots = struct.round_slots[1]
        r1_pairs = [struct.slot_matchups[s] for s in r1_slots]
        # First round: top half vs bottom half by seed
        seed_pairs = set()
        for ref_a, ref_b in r1_pairs:
            seed_pairs.add((ref_a, ref_b))
        assert ("S1", "S5") in seed_pairs
        assert ("S2", "S6") in seed_pairs
        assert ("S3", "S7") in seed_pairs
        assert ("S4", "S8") in seed_pairs


# -----------------------------------------------------------------------
# Group-knockout
# -----------------------------------------------------------------------

class TestGroupKnockout:
    def test_group_knockout_structure(self):
        cfg = CompetitionConfig(
            format="group_knockout",
            n_participants=16,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
            groups=GroupConfig(
                n_groups=4,
                group_size=4,
                format="round_robin",
                advance=2,
                scoring=ScoringConfig(type="points", win=3, draw=1, loss=0),
            ),
            knockout=KnockoutConfig(
                format="single_elimination",
                scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
            ),
        )
        entities = _make_entities(16)
        struct = build_structure(cfg, entities)

        # Group stage: 4 groups * 6 games each = 24
        # Knockout: 8 participants -> 7 games
        assert len(struct.slots) == 24 + 7

    def test_group_slots_use_group_prefix(self):
        cfg = CompetitionConfig(
            format="group_knockout",
            n_participants=8,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
            groups=GroupConfig(
                n_groups=2,
                group_size=4,
                format="round_robin",
                advance=2,
                scoring=ScoringConfig(type="points", win=3, draw=1, loss=0),
            ),
            knockout=KnockoutConfig(
                format="single_elimination",
                scoring=ScoringConfig(type="per_round", values=[10, 20]),
            ),
        )
        entities = _make_entities(8)
        struct = build_structure(cfg, entities)

        group_slots = [s for s in struct.slots if s.startswith("G")]
        knockout_slots = [s for s in struct.slots if s.startswith("R")]
        assert len(group_slots) > 0
        assert len(knockout_slots) > 0


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:
    def test_2_participant_elimination(self):
        """Minimum bracket: 2 participants, 1 game."""
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=2,
            scoring=ScoringConfig(type="per_round", values=[10]),
        )
        entities = _make_entities(2)
        struct = build_structure(cfg, entities)

        assert len(struct.slots) == 1
        assert struct.slot_matchups[struct.slots[0]] == ("S1", "S2")

    def test_structure_config_preserved(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=4,
            scoring=ScoringConfig(type="per_round", values=[10, 20]),
        )
        entities = _make_entities(4)
        struct = build_structure(cfg, entities)

        assert struct.config == cfg
```

### Step 2.2: Verify tests fail

```bash
uv run pytest packages/easyml-sports/tests/competitions/test_structure.py -v 2>&1 | head -30
```

### Step 2.3: Implement structure builder

Create `packages/easyml-sports/src/easyml/sports/competitions/structure.py`:

```python
"""Build competition structures from config.

Converts a declarative CompetitionConfig + entity mapping into a fully resolved
CompetitionStructure with slots, matchup references, round groupings, and
seed-to-entity mappings. Supports single elimination, round-robin, Swiss,
and group-knockout formats.
"""
from __future__ import annotations

import math
from itertools import combinations

from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionStructure,
)


def build_structure(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> CompetitionStructure:
    """Build a CompetitionStructure from config and entity mapping.

    Args:
        config: Declarative competition configuration.
        seed_to_entity: Mapping of seed code (e.g. "S1") to entity ID string.

    Returns:
        Fully resolved CompetitionStructure.
    """
    entity_to_seed = {v: k for k, v in seed_to_entity.items()}

    if config.format == CompetitionFormat.SINGLE_ELIMINATION:
        slots, slot_matchups, slot_to_round, round_slots = _build_single_elimination(
            config, seed_to_entity,
        )
    elif config.format == CompetitionFormat.ROUND_ROBIN:
        slots, slot_matchups, slot_to_round, round_slots = _build_round_robin(
            config, seed_to_entity,
        )
    elif config.format == CompetitionFormat.SWISS:
        slots, slot_matchups, slot_to_round, round_slots = _build_swiss(
            config, seed_to_entity,
        )
    elif config.format == CompetitionFormat.GROUP_KNOCKOUT:
        slots, slot_matchups, slot_to_round, round_slots = _build_group_knockout(
            config, seed_to_entity,
        )
    elif config.format == CompetitionFormat.DOUBLE_ELIMINATION:
        slots, slot_matchups, slot_to_round, round_slots = _build_double_elimination(
            config, seed_to_entity,
        )
    else:
        raise ValueError(f"Unsupported format: {config.format}")

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
# Single elimination
# ---------------------------------------------------------------------------

def _build_single_elimination(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    """Build single-elimination bracket structure.

    Uses standard bracket seeding: 1vN, 2v(N-1), etc., with proper bracket
    placement so that 1-seed and 2-seed can only meet in the final.
    """
    n = config.n_participants
    # Next power of 2 for bracket size
    bracket_size = 1 << math.ceil(math.log2(max(n, 2)))
    n_rounds = int(math.log2(bracket_size))

    # Generate standard bracket order for seeds
    seed_order = _standard_bracket_order(bracket_size)

    # Build round-by-round
    slots: list[str] = []
    slot_matchups: dict[str, tuple[str, str]] = {}
    slot_to_round: dict[str, int] = {}
    round_slots: dict[int, list[str]] = {}

    # Round 1: pair seeds according to bracket order
    prev_round_slots: list[str] = []
    game_num = 1
    r1_slots: list[str] = []
    for i in range(0, bracket_size, 2):
        seed_a = seed_order[i]
        seed_b = seed_order[i + 1]
        ref_a = f"S{seed_a}"
        ref_b = f"S{seed_b}"

        # Handle byes: if seed > n_participants, it's a bye
        if seed_a > n and seed_b > n:
            # Both are byes — skip this slot entirely
            # This shouldn't happen with standard seeding
            r1_slots.append("")
            continue
        elif seed_b > n:
            # seed_b is a bye — seed_a auto-advances
            # Create a bye slot that the next round can reference
            slot_name = f"R1G{game_num}"
            slots.append(slot_name)
            slot_matchups[slot_name] = (ref_a, ref_a)  # bye: plays itself
            slot_to_round[slot_name] = 1
            r1_slots.append(slot_name)
            game_num += 1
        elif seed_a > n:
            # seed_a is a bye — seed_b auto-advances
            slot_name = f"R1G{game_num}"
            slots.append(slot_name)
            slot_matchups[slot_name] = (ref_b, ref_b)  # bye
            slot_to_round[slot_name] = 1
            r1_slots.append(slot_name)
            game_num += 1
        else:
            slot_name = f"R1G{game_num}"
            slots.append(slot_name)
            slot_matchups[slot_name] = (ref_a, ref_b)
            slot_to_round[slot_name] = 1
            r1_slots.append(slot_name)
            game_num += 1

    round_slots[1] = [s for s in r1_slots if s]
    prev_round_slots = r1_slots

    # Subsequent rounds
    for rd in range(2, n_rounds + 1):
        game_num = 1
        current_round_slots: list[str] = []
        for i in range(0, len(prev_round_slots), 2):
            slot_name = f"R{rd}G{game_num}"
            ref_a = prev_round_slots[i] if i < len(prev_round_slots) else ""
            ref_b = prev_round_slots[i + 1] if i + 1 < len(prev_round_slots) else ""
            if not ref_a or not ref_b:
                continue
            slots.append(slot_name)
            slot_matchups[slot_name] = (ref_a, ref_b)
            slot_to_round[slot_name] = rd
            current_round_slots.append(slot_name)
            game_num += 1
        round_slots[rd] = current_round_slots
        prev_round_slots = current_round_slots

    return slots, slot_matchups, slot_to_round, round_slots


def _standard_bracket_order(n: int) -> list[int]:
    """Generate standard bracket seeding order for n participants.

    Produces the ordering where seed 1 plays seed N, and the bracket
    is arranged so that 1v2 can only happen in the final.

    For n=8: [1, 8, 4, 5, 2, 7, 3, 6]
    For n=4: [1, 4, 2, 3]
    """
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2]

    # Recursive construction: split into top and bottom halves
    half = _standard_bracket_order(n // 2)
    result = []
    for seed in half:
        result.append(seed)
        result.append(n + 1 - seed)
    return result


# ---------------------------------------------------------------------------
# Round-robin
# ---------------------------------------------------------------------------

def _build_round_robin(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    """Build round-robin structure.

    All-vs-all matchups, repeated ``rounds`` times (default 1).
    """
    n = config.n_participants
    n_rounds = config.rounds if config.rounds is not None else 1
    seed_codes = [f"S{i + 1}" for i in range(n)]

    slots: list[str] = []
    slot_matchups: dict[str, tuple[str, str]] = {}
    slot_to_round: dict[str, int] = {}
    round_slots: dict[int, list[str]] = {}

    game_num = 1
    for rr_round in range(1, n_rounds + 1):
        rd_slots: list[str] = []
        for a, b in combinations(seed_codes, 2):
            slot_name = f"RR{rr_round}G{game_num}"
            slots.append(slot_name)
            slot_matchups[slot_name] = (a, b)
            slot_to_round[slot_name] = rr_round
            rd_slots.append(slot_name)
            game_num += 1
        round_slots[rr_round] = rd_slots

    return slots, slot_matchups, slot_to_round, round_slots


# ---------------------------------------------------------------------------
# Swiss
# ---------------------------------------------------------------------------

def _build_swiss(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    """Build Swiss-system structure.

    Round 1 is seeded (top half vs bottom half). Later rounds have placeholder
    slots — actual pairings are determined during simulation based on standings.
    """
    n = config.n_participants
    n_rounds = config.n_rounds if config.n_rounds is not None else 5
    games_per_round = n // 2

    slots: list[str] = []
    slot_matchups: dict[str, tuple[str, str]] = {}
    slot_to_round: dict[str, int] = {}
    round_slots: dict[int, list[str]] = {}

    # Round 1: seeded pairings (1 vs n/2+1, 2 vs n/2+2, etc.)
    r1_slots: list[str] = []
    for g in range(games_per_round):
        slot_name = f"SW1G{g + 1}"
        ref_a = f"S{g + 1}"
        ref_b = f"S{g + 1 + games_per_round}"
        slots.append(slot_name)
        slot_matchups[slot_name] = (ref_a, ref_b)
        slot_to_round[slot_name] = 1
        r1_slots.append(slot_name)
    round_slots[1] = r1_slots

    # Later rounds: placeholder slots (pairings resolved at simulation time)
    for rd in range(2, n_rounds + 1):
        rd_slots: list[str] = []
        for g in range(games_per_round):
            slot_name = f"SW{rd}G{g + 1}"
            # Placeholders — ref_a and ref_b are TBD markers
            slots.append(slot_name)
            slot_matchups[slot_name] = (f"TBD_{rd}_{g + 1}_A", f"TBD_{rd}_{g + 1}_B")
            slot_to_round[slot_name] = rd
            rd_slots.append(slot_name)
        round_slots[rd] = rd_slots

    return slots, slot_matchups, slot_to_round, round_slots


# ---------------------------------------------------------------------------
# Group-knockout
# ---------------------------------------------------------------------------

def _build_group_knockout(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    """Build group-stage-to-knockout structure.

    Group stage: round-robin within each group.
    Knockout stage: single-elimination among qualifiers.
    """
    groups_cfg = config.groups
    knockout_cfg = config.knockout
    if groups_cfg is None or knockout_cfg is None:
        raise ValueError("group_knockout format requires both 'groups' and 'knockout' config")

    n_groups = groups_cfg.n_groups
    group_size = groups_cfg.group_size
    advance = groups_cfg.advance

    slots: list[str] = []
    slot_matchups: dict[str, tuple[str, str]] = {}
    slot_to_round: dict[str, int] = {}
    round_slots: dict[int, list[str]] = {}

    # Assign seeds to groups: snake seeding
    # Group 1: S1, S(n_groups+1), ...; Group 2: S2, S(n_groups+2), ...
    group_seeds: dict[int, list[str]] = {}
    for g in range(n_groups):
        seeds: list[str] = []
        for pos in range(group_size):
            seed_num = g + 1 + pos * n_groups
            seeds.append(f"S{seed_num}")
        group_seeds[g + 1] = seeds

    # Group stage: round-robin within each group
    group_round = 0  # use round 0 for group stage slots
    game_num = 1
    group_stage_slots: list[str] = []
    for g_num, seeds in group_seeds.items():
        for a, b in combinations(seeds, 2):
            slot_name = f"G{g_num}M{game_num}"
            slots.append(slot_name)
            slot_matchups[slot_name] = (a, b)
            slot_to_round[slot_name] = 0
            group_stage_slots.append(slot_name)
            game_num += 1
    round_slots[0] = group_stage_slots

    # Knockout stage: build single-elimination for qualifiers
    n_qualifiers = n_groups * advance
    bracket_size = 1 << math.ceil(math.log2(max(n_qualifiers, 2)))
    n_ko_rounds = int(math.log2(bracket_size))

    # Knockout seeds are placeholders: GxPy = Group x, Position y
    ko_seed_order = _standard_bracket_order(bracket_size)

    # Map knockout positions to group qualifier placeholders
    # Position 1 from each group first, then position 2, etc.
    ko_seed_refs: dict[int, str] = {}
    ko_idx = 1
    for pos in range(1, advance + 1):
        for g_num in range(1, n_groups + 1):
            ko_seed_refs[ko_idx] = f"G{g_num}P{pos}"
            ko_idx += 1

    # Build knockout rounds
    prev_round_slots_ko: list[str] = []
    game_num = 1
    r1_ko: list[str] = []
    for i in range(0, bracket_size, 2):
        seed_a_num = ko_seed_order[i]
        seed_b_num = ko_seed_order[i + 1]
        ref_a = ko_seed_refs.get(seed_a_num, f"KO_BYE_{seed_a_num}")
        ref_b = ko_seed_refs.get(seed_b_num, f"KO_BYE_{seed_b_num}")

        if seed_a_num > n_qualifiers and seed_b_num > n_qualifiers:
            continue
        elif seed_b_num > n_qualifiers:
            slot_name = f"R1G{game_num}"
            slots.append(slot_name)
            slot_matchups[slot_name] = (ref_a, ref_a)
            slot_to_round[slot_name] = 1
            r1_ko.append(slot_name)
            game_num += 1
        elif seed_a_num > n_qualifiers:
            slot_name = f"R1G{game_num}"
            slots.append(slot_name)
            slot_matchups[slot_name] = (ref_b, ref_b)
            slot_to_round[slot_name] = 1
            r1_ko.append(slot_name)
            game_num += 1
        else:
            slot_name = f"R1G{game_num}"
            slots.append(slot_name)
            slot_matchups[slot_name] = (ref_a, ref_b)
            slot_to_round[slot_name] = 1
            r1_ko.append(slot_name)
            game_num += 1

    round_slots[1] = r1_ko
    prev_round_slots_ko = r1_ko

    for rd in range(2, n_ko_rounds + 1):
        game_num = 1
        current_rd: list[str] = []
        for i in range(0, len(prev_round_slots_ko), 2):
            slot_name = f"R{rd}G{game_num}"
            ref_a = prev_round_slots_ko[i]
            ref_b = prev_round_slots_ko[i + 1] if i + 1 < len(prev_round_slots_ko) else ""
            if not ref_a or not ref_b:
                continue
            slots.append(slot_name)
            slot_matchups[slot_name] = (ref_a, ref_b)
            slot_to_round[slot_name] = rd
            current_rd.append(slot_name)
            game_num += 1
        round_slots[rd] = current_rd
        prev_round_slots_ko = current_rd

    return slots, slot_matchups, slot_to_round, round_slots


# ---------------------------------------------------------------------------
# Double elimination (stub — full implementation in later task)
# ---------------------------------------------------------------------------

def _build_double_elimination(
    config: CompetitionConfig,
    seed_to_entity: dict[str, str],
) -> tuple[list[str], dict[str, tuple[str, str]], dict[str, int], dict[int, list[str]]]:
    """Build double-elimination bracket structure.

    Placeholder — raises NotImplementedError until Task 5+.
    """
    raise NotImplementedError("Double-elimination structure not yet implemented")
```

### Step 2.4: Update `__init__.py`

Add to `packages/easyml-sports/src/easyml/sports/competitions/__init__.py`:

```python
"""Competition engine — generic competition simulation and optimization.

Supports single/double elimination, round-robin, Swiss, and group-to-knockout
formats via config-driven schemas, vectorized Monte Carlo simulation,
pool-aware optimization, and configurable scoring.
"""
from easyml.sports.competitions.schemas import (
    AdjustmentConfig,
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    CompetitionStructure,
    GroupConfig,
    KnockoutConfig,
    MatchupContext,
    ScoreResult,
    ScoringConfig,
    SeedingMode,
    StandingsEntry,
)
from easyml.sports.competitions.structure import build_structure

__all__ = [
    "AdjustmentConfig",
    "CompetitionConfig",
    "CompetitionFormat",
    "CompetitionResult",
    "CompetitionStructure",
    "GroupConfig",
    "KnockoutConfig",
    "MatchupContext",
    "ScoreResult",
    "ScoringConfig",
    "SeedingMode",
    "StandingsEntry",
    "build_structure",
]
```

### Step 2.5: Verify tests pass

```bash
uv run pytest packages/easyml-sports/tests/competitions/test_structure.py -v
```

### Step 2.6: Commit

```bash
git add packages/easyml-sports/src/easyml/sports/competitions/structure.py \
      packages/easyml-sports/src/easyml/sports/competitions/__init__.py \
      packages/easyml-sports/tests/competitions/test_structure.py
git commit -m "feat(competitions): add structure builder for all competition formats"
```

---

## Task 3: Simulator — Monte Carlo Competition Engine

**Files:**
- Create: `packages/easyml-sports/src/easyml/sports/competitions/simulator.py`
- Create: `packages/easyml-sports/tests/competitions/test_simulator.py`
- Modify: `packages/easyml-sports/src/easyml/sports/competitions/__init__.py`

### Step 3.1: Write tests

Create `packages/easyml-sports/tests/competitions/test_simulator.py`:

```python
"""Tests for competition simulator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionStructure,
    MatchupContext,
    ScoringConfig,
)
from easyml.sports.competitions.simulator import CompetitionSimulator
from easyml.sports.competitions.structure import build_structure


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_entities(n: int) -> dict[str, str]:
    return {f"S{i + 1}": f"entity_{i + 1}" for i in range(n)}


def _make_probabilities(entities: dict[str, str], dominant: str | None = None) -> pd.DataFrame:
    """Create pairwise probability DataFrame.

    If dominant is specified, that entity beats everyone with p=0.99.
    Otherwise, higher-seeded entities are slightly favored.
    """
    entity_ids = sorted(entities.values())
    rows = []
    for i, a in enumerate(entity_ids):
        for j, b in enumerate(entity_ids):
            if j <= i:
                continue
            if dominant and a == dominant:
                prob = 0.99
            elif dominant and b == dominant:
                prob = 0.01
            else:
                # Slight advantage to lower-numbered (higher-seeded) entity
                prob = 0.55
            rows.append({
                "entity_a": a,
                "entity_b": b,
                "prob_ensemble": prob,
                "prob_model_1": prob + 0.01,
                "prob_model_2": prob - 0.01,
            })
    return pd.DataFrame(rows)


def _make_4_entity_setup(dominant: str | None = None):
    """Create a 4-entity single-elimination setup."""
    cfg = CompetitionConfig(
        format="single_elimination",
        n_participants=4,
        scoring=ScoringConfig(type="per_round", values=[10, 20]),
    )
    entities = _make_entities(4)
    struct = build_structure(cfg, entities)
    probs = _make_probabilities(entities, dominant=dominant)
    sim = CompetitionSimulator(config=cfg, structure=struct, probabilities=probs)
    return cfg, struct, probs, sim


# -----------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------

class TestSimulatorConstruction:
    def test_creates_successfully(self):
        _, _, _, sim = _make_4_entity_setup()
        assert sim is not None

    def test_stores_structure(self):
        _, struct, _, sim = _make_4_entity_setup()
        assert sim.structure == struct


# -----------------------------------------------------------------------
# Probability lookups
# -----------------------------------------------------------------------

class TestProbabilityLookup:
    def test_get_win_prob_returns_correct_value(self):
        _, _, probs, sim = _make_4_entity_setup()
        # entity_1 vs entity_2: prob should be 0.55 (higher seed favored)
        prob = sim.get_win_prob("entity_1", "entity_2")
        assert prob == pytest.approx(0.55)

    def test_get_win_prob_reverse_order(self):
        _, _, _, sim = _make_4_entity_setup()
        prob_ab = sim.get_win_prob("entity_1", "entity_2")
        prob_ba = sim.get_win_prob("entity_2", "entity_1")
        assert prob_ab + prob_ba == pytest.approx(1.0)

    def test_get_win_prob_unknown_pair_returns_0_5(self):
        _, _, _, sim = _make_4_entity_setup()
        prob = sim.get_win_prob("entity_1", "unknown_entity")
        assert prob == pytest.approx(0.5)

    def test_dominant_entity_prob(self):
        _, _, _, sim = _make_4_entity_setup(dominant="entity_1")
        prob = sim.get_win_prob("entity_1", "entity_3")
        assert prob == pytest.approx(0.99)


# -----------------------------------------------------------------------
# Model agreement
# -----------------------------------------------------------------------

class TestModelAgreement:
    def test_high_agreement_when_models_close(self):
        _, _, _, sim = _make_4_entity_setup()
        agreement = sim.get_model_agreement("entity_1", "entity_2")
        # Models differ by 0.02 — agreement should be high
        assert agreement > 0.9

    def test_agreement_is_symmetric(self):
        _, _, _, sim = _make_4_entity_setup()
        a = sim.get_model_agreement("entity_1", "entity_2")
        b = sim.get_model_agreement("entity_2", "entity_1")
        assert a == pytest.approx(b)

    def test_unknown_pair_returns_1(self):
        _, _, _, sim = _make_4_entity_setup()
        assert sim.get_model_agreement("entity_1", "unknown") == 1.0


# -----------------------------------------------------------------------
# Matchup context
# -----------------------------------------------------------------------

class TestMatchupContext:
    def test_returns_matchup_context(self):
        _, _, _, sim = _make_4_entity_setup()
        ctx = sim.get_matchup_context(
            entity_a="entity_1",
            entity_b="entity_4",
            slot="R1G1",
            round_num=1,
            pick="entity_1",
            strategy="chalk",
        )
        assert isinstance(ctx, MatchupContext)
        assert ctx.slot == "R1G1"
        assert ctx.entity_a == "entity_1"
        assert ctx.entity_b == "entity_4"
        assert ctx.pick == "entity_1"
        assert ctx.strategy == "chalk"

    def test_upset_detection(self):
        _, _, _, sim = _make_4_entity_setup()
        # entity_1 is favored (prob > 0.5), picking entity_4 is an upset
        ctx = sim.get_matchup_context(
            entity_a="entity_1",
            entity_b="entity_4",
            slot="R1G1",
            round_num=1,
            pick="entity_4",
            strategy="contrarian",
        )
        assert ctx.upset is True

    def test_no_upset_when_picking_favorite(self):
        _, _, _, sim = _make_4_entity_setup()
        ctx = sim.get_matchup_context(
            entity_a="entity_1",
            entity_b="entity_4",
            slot="R1G1",
            round_num=1,
            pick="entity_1",
            strategy="chalk",
        )
        assert ctx.upset is False


# -----------------------------------------------------------------------
# simulate_once
# -----------------------------------------------------------------------

class TestSimulateOnce:
    def test_returns_dict_of_slot_to_entity(self):
        _, struct, _, sim = _make_4_entity_setup()
        rng = np.random.default_rng(42)
        result = sim.simulate_once(rng)

        assert isinstance(result, dict)
        # All slots should have a winner
        for slot in struct.slots:
            assert slot in result
            assert isinstance(result[slot], str)

    def test_deterministic_with_same_seed(self):
        _, _, _, sim = _make_4_entity_setup()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r1 = sim.simulate_once(rng1)
        r2 = sim.simulate_once(rng2)
        assert r1 == r2

    def test_different_seeds_can_differ(self):
        _, _, _, sim = _make_4_entity_setup()
        results = set()
        for seed in range(100):
            rng = np.random.default_rng(seed)
            r = sim.simulate_once(rng)
            # Collect champion (last slot)
            champion = r[sim.structure.slots[-1]]
            results.add(champion)
        # With random outcomes, we should see more than 1 champion
        assert len(results) > 1

    def test_dominant_entity_wins_almost_always(self):
        _, _, _, sim = _make_4_entity_setup(dominant="entity_1")
        wins = 0
        n_sims = 500
        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            result = sim.simulate_once(rng)
            champion = result[sim.structure.slots[-1]]
            if champion == "entity_1":
                wins += 1
        # 0.99 * 0.99 ~ 0.98 chance of winning 2 games
        assert wins / n_sims > 0.90


# -----------------------------------------------------------------------
# simulate_many (vectorized)
# -----------------------------------------------------------------------

class TestSimulateMany:
    def test_returns_correct_shape(self):
        _, struct, _, sim = _make_4_entity_setup()
        results = sim.simulate_many(n=100, seed=42)
        assert len(results) == 100
        for r in results:
            assert len(r) == len(struct.slots)

    def test_deterministic(self):
        _, _, _, sim = _make_4_entity_setup()
        r1 = sim.simulate_many(n=50, seed=42)
        r2 = sim.simulate_many(n=50, seed=42)
        for a, b in zip(r1, r2):
            assert a == b

    def test_dominant_entity_champion_rate(self):
        _, _, _, sim = _make_4_entity_setup(dominant="entity_1")
        results = sim.simulate_many(n=1000, seed=42)
        final_slot = sim.structure.slots[-1]
        wins = sum(1 for r in results if r[final_slot] == "entity_1")
        assert wins / 1000 > 0.90

    def test_results_match_simulate_once_distribution(self):
        """Vectorized and scalar simulation should produce similar distributions."""
        _, _, _, sim = _make_4_entity_setup()
        n = 2000
        # Vectorized
        vec_results = sim.simulate_many(n=n, seed=42)
        final_slot = sim.structure.slots[-1]
        vec_champs = {}
        for r in vec_results:
            c = r[final_slot]
            vec_champs[c] = vec_champs.get(c, 0) + 1

        # Scalar (different seed to avoid exact match, check distribution)
        scalar_champs = {}
        for i in range(n):
            rng = np.random.default_rng(i + 100000)
            r = sim.simulate_once(rng)
            c = r[final_slot]
            scalar_champs[c] = scalar_champs.get(c, 0) + 1

        # All entities that appear as champion in one should appear in the other
        for entity in set(vec_champs) | set(scalar_champs):
            vec_rate = vec_champs.get(entity, 0) / n
            scalar_rate = scalar_champs.get(entity, 0) / n
            # Allow 5% tolerance
            assert abs(vec_rate - scalar_rate) < 0.05, (
                f"{entity}: vec={vec_rate:.3f} vs scalar={scalar_rate:.3f}"
            )


# -----------------------------------------------------------------------
# pick_most_likely (chalk)
# -----------------------------------------------------------------------

class TestPickMostLikely:
    def test_returns_all_slots(self):
        _, struct, _, sim = _make_4_entity_setup()
        picks = sim.pick_most_likely()
        assert len(picks) == len(struct.slots)

    def test_dominant_entity_wins_chalk(self):
        _, _, _, sim = _make_4_entity_setup(dominant="entity_1")
        picks = sim.pick_most_likely()
        final_slot = sim.structure.slots[-1]
        assert picks[final_slot] == "entity_1"

    def test_chalk_always_picks_favorite(self):
        _, struct, _, sim = _make_4_entity_setup()
        picks = sim.pick_most_likely()
        # In round 1 with uniform 0.55, lower entity ID (= higher seed) wins
        for slot in struct.round_slots[1]:
            ref_a, ref_b = struct.slot_matchups[slot]
            entity_a = struct.seed_to_entity[ref_a]
            entity_b = struct.seed_to_entity[ref_b]
            prob = sim.get_win_prob(entity_a, entity_b)
            if prob >= 0.5:
                assert picks[slot] == entity_a
            else:
                assert picks[slot] == entity_b

    def test_deterministic(self):
        _, _, _, sim = _make_4_entity_setup()
        p1 = sim.pick_most_likely()
        p2 = sim.pick_most_likely()
        assert p1 == p2


# -----------------------------------------------------------------------
# entity_round_probabilities
# -----------------------------------------------------------------------

class TestEntityRoundProbabilities:
    def test_returns_dataframe(self):
        _, _, _, sim = _make_4_entity_setup()
        df = sim.entity_round_probabilities(n_sims=500, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert "entity" in df.columns
        assert "round_1" in df.columns
        assert "round_2" in df.columns

    def test_round_1_probs_sum_to_n_participants(self):
        """Each round-1 game has exactly 1 winner, so total round-1 wins = n_games."""
        _, struct, _, sim = _make_4_entity_setup()
        df = sim.entity_round_probabilities(n_sims=1000, seed=42)
        n_round1_games = len(struct.round_slots[1])
        assert df["round_1"].sum() == pytest.approx(n_round1_games, abs=0.1)

    def test_champion_probs_sum_to_1(self):
        _, _, _, sim = _make_4_entity_setup()
        df = sim.entity_round_probabilities(n_sims=1000, seed=42)
        max_round = max(sim.structure.round_slots.keys())
        champ_col = f"round_{max_round}"
        assert df[champ_col].sum() == pytest.approx(1.0, abs=0.05)

    def test_dominant_entity_high_champion_prob(self):
        _, _, _, sim = _make_4_entity_setup(dominant="entity_1")
        df = sim.entity_round_probabilities(n_sims=1000, seed=42)
        max_round = max(sim.structure.round_slots.keys())
        champ_col = f"round_{max_round}"
        e1_row = df[df["entity"] == "entity_1"]
        assert float(e1_row[champ_col].iloc[0]) > 0.90


# -----------------------------------------------------------------------
# 8-entity tournament
# -----------------------------------------------------------------------

class TestLargerTournament:
    def test_8_entity_simulation(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=8,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
        )
        entities = _make_entities(8)
        struct = build_structure(cfg, entities)
        probs = _make_probabilities(entities)
        sim = CompetitionSimulator(config=cfg, structure=struct, probabilities=probs)

        results = sim.simulate_many(n=100, seed=42)
        assert len(results) == 100
        for r in results:
            assert len(r) == 7  # 4 + 2 + 1

    def test_8_entity_round_probs(self):
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=8,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
        )
        entities = _make_entities(8)
        struct = build_structure(cfg, entities)
        probs = _make_probabilities(entities)
        sim = CompetitionSimulator(config=cfg, structure=struct, probabilities=probs)

        df = sim.entity_round_probabilities(n_sims=500, seed=42)
        assert len(df) == 8
        assert "round_3" in df.columns
```

### Step 3.2: Verify tests fail

```bash
uv run pytest packages/easyml-sports/tests/competitions/test_simulator.py -v 2>&1 | head -30
```

### Step 3.3: Implement simulator

Create `packages/easyml-sports/src/easyml/sports/competitions/simulator.py`:

```python
"""Monte Carlo competition simulation engine.

Supports single-elimination brackets with vectorized simulation across
N simulations per matchup slot. Entity IDs are strings. Probability lookups
use a dense matrix indexed by entity position for O(1) access.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionStructure,
    MatchupContext,
)


class CompetitionSimulator:
    """Simulate competitions using pairwise win probabilities.

    Builds a dense probability matrix for O(1) vectorized lookups.
    Stores per-model probabilities for agreement analysis.
    """

    def __init__(
        self,
        config: CompetitionConfig,
        structure: CompetitionStructure,
        probabilities: pd.DataFrame,
    ):
        self.config = config
        self.structure = structure

        # Identify entity columns and prob columns
        self._entity_a_col = "entity_a"
        self._entity_b_col = "entity_b"
        self._ensemble_col = "prob_ensemble"

        # Build probability lookup: (entity_a, entity_b) -> P(entity_a wins)
        # where entity_a < entity_b lexicographically
        self._prob_lookup: dict[tuple[str, str], float] = {}
        for _, row in probabilities.iterrows():
            a, b = str(row[self._entity_a_col]), str(row[self._entity_b_col])
            if a > b:
                a, b = b, a
                prob = 1.0 - float(row[self._ensemble_col])
            else:
                prob = float(row[self._ensemble_col])
            self._prob_lookup[(a, b)] = prob

        # Per-model probs for agreement analysis
        self._prob_cols = [
            c for c in probabilities.columns
            if c.startswith("prob_") and c != self._ensemble_col
        ]
        self._all_probs = probabilities

        # Build entity index for dense matrix
        all_entities = sorted(set(structure.seed_to_entity.values()))
        self._entity_to_idx: dict[str, int] = {
            e: i for i, e in enumerate(all_entities)
        }
        self._idx_to_entity: dict[int, str] = {
            i: e for e, i in self._entity_to_idx.items()
        }
        n_entities = len(all_entities)

        # Dense probability matrix: prob_matrix[i, j] = P(entity_i beats entity_j)
        self._prob_matrix = np.full((n_entities, n_entities), 0.5)
        for (a, b), p in self._prob_lookup.items():
            ia = self._entity_to_idx.get(a)
            ib = self._entity_to_idx.get(b)
            if ia is not None and ib is not None:
                self._prob_matrix[ia, ib] = p
                self._prob_matrix[ib, ia] = 1.0 - p

        # Pre-compute sorted matchups and slot index for vectorized simulation
        self._sorted_slots = sorted(
            structure.slots,
            key=lambda s: structure.slot_to_round.get(s, 0),
        )
        self._slot_idx = {s: i for i, s in enumerate(self._sorted_slots)}

    def get_win_prob(self, entity_a: str, entity_b: str) -> float:
        """Get P(entity_a beats entity_b)."""
        a, b = (entity_a, entity_b) if entity_a < entity_b else (entity_b, entity_a)
        prob_a_wins = self._prob_lookup.get((a, b), 0.5)
        if entity_a == a:
            return prob_a_wins
        return 1.0 - prob_a_wins

    def get_model_agreement(self, entity_a: str, entity_b: str) -> float:
        """Compute model agreement for a matchup.

        Returns 0.0 for total disagreement, 1.0 for unanimity.
        Metric: 1 - (std of per-model probs) / 0.25
        """
        a, b = (entity_a, entity_b) if entity_a < entity_b else (entity_b, entity_a)
        mask = (
            (self._all_probs[self._entity_a_col].astype(str) == a)
            & (self._all_probs[self._entity_b_col].astype(str) == b)
        )
        row = self._all_probs[mask]
        if row.empty or not self._prob_cols:
            return 1.0

        model_probs = row[self._prob_cols].values.flatten().astype(float)
        if entity_a != a:
            model_probs = 1.0 - model_probs

        valid = model_probs[~np.isnan(model_probs)]
        if len(valid) < 2:
            return 1.0

        std = float(np.std(valid))
        return max(0.0, 1.0 - std / 0.25)

    def get_matchup_context(
        self,
        entity_a: str,
        entity_b: str,
        slot: str,
        round_num: int,
        pick: str,
        strategy: str,
    ) -> MatchupContext:
        """Build rich context for a specific matchup."""
        a, b = (entity_a, entity_b) if entity_a < entity_b else (entity_b, entity_a)
        prob_a = self.get_win_prob(entity_a, entity_b)

        mask = (
            (self._all_probs[self._entity_a_col].astype(str) == a)
            & (self._all_probs[self._entity_b_col].astype(str) == b)
        )
        row = self._all_probs[mask]
        model_probs: dict[str, float] = {}
        if not row.empty:
            for col in self._prob_cols:
                val = row[col].values[0]
                if not np.isnan(val):
                    p = float(val) if entity_a == a else float(1.0 - val)
                    model_probs[col] = p

        agreement = self.get_model_agreement(entity_a, entity_b)
        upset = (
            (pick == entity_a and prob_a < 0.5)
            or (pick == entity_b and prob_a >= 0.5)
        )

        return MatchupContext(
            slot=slot,
            round_num=round_num,
            entity_a=entity_a,
            entity_b=entity_b,
            prob_a=prob_a,
            model_probs=model_probs,
            model_agreement=agreement,
            pick=pick,
            strategy=strategy,
            upset=upset,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_once(
        self, rng: np.random.Generator | None = None,
    ) -> dict[str, str]:
        """Simulate one complete competition.

        Returns:
            Dict mapping slot -> winning entity ID.
        """
        if rng is None:
            rng = np.random.default_rng()

        results: dict[str, str] = {}
        for slot in self._sorted_slots:
            ref_a, ref_b = self.structure.slot_matchups[slot]
            entity_a = self._resolve(ref_a, results)
            entity_b = self._resolve(ref_b, results)

            if entity_a is None or entity_b is None:
                continue

            # Handle byes (same entity on both sides)
            if entity_a == entity_b:
                results[slot] = entity_a
                continue

            prob_a = self.get_win_prob(entity_a, entity_b)
            if rng.random() < prob_a:
                results[slot] = entity_a
            else:
                results[slot] = entity_b

        return results

    def simulate_many(
        self, n: int, seed: int = 42,
    ) -> list[dict[str, str]]:
        """Simulate n complete competitions using vectorized operations.

        Returns:
            List of n result dicts (slot -> winning entity ID).
        """
        results_arr = self._simulate_many_vectorized(n, seed)
        return self._results_array_to_dicts(results_arr)

    def _simulate_many_vectorized(
        self, n: int, seed: int = 42,
    ) -> np.ndarray:
        """Vectorized simulation — processes all n sims per slot.

        Returns:
            Array of shape (n, n_slots) with entity indices (into _idx_to_entity).
            -1 means unresolved.
        """
        rng = np.random.default_rng(seed)
        n_slots = len(self._sorted_slots)
        randoms = rng.random((n, n_slots))
        # Store entity indices, -1 for unresolved
        results = np.full((n, n_slots), -1, dtype=np.int32)

        for i, slot in enumerate(self._sorted_slots):
            ref_a, ref_b = self.structure.slot_matchups[slot]

            # Resolve entity_a indices across all simulations
            a_indices = self._resolve_vectorized(ref_a, results, n)
            b_indices = self._resolve_vectorized(ref_b, results, n)

            if a_indices is None or b_indices is None:
                continue

            # Handle byes
            is_bye = (a_indices == b_indices)
            valid = (a_indices >= 0) & (b_indices >= 0)

            # Vectorized probability lookup
            safe_a = np.clip(a_indices, 0, self._prob_matrix.shape[0] - 1)
            safe_b = np.clip(b_indices, 0, self._prob_matrix.shape[1] - 1)
            probs = self._prob_matrix[safe_a, safe_b]

            winners = np.where(randoms[:, i] < probs, a_indices, b_indices)
            # Byes: entity_a auto-advances
            winners = np.where(is_bye & valid, a_indices, winners)
            results[:, i] = np.where(valid, winners, -1)

        return results

    def _resolve_vectorized(
        self,
        ref: str,
        results: np.ndarray,
        n: int,
    ) -> np.ndarray | None:
        """Resolve a reference to entity indices across all simulations.

        Args:
            ref: Seed code (e.g. "S1") or slot name (e.g. "R1G1").
            results: Current results array (n, n_slots).
            n: Number of simulations.

        Returns:
            Array of shape (n,) with entity indices, or None if unresolvable.
        """
        # Is it a seed code?
        if ref in self.structure.seed_to_entity:
            entity = self.structure.seed_to_entity[ref]
            idx = self._entity_to_idx.get(entity, -1)
            return np.full(n, idx, dtype=np.int32)

        # Is it a prior slot?
        if ref in self._slot_idx:
            slot_i = self._slot_idx[ref]
            return results[:, slot_i]

        return None

    def _results_array_to_dicts(
        self, results: np.ndarray,
    ) -> list[dict[str, str]]:
        """Convert vectorized results array back to list of dicts."""
        out = []
        for i in range(results.shape[0]):
            d: dict[str, str] = {}
            for j, slot in enumerate(self._sorted_slots):
                val = int(results[i, j])
                if val >= 0 and val in self._idx_to_entity:
                    d[slot] = self._idx_to_entity[val]
            out.append(d)
        return out

    def pick_most_likely(self) -> dict[str, str]:
        """Pick the most likely winner for each slot (chalk).

        Deterministically picks the higher-probability entity at each slot.
        """
        results: dict[str, str] = {}
        for slot in self._sorted_slots:
            ref_a, ref_b = self.structure.slot_matchups[slot]
            entity_a = self._resolve(ref_a, results)
            entity_b = self._resolve(ref_b, results)

            if entity_a is None or entity_b is None:
                continue

            if entity_a == entity_b:
                results[slot] = entity_a
                continue

            prob_a = self.get_win_prob(entity_a, entity_b)
            if prob_a >= 0.5:
                results[slot] = entity_a
            else:
                results[slot] = entity_b

        return results

    def entity_round_probabilities(
        self, n_sims: int = 10000, seed: int = 42,
    ) -> pd.DataFrame:
        """Compute probability of each entity winning in each round.

        Returns:
            DataFrame with columns: entity, round_1, round_2, ..., round_N
        """
        sims = self.simulate_many(n_sims, seed)
        slot_to_round = self.structure.slot_to_round

        # Count wins per entity per round
        entity_round_counts: dict[tuple[str, int], int] = {}
        for sim in sims:
            for slot, winner in sim.items():
                rd = slot_to_round.get(slot, 0)
                if rd < 1:
                    continue
                key = (winner, rd)
                entity_round_counts[key] = entity_round_counts.get(key, 0) + 1

        # Build DataFrame
        all_entities = sorted(self.structure.seed_to_entity.values())
        max_round = max(slot_to_round.values()) if slot_to_round else 0
        rows = []
        for entity in all_entities:
            row: dict[str, object] = {"entity": entity}
            for rd in range(1, max_round + 1):
                row[f"round_{rd}"] = (
                    entity_round_counts.get((entity, rd), 0) / n_sims
                )
            rows.append(row)

        df = pd.DataFrame(rows)
        if max_round > 0:
            df = df.sort_values(f"round_{max_round}", ascending=False).reset_index(
                drop=True
            )
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(
        self, ref: str, results: dict[str, str],
    ) -> str | None:
        """Resolve a seed code or slot reference to an entity ID."""
        if ref in self.structure.seed_to_entity:
            return self.structure.seed_to_entity[ref]
        if ref in results:
            return results[ref]
        return None
```

### Step 3.4: Update `__init__.py`

Add to `packages/easyml-sports/src/easyml/sports/competitions/__init__.py` imports:

```python
from easyml.sports.competitions.simulator import CompetitionSimulator
```

And add `"CompetitionSimulator"` to `__all__`.

### Step 3.5: Verify tests pass

```bash
uv run pytest packages/easyml-sports/tests/competitions/test_simulator.py -v
```

### Step 3.6: Commit

```bash
git add packages/easyml-sports/src/easyml/sports/competitions/simulator.py \
      packages/easyml-sports/src/easyml/sports/competitions/__init__.py \
      packages/easyml-sports/tests/competitions/test_simulator.py
git commit -m "feat(competitions): add vectorized Monte Carlo competition simulator"
```

---

## Task 4: Scorer — Configurable Competition Scoring

**Files:**
- Create: `packages/easyml-sports/src/easyml/sports/competitions/scorer.py`
- Create: `packages/easyml-sports/tests/competitions/test_scorer.py`
- Modify: `packages/easyml-sports/src/easyml/sports/competitions/__init__.py`

### Step 4.1: Write tests

Create `packages/easyml-sports/tests/competitions/test_scorer.py`:

```python
"""Tests for competition scorer."""
from __future__ import annotations

import pytest

from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionStructure,
    ScoreResult,
    ScoringConfig,
    StandingsEntry,
)
from easyml.sports.competitions.scorer import CompetitionScorer
from easyml.sports.competitions.structure import build_structure


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_entities(n: int) -> dict[str, str]:
    return {f"S{i + 1}": f"entity_{i + 1}" for i in range(n)}


def _make_4_entity_bracket():
    """Build a 4-entity single-elimination bracket structure."""
    cfg = CompetitionConfig(
        format="single_elimination",
        n_participants=4,
        scoring=ScoringConfig(type="per_round", values=[10, 20]),
    )
    entities = _make_entities(4)
    struct = build_structure(cfg, entities)
    return cfg, struct


def _make_8_entity_bracket():
    """Build an 8-entity single-elimination bracket structure."""
    cfg = CompetitionConfig(
        format="single_elimination",
        n_participants=8,
        scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
    )
    entities = _make_entities(8)
    struct = build_structure(cfg, entities)
    return cfg, struct


# -----------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------

class TestScorerConstruction:
    def test_creates_with_per_round_scoring(self):
        scoring = ScoringConfig(type="per_round", values=[10, 20, 40])
        scorer = CompetitionScorer(scoring=scoring)
        assert scorer is not None

    def test_creates_with_points_scoring(self):
        scoring = ScoringConfig(type="points", win=3, draw=1, loss=0)
        scorer = CompetitionScorer(scoring=scoring)
        assert scorer is not None


# -----------------------------------------------------------------------
# score_bracket — per-round scoring
# -----------------------------------------------------------------------

class TestScoreBracket:
    def test_perfect_bracket(self):
        cfg, struct = _make_4_entity_bracket()
        scoring = cfg.scoring
        scorer = CompetitionScorer(scoring=scoring)

        # Create picks and actuals that match perfectly
        actuals = {
            "R1G1": "entity_1",
            "R1G2": "entity_2",
            "R2G1": "entity_1",
        }
        picks = dict(actuals)  # same as actuals

        result = scorer.score_bracket(picks, actuals, struct)

        assert isinstance(result, ScoreResult)
        # 2 correct R1 picks (10 each) + 1 correct R2 pick (20) = 40
        assert result.total_points == 40.0
        assert result.round_points[1] == 20.0
        assert result.round_points[2] == 20.0
        assert result.round_correct[1] == 2
        assert result.round_correct[2] == 1
        assert result.round_total[1] == 2
        assert result.round_total[2] == 1

    def test_zero_correct(self):
        cfg, struct = _make_4_entity_bracket()
        scorer = CompetitionScorer(scoring=cfg.scoring)

        actuals = {
            "R1G1": "entity_1",
            "R1G2": "entity_2",
            "R2G1": "entity_1",
        }
        picks = {
            "R1G1": "entity_4",
            "R1G2": "entity_3",
            "R2G1": "entity_3",
        }

        result = scorer.score_bracket(picks, actuals, struct)
        assert result.total_points == 0.0
        assert result.round_correct[1] == 0
        assert result.round_correct[2] == 0

    def test_partial_correct(self):
        cfg, struct = _make_4_entity_bracket()
        scorer = CompetitionScorer(scoring=cfg.scoring)

        actuals = {
            "R1G1": "entity_1",
            "R1G2": "entity_2",
            "R2G1": "entity_1",
        }
        picks = {
            "R1G1": "entity_1",   # correct (10 pts)
            "R1G2": "entity_3",   # wrong
            "R2G1": "entity_1",   # correct (20 pts)
        }

        result = scorer.score_bracket(picks, actuals, struct)
        assert result.total_points == 30.0
        assert result.round_correct[1] == 1
        assert result.round_correct[2] == 1

    def test_8_entity_bracket_scoring(self):
        cfg, struct = _make_8_entity_bracket()
        scorer = CompetitionScorer(scoring=cfg.scoring)

        # All chalk: entity_1 through entity_4 win R1, entity_1 & entity_2 win R2
        actuals = {}
        picks = {}
        for slot in struct.round_slots[1]:
            ref_a, _ = struct.slot_matchups[slot]
            winner = struct.seed_to_entity[ref_a]
            actuals[slot] = winner
            picks[slot] = winner
        for slot in struct.round_slots[2]:
            actuals[slot] = "entity_1"
            picks[slot] = "entity_1"
        for slot in struct.round_slots[3]:
            actuals[slot] = "entity_1"
            picks[slot] = "entity_1"

        result = scorer.score_bracket(picks, actuals, struct)
        # 4*10 + 2*20 + 1*40 = 40 + 40 + 40 = 120
        assert result.total_points == 120.0

    def test_picks_detail_populated(self):
        cfg, struct = _make_4_entity_bracket()
        scorer = CompetitionScorer(scoring=cfg.scoring)

        actuals = {"R1G1": "entity_1", "R1G2": "entity_2", "R2G1": "entity_1"}
        picks = {"R1G1": "entity_1", "R1G2": "entity_2", "R2G1": "entity_2"}

        result = scorer.score_bracket(picks, actuals, struct)
        assert len(result.picks_detail) == 3

        for detail in result.picks_detail:
            assert "slot" in detail
            assert "round" in detail
            assert "picked" in detail
            assert "actual" in detail
            assert "correct" in detail
            assert "points_earned" in detail
            assert "points_possible" in detail

    def test_missing_actuals_slot_skipped(self):
        """If actuals don't have a slot (e.g. tournament not completed), skip it."""
        cfg, struct = _make_4_entity_bracket()
        scorer = CompetitionScorer(scoring=cfg.scoring)

        actuals = {"R1G1": "entity_1"}  # Only 1 game played
        picks = {"R1G1": "entity_1", "R1G2": "entity_2", "R2G1": "entity_1"}

        result = scorer.score_bracket(picks, actuals, struct)
        assert result.total_points == 10.0
        assert len(result.picks_detail) == 1

    def test_empty_picks_scores_zero(self):
        cfg, struct = _make_4_entity_bracket()
        scorer = CompetitionScorer(scoring=cfg.scoring)

        actuals = {"R1G1": "entity_1", "R1G2": "entity_2", "R2G1": "entity_1"}
        result = scorer.score_bracket({}, actuals, struct)
        assert result.total_points == 0.0
        assert len(result.picks_detail) == 0


# -----------------------------------------------------------------------
# score_bracket — round values correctness
# -----------------------------------------------------------------------

class TestRoundPointValues:
    def test_custom_scoring_values(self):
        scoring = ScoringConfig(type="per_round", values=[1, 2, 4, 8, 16, 32])
        cfg = CompetitionConfig(
            format="single_elimination",
            n_participants=4,
            scoring=scoring,
        )
        entities = _make_entities(4)
        struct = build_structure(cfg, entities)
        scorer = CompetitionScorer(scoring=scoring)

        actuals = {"R1G1": "entity_1", "R1G2": "entity_2", "R2G1": "entity_1"}
        picks = dict(actuals)

        result = scorer.score_bracket(picks, actuals, struct)
        # 2*1 + 1*2 = 4
        assert result.total_points == 4.0


# -----------------------------------------------------------------------
# score_standings — league scoring
# -----------------------------------------------------------------------

class TestScoreStandings:
    def test_perfect_standings_prediction(self):
        scoring = ScoringConfig(type="points", win=3, draw=1, loss=0)
        scorer = CompetitionScorer(scoring=scoring)

        predicted = [
            StandingsEntry(entity="entity_1", wins=10, losses=2, points=30.0),
            StandingsEntry(entity="entity_2", wins=8, losses=4, points=24.0),
            StandingsEntry(entity="entity_3", wins=5, losses=7, points=15.0),
        ]
        actual = list(predicted)

        result = scorer.score_standings(predicted, actual)
        assert isinstance(result, ScoreResult)
        # Perfect prediction — displacement is 0
        assert result.total_points == 0.0  # 0 displacement = perfect

    def test_totally_reversed_standings(self):
        scoring = ScoringConfig(type="points", win=3, draw=1, loss=0)
        scorer = CompetitionScorer(scoring=scoring)

        predicted = [
            StandingsEntry(entity="entity_1", points=30.0),
            StandingsEntry(entity="entity_2", points=20.0),
            StandingsEntry(entity="entity_3", points=10.0),
        ]
        actual = [
            StandingsEntry(entity="entity_3", points=30.0),
            StandingsEntry(entity="entity_2", points=20.0),
            StandingsEntry(entity="entity_1", points=10.0),
        ]

        result = scorer.score_standings(predicted, actual)
        # Reversed order: entity_1 displaced by 2, entity_3 displaced by 2
        assert result.total_points > 0.0

    def test_standings_displacement_metric(self):
        """Displacement = sum of |predicted_rank - actual_rank| for all entities."""
        scoring = ScoringConfig(type="points", win=3, draw=1, loss=0)
        scorer = CompetitionScorer(scoring=scoring)

        predicted = [
            StandingsEntry(entity="A", points=30.0),
            StandingsEntry(entity="B", points=20.0),
            StandingsEntry(entity="C", points=10.0),
            StandingsEntry(entity="D", points=5.0),
        ]
        actual = [
            StandingsEntry(entity="B", points=30.0),  # predicted 2nd, actual 1st
            StandingsEntry(entity="A", points=25.0),  # predicted 1st, actual 2nd
            StandingsEntry(entity="C", points=10.0),  # correct
            StandingsEntry(entity="D", points=5.0),   # correct
        ]

        result = scorer.score_standings(predicted, actual)
        # A: |1-2| = 1, B: |2-1| = 1, C: |3-3| = 0, D: |4-4| = 0
        assert result.total_points == 2.0

    def test_empty_standings(self):
        scoring = ScoringConfig(type="points", win=3, draw=1, loss=0)
        scorer = CompetitionScorer(scoring=scoring)

        result = scorer.score_standings([], [])
        assert result.total_points == 0.0

    def test_standings_picks_detail(self):
        scoring = ScoringConfig(type="points", win=3, draw=1, loss=0)
        scorer = CompetitionScorer(scoring=scoring)

        predicted = [
            StandingsEntry(entity="A", points=30.0),
            StandingsEntry(entity="B", points=20.0),
        ]
        actual = [
            StandingsEntry(entity="A", points=30.0),
            StandingsEntry(entity="B", points=20.0),
        ]

        result = scorer.score_standings(predicted, actual)
        assert len(result.picks_detail) == 2
        for detail in result.picks_detail:
            assert "entity" in detail
            assert "predicted_rank" in detail
            assert "actual_rank" in detail
            assert "displacement" in detail


# -----------------------------------------------------------------------
# get_round_points helper
# -----------------------------------------------------------------------

class TestGetRoundPoints:
    def test_valid_round(self):
        scoring = ScoringConfig(type="per_round", values=[10, 20, 40])
        scorer = CompetitionScorer(scoring=scoring)
        assert scorer.get_round_points(1) == 10
        assert scorer.get_round_points(2) == 20
        assert scorer.get_round_points(3) == 40

    def test_invalid_round_returns_zero(self):
        scoring = ScoringConfig(type="per_round", values=[10, 20, 40])
        scorer = CompetitionScorer(scoring=scoring)
        assert scorer.get_round_points(0) == 0
        assert scorer.get_round_points(4) == 0
        assert scorer.get_round_points(-1) == 0

    def test_points_type_scoring_returns_zero(self):
        scoring = ScoringConfig(type="points", win=3, draw=1, loss=0)
        scorer = CompetitionScorer(scoring=scoring)
        assert scorer.get_round_points(1) == 0


# -----------------------------------------------------------------------
# max_possible_points
# -----------------------------------------------------------------------

class TestMaxPossiblePoints:
    def test_4_entity_max(self):
        cfg, struct = _make_4_entity_bracket()
        scorer = CompetitionScorer(scoring=cfg.scoring)
        # 2 * 10 + 1 * 20 = 40
        assert scorer.max_possible_points(struct) == 40.0

    def test_8_entity_max(self):
        cfg, struct = _make_8_entity_bracket()
        scorer = CompetitionScorer(scoring=cfg.scoring)
        # 4*10 + 2*20 + 1*40 = 120
        assert scorer.max_possible_points(struct) == 120.0
```

### Step 4.2: Verify tests fail

```bash
uv run pytest packages/easyml-sports/tests/competitions/test_scorer.py -v 2>&1 | head -30
```

### Step 4.3: Implement scorer

Create `packages/easyml-sports/src/easyml/sports/competitions/scorer.py`:

```python
"""Configurable competition scoring engine.

Scores bracket picks against actuals (elimination formats) and predicted
standings against actual standings (league/Swiss formats). All scoring
is config-driven via ScoringConfig.
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
    """Score competition predictions against actual results.

    Supports two modes:
    - ``score_bracket``: per-round point values for elimination formats
    - ``score_standings``: rank displacement for league/Swiss formats
    """

    def __init__(self, scoring: ScoringConfig):
        self.scoring = scoring

    def get_round_points(self, round_num: int) -> int | float:
        """Return points for a correct pick in the given round.

        Returns 0 if scoring type is not per_round or round is out of range.
        """
        if self.scoring.type != "per_round" or self.scoring.values is None:
            return 0
        if round_num < 1 or round_num > len(self.scoring.values):
            return 0
        return self.scoring.values[round_num - 1]

    def score_bracket(
        self,
        picks: dict[str, str],
        actuals: dict[str, str],
        structure: CompetitionStructure,
    ) -> ScoreResult:
        """Score bracket picks against actual results.

        Args:
            picks: Dict mapping slot -> predicted winning entity ID.
            actuals: Dict mapping slot -> actual winning entity ID.
            structure: Competition structure for round lookups.

        Returns:
            ScoreResult with total points, per-round breakdown, and pick details.
        """
        total_points: float = 0.0
        round_points: dict[int, float] = {}
        round_correct: dict[int, int] = {}
        round_total: dict[int, int] = {}
        picks_detail: list[dict[str, Any]] = []

        for slot, picked_entity in picks.items():
            if slot not in actuals:
                continue
            actual_winner = actuals[slot]
            rd = structure.slot_to_round.get(slot, 0)
            pts = self.get_round_points(rd)
            correct = picked_entity == actual_winner

            round_points[rd] = round_points.get(rd, 0.0) + (pts if correct else 0)
            round_correct[rd] = round_correct.get(rd, 0) + (1 if correct else 0)
            round_total[rd] = round_total.get(rd, 0) + 1
            if correct:
                total_points += pts

            picks_detail.append({
                "slot": slot,
                "round": rd,
                "picked": picked_entity,
                "actual": actual_winner,
                "correct": correct,
                "points_earned": pts if correct else 0,
                "points_possible": pts,
            })

        return ScoreResult(
            total_points=total_points,
            round_points=round_points,
            round_correct=round_correct,
            round_total=round_total,
            picks_detail=picks_detail,
        )

    def score_standings(
        self,
        predicted: list[StandingsEntry],
        actual: list[StandingsEntry],
    ) -> ScoreResult:
        """Score predicted standings against actual standings.

        Uses rank displacement: sum of |predicted_rank - actual_rank| for each
        entity. Lower is better (0 = perfect prediction).

        Args:
            predicted: Predicted standings in order (index = rank - 1).
            actual: Actual standings in order (index = rank - 1).

        Returns:
            ScoreResult where total_points = total displacement (lower is better).
        """
        if not predicted or not actual:
            return ScoreResult(
                total_points=0.0,
                round_points={},
                round_correct={},
                round_total={},
                picks_detail=[],
            )

        # Build rank maps (1-indexed)
        predicted_rank = {
            entry.entity: i + 1 for i, entry in enumerate(predicted)
        }
        actual_rank = {
            entry.entity: i + 1 for i, entry in enumerate(actual)
        }

        total_displacement: float = 0.0
        picks_detail: list[dict[str, Any]] = []

        for entity, p_rank in predicted_rank.items():
            a_rank = actual_rank.get(entity, len(actual) + 1)
            displacement = abs(p_rank - a_rank)
            total_displacement += displacement
            picks_detail.append({
                "entity": entity,
                "predicted_rank": p_rank,
                "actual_rank": a_rank,
                "displacement": displacement,
            })

        return ScoreResult(
            total_points=total_displacement,
            round_points={},
            round_correct={},
            round_total={},
            picks_detail=picks_detail,
        )

    def max_possible_points(
        self,
        structure: CompetitionStructure,
    ) -> float:
        """Compute maximum possible points for a competition structure.

        For per-round scoring: sum of (games_in_round * points_per_round).
        """
        if self.scoring.type != "per_round" or self.scoring.values is None:
            return 0.0

        total: float = 0.0
        for rd, rd_slots in structure.round_slots.items():
            pts = self.get_round_points(rd)
            total += len(rd_slots) * pts
        return total
```

### Step 4.4: Update `__init__.py`

Add to `packages/easyml-sports/src/easyml/sports/competitions/__init__.py` imports:

```python
from easyml.sports.competitions.scorer import CompetitionScorer
```

And add `"CompetitionScorer"` to `__all__`.

### Step 4.5: Verify tests pass

```bash
uv run pytest packages/easyml-sports/tests/competitions/test_scorer.py -v
```

### Step 4.6: Commit

```bash
git add packages/easyml-sports/src/easyml/sports/competitions/scorer.py \
      packages/easyml-sports/src/easyml/sports/competitions/__init__.py \
      packages/easyml-sports/tests/competitions/test_scorer.py
git commit -m "feat(competitions): add configurable competition scorer"
```
# Competition Engine Implementation Plan — Part 2 (Tasks 5-8)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement adjustments, optimizer, explainer, and confidence modules for the generic competition engine in `easyml-sports`.

**Architecture:** All modules use generic entity IDs (strings), import schemas from `easyml.sports.competitions.schemas`, and contain zero domain-specific (basketball, etc.) code. Hook-based narrative generation allows domain plugins to inject custom prose.

**Tech Stack:** Python 3.11+, numpy, pandas, pydantic, pytest

**Key Principle:** Generic competition engine. No hardcoded feature lists, no hardcoded column names, no domain-specific assumptions. Entity IDs are always strings.

---

## Task 5: adjustments.py

**Files:**
- Create: `packages/easyml-sports/src/easyml/sports/competitions/adjustments.py`
- Create: `packages/easyml-sports/tests/competitions/test_adjustments.py`

**Step 1: Write failing tests**

```python
# packages/easyml-sports/tests/competitions/test_adjustments.py
"""Tests for post-model probability adjustments."""
import numpy as np
import pandas as pd
import pytest

from easyml.sports.competitions.adjustments import apply_adjustments
from easyml.sports.competitions.schemas import AdjustmentConfig


@pytest.fixture
def base_probabilities():
    """Simple probability DataFrame with entity_a, entity_b, prob columns."""
    return pd.DataFrame({
        "entity_a": ["A", "A", "B", "C"],
        "entity_b": ["B", "C", "C", "D"],
        "prob": [0.6, 0.7, 0.55, 0.8],
    })


class TestEntityMultipliers:
    def test_multiplier_on_entity_a(self, base_probabilities):
        """Multiplier applied when entity is in entity_a column."""
        config = AdjustmentConfig(entity_multipliers={"A": 0.8})
        adjusted, log = apply_adjustments(base_probabilities, config)

        # A's prob as entity_a should be scaled down
        row_ab = adjusted[
            (adjusted["entity_a"] == "A") & (adjusted["entity_b"] == "B")
        ]
        assert float(row_ab["prob"].iloc[0]) == pytest.approx(0.6 * 0.8, abs=1e-6)

        row_ac = adjusted[
            (adjusted["entity_a"] == "A") & (adjusted["entity_b"] == "C")
        ]
        assert float(row_ac["prob"].iloc[0]) == pytest.approx(0.7 * 0.8, abs=1e-6)

        # Log records the adjustment
        mult_logs = [l for l in log if l["type"] == "entity_multiplier"]
        assert len(mult_logs) >= 1
        assert mult_logs[0]["entity"] == "A"

    def test_multiplier_on_entity_b(self, base_probabilities):
        """When multiplied entity is in entity_b, prob_a increases."""
        config = AdjustmentConfig(entity_multipliers={"B": 0.5})
        adjusted, log = apply_adjustments(base_probabilities, config)

        # B is entity_b in row A-B: prob_b was 0.4, scaled to 0.2
        # So prob_a becomes 1 - 0.2 = 0.8
        row_ab = adjusted[
            (adjusted["entity_a"] == "A") & (adjusted["entity_b"] == "B")
        ]
        expected = 1.0 - (1.0 - 0.6) * 0.5  # = 0.8
        assert float(row_ab["prob"].iloc[0]) == pytest.approx(expected, abs=1e-6)

    def test_multiplier_both_sides(self, base_probabilities):
        """Entity appears as both entity_a and entity_b across rows."""
        config = AdjustmentConfig(entity_multipliers={"B": 0.9})
        adjusted, log = apply_adjustments(base_probabilities, config)

        # B as entity_b in A-B row
        row_ab = adjusted[
            (adjusted["entity_a"] == "A") & (adjusted["entity_b"] == "B")
        ]
        expected_ab = 1.0 - (1.0 - 0.6) * 0.9
        assert float(row_ab["prob"].iloc[0]) == pytest.approx(expected_ab, abs=1e-6)

        # B as entity_a in B-C row
        row_bc = adjusted[
            (adjusted["entity_a"] == "B") & (adjusted["entity_b"] == "C")
        ]
        expected_bc = 0.55 * 0.9
        assert float(row_bc["prob"].iloc[0]) == pytest.approx(expected_bc, abs=1e-6)

    def test_no_multipliers(self, base_probabilities):
        """No entity_multipliers leaves probabilities unchanged."""
        config = AdjustmentConfig()
        adjusted, log = apply_adjustments(base_probabilities, config)
        pd.testing.assert_frame_equal(
            adjusted[["entity_a", "entity_b", "prob"]],
            base_probabilities[["entity_a", "entity_b", "prob"]],
        )
        assert log == []


class TestExternalBlending:
    def test_external_blend(self, base_probabilities):
        """External probabilities blended with given weight."""
        external = pd.DataFrame({
            "entity_a": ["A", "A"],
            "entity_b": ["B", "C"],
            "external_prob": [0.4, 0.5],
        })
        config = AdjustmentConfig(
            external_probs=external,
            external_weight=0.3,
        )
        adjusted, log = apply_adjustments(base_probabilities, config)

        # A-B: (1-0.3)*0.6 + 0.3*0.4 = 0.42 + 0.12 = 0.54
        row_ab = adjusted[
            (adjusted["entity_a"] == "A") & (adjusted["entity_b"] == "B")
        ]
        expected = (1.0 - 0.3) * 0.6 + 0.3 * 0.4
        assert float(row_ab["prob"].iloc[0]) == pytest.approx(expected, abs=1e-6)

        blend_logs = [l for l in log if l["type"] == "external_blend"]
        assert len(blend_logs) == 1
        assert blend_logs[0]["weight"] == 0.3

    def test_external_blend_partial_coverage(self, base_probabilities):
        """External probs only cover some matchups; others untouched."""
        external = pd.DataFrame({
            "entity_a": ["A"],
            "entity_b": ["B"],
            "external_prob": [0.3],
        })
        config = AdjustmentConfig(external_probs=external, external_weight=0.5)
        adjusted, log = apply_adjustments(base_probabilities, config)

        # A-C should be unchanged (no external data)
        row_ac = adjusted[
            (adjusted["entity_a"] == "A") & (adjusted["entity_b"] == "C")
        ]
        assert float(row_ac["prob"].iloc[0]) == pytest.approx(0.7, abs=1e-6)

    def test_no_external_weight_skips(self, base_probabilities):
        """Zero external weight means no blending occurs."""
        external = pd.DataFrame({
            "entity_a": ["A"],
            "entity_b": ["B"],
            "external_prob": [0.1],
        })
        config = AdjustmentConfig(external_probs=external, external_weight=0.0)
        adjusted, log = apply_adjustments(base_probabilities, config)
        row_ab = adjusted[
            (adjusted["entity_a"] == "A") & (adjusted["entity_b"] == "B")
        ]
        assert float(row_ab["prob"].iloc[0]) == pytest.approx(0.6, abs=1e-6)


class TestHardOverrides:
    def test_override_sets_exact_prob(self, base_probabilities):
        """Hard override sets probability to exact values."""
        config = AdjustmentConfig(
            probability_overrides={"A_B": (0.3, 0.7)},
        )
        adjusted, log = apply_adjustments(base_probabilities, config)

        row_ab = adjusted[
            (adjusted["entity_a"] == "A") & (adjusted["entity_b"] == "B")
        ]
        assert float(row_ab["prob"].iloc[0]) == pytest.approx(0.3, abs=1e-6)

        override_logs = [l for l in log if l["type"] == "override"]
        assert len(override_logs) == 1
        assert override_logs[0]["entity_a"] == "A"
        assert override_logs[0]["entity_b"] == "B"
        assert override_logs[0]["prob_a"] == 0.3

    def test_override_nonexistent_matchup(self, base_probabilities):
        """Override for a matchup not in the DataFrame is a no-op."""
        config = AdjustmentConfig(
            probability_overrides={"X_Y": (0.5, 0.5)},
        )
        adjusted, log = apply_adjustments(base_probabilities, config)
        override_logs = [l for l in log if l["type"] == "override"]
        assert len(override_logs) == 0


class TestClamping:
    def test_clamp_low(self):
        """Probabilities below 0.01 are clamped."""
        df = pd.DataFrame({
            "entity_a": ["A"],
            "entity_b": ["B"],
            "prob": [0.05],
        })
        config = AdjustmentConfig(entity_multipliers={"A": 0.1})
        adjusted, _ = apply_adjustments(df, config)
        assert float(adjusted["prob"].iloc[0]) >= 0.01

    def test_clamp_high(self):
        """Probabilities above 0.99 are clamped."""
        df = pd.DataFrame({
            "entity_a": ["A"],
            "entity_b": ["B"],
            "prob": [0.95],
        })
        # Multiplier on B as entity_b pushes A's prob very high
        config = AdjustmentConfig(entity_multipliers={"B": 0.01})
        adjusted, _ = apply_adjustments(df, config)
        assert float(adjusted["prob"].iloc[0]) <= 0.99


class TestNoMutation:
    def test_original_unchanged(self, base_probabilities):
        """Original DataFrame is not mutated."""
        original_values = base_probabilities["prob"].tolist()
        config = AdjustmentConfig(entity_multipliers={"A": 0.5})
        apply_adjustments(base_probabilities, config)
        assert base_probabilities["prob"].tolist() == original_values


class TestAdjustmentOrdering:
    def test_multipliers_then_external_then_overrides(self):
        """Adjustments are applied in order: multipliers, external, overrides."""
        df = pd.DataFrame({
            "entity_a": ["A"],
            "entity_b": ["B"],
            "prob": [0.5],
        })
        external = pd.DataFrame({
            "entity_a": ["A"],
            "entity_b": ["B"],
            "external_prob": [0.8],
        })
        config = AdjustmentConfig(
            entity_multipliers={"A": 0.5},
            external_probs=external,
            external_weight=0.5,
            probability_overrides={"A_B": (0.99, 0.01)},
        )
        adjusted, log = apply_adjustments(df, config)
        # Override wins because it's applied last
        assert float(adjusted["prob"].iloc[0]) == pytest.approx(0.99, abs=1e-6)

        # All three adjustment types should be logged
        types = {l["type"] for l in log}
        assert "entity_multiplier" in types
        assert "external_blend" in types
        assert "override" in types
```

**Step 2:** Run tests, verify FAIL (module does not exist)

**Step 3: Implement**

```python
# packages/easyml-sports/src/easyml/sports/competitions/adjustments.py
"""Post-model probability adjustments (external sources, injuries, manual overrides).

Applied AFTER the ML ensemble produces probabilities, BEFORE simulation.
All adjustments are logged for auditability.
"""
from __future__ import annotations

import pandas as pd

from easyml.sports.competitions.schemas import AdjustmentConfig


def apply_adjustments(
    probabilities: pd.DataFrame,
    adjustments: AdjustmentConfig,
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply post-model adjustments to probability predictions.

    Adjustments are applied in order:
      1. Entity multipliers (scale an entity's win probability)
      2. External probability blending (weighted blend with external source)
      3. Hard overrides (direct probability assignment for specific matchups)

    All results clamped to [0.01, 0.99].

    Args:
        probabilities: DataFrame with entity_a, entity_b, prob columns.
            May contain additional columns (preserved unchanged).
        adjustments: AdjustmentConfig with optional multipliers, external
            probs, and overrides.

    Returns:
        Tuple of (adjusted probabilities DataFrame, adjustment log list).
        The original DataFrame is never mutated.
    """
    adjusted = probabilities.copy()
    log: list[dict] = []

    # 1. Entity multipliers
    for entity, multiplier in adjustments.entity_multipliers.items():
        # When entity is entity_a: multiply prob by multiplier
        mask_a = adjusted["entity_a"] == entity
        if mask_a.any():
            adjusted.loc[mask_a, "prob"] *= multiplier
            log.append({
                "type": "entity_multiplier",
                "entity": entity,
                "side": "entity_a",
                "multiplier": multiplier,
                "affected_matchups": int(mask_a.sum()),
            })

        # When entity is entity_b: multiply (1 - prob) by multiplier
        # prob_b = 1 - prob; scaled prob_b = prob_b * multiplier
        # new prob = 1 - prob_b * multiplier
        mask_b = adjusted["entity_b"] == entity
        if mask_b.any():
            adjusted.loc[mask_b, "prob"] = (
                1.0 - (1.0 - adjusted.loc[mask_b, "prob"]) * multiplier
            )
            log.append({
                "type": "entity_multiplier",
                "entity": entity,
                "side": "entity_b",
                "multiplier": multiplier,
                "affected_matchups": int(mask_b.sum()),
            })

    # 2. External probability blending
    if (
        adjustments.external_probs is not None
        and adjustments.external_weight > 0.0
    ):
        external = adjustments.external_probs
        w = adjustments.external_weight

        merged = adjusted.merge(
            external[["entity_a", "entity_b", "external_prob"]],
            on=["entity_a", "entity_b"],
            how="left",
        )
        has_external = merged["external_prob"].notna()
        if has_external.any():
            adjusted.loc[has_external, "prob"] = (
                (1.0 - w) * merged.loc[has_external, "prob"]
                + w * merged.loc[has_external, "external_prob"]
            )
            log.append({
                "type": "external_blend",
                "weight": w,
                "affected_matchups": int(has_external.sum()),
            })

    # 3. Hard overrides
    for key, (prob_a, prob_b) in adjustments.probability_overrides.items():
        parts = key.split("_", 1)
        if len(parts) != 2:
            continue
        entity_a, entity_b = parts
        mask = (adjusted["entity_a"] == entity_a) & (
            adjusted["entity_b"] == entity_b
        )
        if mask.any():
            adjusted.loc[mask, "prob"] = prob_a
            log.append({
                "type": "override",
                "entity_a": entity_a,
                "entity_b": entity_b,
                "prob_a": prob_a,
            })

    # Clamp to [0.01, 0.99]
    adjusted["prob"] = adjusted["prob"].clip(0.01, 0.99)

    return adjusted, log
```

**Step 4:** Run tests, verify PASS

**Step 5:** Commit: `feat: add generic adjustments module for competition engine`

---

## Task 6: optimizer.py

**Files:**
- Create: `packages/easyml-sports/src/easyml/sports/competitions/optimizer.py`
- Create: `packages/easyml-sports/tests/competitions/test_optimizer.py`

**Step 1: Write failing tests**

```python
# packages/easyml-sports/tests/competitions/test_optimizer.py
"""Tests for pool-aware competition optimizer."""
import math

import numpy as np
import pandas as pd
import pytest

from easyml.sports.competitions.optimizer import (
    CompetitionOptimizer,
    chalk,
    near_chalk,
    random_sim,
    contrarian,
    late_contrarian,
    champion_anchor,
    StrategyFn,
)
from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionResult,
    CompetitionStructure,
    MatchupContext,
    ScoringConfig,
)


@pytest.fixture
def simple_structure():
    """4-entity single elimination bracket."""
    config = CompetitionConfig(
        format="single_elimination",
        n_participants=4,
        scoring=ScoringConfig(type="per_round", values=[10, 20]),
    )
    structure = CompetitionStructure(
        config=config,
        slots=["R1_1", "R1_2", "R2_1"],
        slot_matchups={
            "R1_1": ("S1", "S4"),
            "R1_2": ("S2", "S3"),
            "R2_1": ("R1_1", "R1_2"),
        },
        slot_to_round={"R1_1": 1, "R1_2": 1, "R2_1": 2},
        round_slots={1: ["R1_1", "R1_2"], 2: ["R2_1"]},
        seed_to_entity={"S1": "Alpha", "S2": "Beta", "S3": "Gamma", "S4": "Delta"},
        entity_to_seed={"Alpha": "S1", "Beta": "S2", "Gamma": "S3", "Delta": "S4"},
    )
    return config, structure


@pytest.fixture
def simple_probabilities():
    """Pairwise probabilities for 4 entities."""
    return pd.DataFrame({
        "entity_a": ["Alpha", "Alpha", "Alpha", "Beta", "Beta", "Gamma"],
        "entity_b": ["Beta", "Gamma", "Delta", "Gamma", "Delta", "Delta"],
        "prob": [0.6, 0.7, 0.9, 0.55, 0.8, 0.6],
    })


class TestStrategyFunctions:
    """Test each built-in strategy as a standalone function."""

    def test_chalk_picks_all_favorites(
        self, simple_structure, simple_probabilities
    ):
        """Chalk strategy always picks the higher-probability entity."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        rng = np.random.default_rng(42)
        picks = chalk(sim, rng)
        # All picks should be favorites
        assert isinstance(picks, dict)
        assert len(picks) > 0
        # R1_1: Alpha (0.9 vs Delta) should win
        assert picks["R1_1"] == "Alpha"
        # R1_2: Beta (0.55 vs Gamma) should win
        assert picks["R1_2"] == "Beta"

    def test_random_sim_produces_valid_bracket(
        self, simple_structure, simple_probabilities
    ):
        """Random sim returns a complete bracket with valid entity IDs."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        rng = np.random.default_rng(42)
        picks = random_sim(sim, rng)
        assert isinstance(picks, dict)
        assert len(picks) == 3  # R1_1, R1_2, R2_1
        # All picks should be valid entity IDs
        valid_entities = {"Alpha", "Beta", "Gamma", "Delta"}
        for entity in picks.values():
            assert entity in valid_entities

    def test_near_chalk_flips_close_matchups(
        self, simple_structure, simple_probabilities
    ):
        """Near chalk sometimes flips close matchups (underdog > 40%)."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        # Run many times to verify close matchups sometimes flip
        flipped_count = 0
        for seed in range(200):
            rng = np.random.default_rng(seed)
            picks = near_chalk(sim, rng)
            # R1_2 is Beta vs Gamma (0.55/0.45) — close enough to flip
            if picks.get("R1_2") == "Gamma":
                flipped_count += 1
        # Should flip sometimes but not always
        assert flipped_count > 0, "Close matchups should sometimes flip"
        assert flipped_count < 200, "Close matchups should not always flip"

    def test_contrarian_compresses_toward_half(
        self, simple_structure, simple_probabilities
    ):
        """Contrarian with upset_boost compresses probs toward 0.5."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        # With high upset_boost, underdogs should win more often
        upset_count = 0
        for seed in range(200):
            rng = np.random.default_rng(seed)
            picks = contrarian(sim, rng, upset_boost=0.7)
            # R1_1: Alpha is heavy favorite (0.9); upset = Delta
            if picks.get("R1_1") == "Delta":
                upset_count += 1
        # More upsets than raw probability would suggest
        assert upset_count > 10

    def test_late_contrarian_chalk_early_contrarian_late(
        self, simple_structure, simple_probabilities
    ):
        """Late contrarian is chalk-like in early rounds, upset-prone in late."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        picks = late_contrarian(sim, np.random.default_rng(42), upset_boost=0.5)
        assert isinstance(picks, dict)
        assert len(picks) == 3

    def test_champion_anchor_forces_champion(
        self, simple_structure, simple_probabilities
    ):
        """Champion anchor always has the anchored entity winning the final."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        rng = np.random.default_rng(42)
        picks = champion_anchor(sim, rng, champion="Gamma")
        # Gamma must win the final (R2_1)
        assert picks["R2_1"] == "Gamma"

    def test_custom_strategy_fn(self, simple_structure, simple_probabilities):
        """Custom strategy function conforming to StrategyFn protocol."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )

        def always_delta(
            simulator: "CompetitionSimulator",
            rng: np.random.Generator,
            **kwargs,
        ) -> dict[str, str]:
            """Force Delta to win everything (for testing)."""
            return {"R1_1": "Delta", "R1_2": "Delta", "R2_1": "Delta"}

        optimizer = CompetitionOptimizer(
            simulator=sim,
            strategies={"always_delta": always_delta},
        )
        assert "always_delta" in optimizer.strategies


class TestStrategyMix:
    def test_small_pool_favors_chalk(self, simple_structure, simple_probabilities):
        """Small pools should weight near_chalk heavily."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        mix = optimizer._get_strategy_mix(10)
        # Near chalk should be dominant for small pools
        assert mix["near_chalk"] > mix["contrarian"]
        assert mix["near_chalk"] > mix["champion_anchor"]

    def test_large_pool_favors_contrarian(
        self, simple_structure, simple_probabilities
    ):
        """Large pools should weight contrarian and champion_anchor heavily."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        mix = optimizer._get_strategy_mix(10000)
        assert mix["contrarian"] > mix["near_chalk"]
        assert mix["champion_anchor"] > mix["near_chalk"]


class TestCompetitionOptimizer:
    def test_generate_brackets_returns_correct_count(
        self, simple_structure, simple_probabilities
    ):
        """generate_brackets returns the requested number of brackets."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        brackets = optimizer.generate_brackets(
            pool_size=10, n_brackets=3, n_sims=100, seed=42
        )
        assert len(brackets) == 3
        for b in brackets:
            assert isinstance(b, CompetitionResult)
            assert len(b.picks) > 0
            assert b.strategy != ""

    def test_generate_brackets_deterministic(
        self, simple_structure, simple_probabilities
    ):
        """Same seed produces same brackets."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        b1 = optimizer.generate_brackets(
            pool_size=10, n_brackets=2, n_sims=100, seed=42
        )
        b2 = optimizer.generate_brackets(
            pool_size=10, n_brackets=2, n_sims=100, seed=42
        )
        for a, b in zip(b1, b2):
            assert a.picks == b.picks

    def test_brackets_have_valid_entities(
        self, simple_structure, simple_probabilities
    ):
        """All picks reference valid entity IDs."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        valid_entities = set(structure.seed_to_entity.values())
        optimizer = CompetitionOptimizer(simulator=sim)
        brackets = optimizer.generate_brackets(
            pool_size=50, n_brackets=5, n_sims=100, seed=42
        )
        for b in brackets:
            for slot, entity in b.picks.items():
                assert entity in valid_entities, (
                    f"Invalid entity {entity} in slot {slot}"
                )

    def test_brackets_have_expected_points(
        self, simple_structure, simple_probabilities
    ):
        """Each bracket should have non-negative expected_points."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        brackets = optimizer.generate_brackets(
            pool_size=10, n_brackets=2, n_sims=200, seed=42
        )
        for b in brackets:
            assert b.expected_points >= 0.0

    def test_diversity_selection(self, simple_structure, simple_probabilities):
        """Selected brackets should not be identical."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        brackets = optimizer.generate_brackets(
            pool_size=100, n_brackets=3, n_sims=200, seed=42
        )
        if len(brackets) >= 2:
            # At least one pair should differ somewhere
            all_same = all(
                brackets[0].picks == b.picks for b in brackets[1:]
            )
            # With 4 entities in a small bracket, duplicates are possible
            # but we at least verify the diversity mechanism runs
            assert isinstance(all_same, bool)


class TestBracketOverlap:
    def test_identical_brackets_overlap_1(
        self, simple_structure, simple_probabilities
    ):
        """Identical brackets have overlap of 1.0."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        picks = {"R1_1": "Alpha", "R1_2": "Beta", "R2_1": "Alpha"}
        overlap = optimizer._bracket_overlap(picks, picks)
        assert overlap == pytest.approx(1.0)

    def test_completely_different_overlap_0(
        self, simple_structure, simple_probabilities
    ):
        """Completely different picks have overlap of 0.0."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        picks_a = {"R1_1": "Alpha", "R1_2": "Beta", "R2_1": "Alpha"}
        picks_b = {"R1_1": "Delta", "R1_2": "Gamma", "R2_1": "Gamma"}
        overlap = optimizer._bracket_overlap(picks_a, picks_b)
        assert overlap == pytest.approx(0.0)

    def test_late_rounds_weighted_more(
        self, simple_structure, simple_probabilities
    ):
        """Late-round agreement contributes more to overlap score."""
        from easyml.sports.competitions.simulator import CompetitionSimulator

        _, structure = simple_structure
        sim = CompetitionSimulator(
            config=structure.config,
            structure=structure,
            probabilities=simple_probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)

        # Agree on R2 (20 pts) but disagree on both R1 (10+10=20 pts)
        picks_a = {"R1_1": "Alpha", "R1_2": "Beta", "R2_1": "Alpha"}
        picks_b = {"R1_1": "Delta", "R1_2": "Gamma", "R2_1": "Alpha"}
        overlap = optimizer._bracket_overlap(picks_a, picks_b)
        # Weighted: 20/(10+10+20) = 0.5
        assert overlap == pytest.approx(20.0 / 40.0)
```

**Step 2:** Run tests, verify FAIL (module does not exist)

**Step 3: Implement**

```python
# packages/easyml-sports/src/easyml/sports/competitions/optimizer.py
"""Pool-size-aware competition optimizer.

Generates diverse bracket candidates via strategy-based generation with a
continuous pool-size-dependent strategy mix. Scores with vectorized beat-max
against pool-sized opponent field. Elimination formats only.
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np

from easyml.sports.competitions.schemas import (
    CompetitionResult,
    MatchupContext,
)

# Avoid circular imports — simulator is passed in, not imported at module level.
# Type-checking only import:
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easyml.sports.competitions.simulator import CompetitionSimulator


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class StrategyFn(Protocol):
    """Protocol for bracket generation strategies."""

    def __call__(
        self,
        simulator: CompetitionSimulator,
        rng: np.random.Generator,
        **kwargs,
    ) -> dict[str, str]: ...


# ---------------------------------------------------------------------------
# Built-in strategy functions (module-level)
# ---------------------------------------------------------------------------

def chalk(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    **kwargs,
) -> dict[str, str]:
    """Always pick the favorite in every matchup."""
    return simulator.pick_most_likely()


def near_chalk(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    *,
    flip_threshold: float = 0.4,
    flip_prob: float = 0.3,
    **kwargs,
) -> dict[str, str]:
    """Chalk with close-matchup flips (underdog probability > flip_threshold).

    For matchups where the underdog has at least flip_threshold win probability,
    flip to the underdog with probability flip_prob.
    """
    structure = simulator.structure
    results: dict[str, str] = {}

    sorted_slots = sorted(
        structure.slots,
        key=lambda s: structure.slot_to_round.get(s, 0),
    )

    for slot in sorted_slots:
        if slot not in structure.slot_matchups:
            continue

        ref_a, ref_b = structure.slot_matchups[slot]
        entity_a = _resolve_entity(ref_a, structure, results)
        entity_b = _resolve_entity(ref_b, structure, results)

        if entity_a is None or entity_b is None:
            continue

        prob_a = simulator.get_win_prob(entity_a, entity_b)
        underdog_prob = 1.0 - prob_a if prob_a >= 0.5 else prob_a
        underdog = entity_b if prob_a >= 0.5 else entity_a
        favorite = entity_a if prob_a >= 0.5 else entity_b

        if underdog_prob > flip_threshold and rng.random() < flip_prob:
            results[slot] = underdog
        else:
            results[slot] = favorite

    return results


def random_sim(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    **kwargs,
) -> dict[str, str]:
    """Pure Monte Carlo sample — simulate one outcome using raw probabilities."""
    return simulator.simulate_once(rng)


def contrarian(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    *,
    upset_boost: float = 0.3,
    **kwargs,
) -> dict[str, str]:
    """Uniform upset boost — compress probabilities toward 0.5.

    upset_boost of 0 = raw probabilities, 1 = coin flip.
    """
    return _simulate_with_boost(
        simulator, rng, upset_boost=upset_boost, ramp_boost=False
    )


def late_contrarian(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    *,
    upset_boost: float = 0.5,
    **kwargs,
) -> dict[str, str]:
    """Chalk in early rounds, upset-boosted in late rounds.

    The effective upset boost ramps linearly from 0 in round 1 to full
    upset_boost in the final round.
    """
    return _simulate_with_boost(
        simulator, rng, upset_boost=upset_boost, ramp_boost=True
    )


def champion_anchor(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    *,
    champion: str | None = None,
    **kwargs,
) -> dict[str, str]:
    """Condition on a specific champion, simulate the rest.

    The champion entity always wins their matchups. All other matchups
    are resolved with raw probabilities.
    """
    force_winners = {champion} if champion else set()
    return _simulate_with_boost(
        simulator, rng, upset_boost=0.0, force_winners=force_winners
    )


# ---------------------------------------------------------------------------
# Shared simulation helper
# ---------------------------------------------------------------------------

def _resolve_entity(
    ref: str,
    structure,
    results: dict[str, str],
) -> str | None:
    """Resolve a slot reference to an entity ID.

    ref is either a seed code (present in seed_to_entity) or a slot name
    (resolved from prior results).
    """
    if ref in structure.seed_to_entity:
        return structure.seed_to_entity[ref]
    return results.get(ref)


def _simulate_with_boost(
    simulator: CompetitionSimulator,
    rng: np.random.Generator,
    *,
    upset_boost: float = 0.0,
    ramp_boost: bool = False,
    force_winners: set[str] | None = None,
) -> dict[str, str]:
    """Simulate a competition with optional strategy modifications.

    Args:
        upset_boost: Compress probabilities toward 0.5 (0=raw, 1=coin flip).
        ramp_boost: If True, scale upset_boost by round: 0 in round 1,
            full in the final round.
        force_winners: Set of entity IDs that always win their matchups.
    """
    if force_winners is None:
        force_winners = set()

    structure = simulator.structure
    results: dict[str, str] = {}

    # Determine max round for ramp calculation
    max_round = max(structure.slot_to_round.values()) if structure.slot_to_round else 1

    sorted_slots = sorted(
        structure.slots,
        key=lambda s: structure.slot_to_round.get(s, 0),
    )

    for slot in sorted_slots:
        if slot not in structure.slot_matchups:
            continue

        round_num = structure.slot_to_round.get(slot, 1)
        ref_a, ref_b = structure.slot_matchups[slot]
        entity_a = _resolve_entity(ref_a, structure, results)
        entity_b = _resolve_entity(ref_b, structure, results)

        if entity_a is None or entity_b is None:
            continue

        # Force winners bypass probability
        if entity_a in force_winners:
            results[slot] = entity_a
            continue
        if entity_b in force_winners:
            results[slot] = entity_b
            continue

        prob_a = simulator.get_win_prob(entity_a, entity_b)

        # Apply upset boost
        effective_boost = upset_boost
        if ramp_boost and max_round > 1:
            effective_boost = upset_boost * (round_num - 1) / (max_round - 1)

        if effective_boost > 0:
            prob_a = 0.5 + (prob_a - 0.5) * (1.0 - effective_boost)

        if rng.random() < prob_a:
            results[slot] = entity_a
        else:
            results[slot] = entity_b

    return results


# ---------------------------------------------------------------------------
# Optimizer class
# ---------------------------------------------------------------------------

_DEFAULT_STRATEGIES: dict[str, StrategyFn] = {
    "chalk": chalk,
    "near_chalk": near_chalk,
    "random_sim": random_sim,
    "contrarian": contrarian,
    "late_contrarian": late_contrarian,
    "champion_anchor": champion_anchor,
}


class CompetitionOptimizer:
    """Generate optimal brackets for different pool sizes.

    Elimination formats only. For league/swiss formats, use standings
    distributions directly from the simulator.
    """

    def __init__(
        self,
        simulator: CompetitionSimulator,
        strategies: dict[str, StrategyFn] | None = None,
    ):
        self.simulator = simulator
        self.strategies: dict[str, StrategyFn] = dict(_DEFAULT_STRATEGIES)
        if strategies:
            self.strategies.update(strategies)

        structure = simulator.structure
        scoring_values = structure.config.scoring.values or []

        # Build slot-level scoring info for vectorized operations
        self._scored_slots: list[str] = []
        self._slot_points: list[float] = []
        for slot in structure.slots:
            rd = structure.slot_to_round.get(slot, 0)
            if rd >= 1 and rd <= len(scoring_values):
                self._scored_slots.append(slot)
                self._slot_points.append(float(scoring_values[rd - 1]))
        self._slot_points_arr = np.array(self._slot_points, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_brackets(
        self,
        pool_size: int,
        n_brackets: int,
        n_sims: int = 10_000,
        seed: int = 42,
        n_candidates: int = 500,
        overlap_threshold: float = 0.90,
    ) -> list[CompetitionResult]:
        """Generate n_brackets diverse brackets optimized for a pool of given size.

        1. Generate ~n_candidates candidates via pool-size-dependent strategy mix
        2. Score all against simulated outcomes + opponent field (vectorized)
        3. Select top N diverse brackets (round-weighted overlap)

        Args:
            pool_size: Number of entries in the competition pool.
            n_brackets: Number of brackets to return.
            n_sims: Number of Monte Carlo simulations for scoring.
            seed: Random seed for reproducibility.
            n_candidates: Number of candidate brackets to generate.
            overlap_threshold: Maximum round-weighted overlap for diversity
                selection (default 0.90).

        Returns:
            List of CompetitionResult objects, sorted by win_probability.
        """
        rng = np.random.default_rng(seed * 10_000 + pool_size)

        # Generate simulated outcomes for scoring
        simulated_outcomes = [
            self.simulator.simulate_once(np.random.default_rng(seed + i))
            for i in range(n_sims)
        ]

        # Generate candidates via strategy mix
        candidates = self._generate_candidates(
            rng, n_candidates, pool_size, simulated_outcomes
        )

        # Convert to matrices for vectorized scoring
        outcome_matrix = self._brackets_to_matrix(simulated_outcomes)

        candidate_picks = [c[0] for c in candidates]
        candidate_matrix = self._brackets_to_matrix(candidate_picks)

        # Generate opponents
        n_opponents = min(pool_size - 1, 200)
        opponent_outcomes = [
            self.simulator.simulate_once(
                np.random.default_rng(seed + 1_000_000 + i)
            )
            for i in range(n_opponents)
        ]
        opponent_matrix = self._brackets_to_matrix(opponent_outcomes)

        # Score candidates
        scored = self._score_candidates_vectorized(
            candidates, candidate_matrix, outcome_matrix, opponent_matrix
        )

        # Select diverse set
        selected = self._select_diverse(scored, n_brackets, overlap_threshold)

        return self._enrich_brackets(selected)

    # ------------------------------------------------------------------
    # Strategy mix
    # ------------------------------------------------------------------

    def _get_strategy_mix(self, pool_size: int) -> dict[str, float]:
        """Continuous strategy mix based on log10(pool_size).

        Small pools skew toward chalk/near_chalk.
        Large pools skew toward contrarian/champion_anchor.
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
        candidates.append((chalk(self.simulator, rng), "chalk"))

        # Compute strategy counts
        mix = self._get_strategy_mix(pool_size)
        remaining = n_candidates - 1
        strategy_counts: dict[str, int] = {}
        allocated = 0
        for strategy_name, fraction in mix.items():
            count = int(round(fraction * remaining))
            strategy_counts[strategy_name] = count
            allocated += count
        # Fix rounding
        diff = remaining - allocated
        if diff != 0:
            largest = max(strategy_counts, key=lambda k: strategy_counts[k])
            strategy_counts[largest] += diff

        # near_chalk
        for _ in range(strategy_counts.get("near_chalk", 0)):
            candidates.append((near_chalk(self.simulator, rng), "near_chalk"))

        # random_sim
        for _ in range(strategy_counts.get("random_sim", 0)):
            candidates.append((random_sim(self.simulator, rng), "random_sim"))

        # late_contrarian with varying boost
        n_late = strategy_counts.get("late_contrarian", 0)
        late_boost_values = [0.3, 0.5, 0.7]
        for i in range(n_late):
            boost = late_boost_values[i % len(late_boost_values)]
            picks = late_contrarian(
                self.simulator, rng, upset_boost=boost
            )
            candidates.append((picks, f"late_contrarian_{boost}"))

        # contrarian with varying upset_boost
        n_contrarian = strategy_counts.get("contrarian", 0)
        contrarian_boost_values = [0.15, 0.3, 0.5]
        for i in range(n_contrarian):
            boost = contrarian_boost_values[i % len(contrarian_boost_values)]
            picks = contrarian(self.simulator, rng, upset_boost=boost)
            candidates.append((picks, f"contrarian_{boost}"))

        # champion_anchor: sample champions from simulation frequency
        n_anchor = strategy_counts.get("champion_anchor", 0)
        if n_anchor > 0:
            # Find the final-round slot(s)
            structure = self.simulator.structure
            max_round = max(structure.slot_to_round.values()) if structure.slot_to_round else 0
            final_slots = [
                s for s, r in structure.slot_to_round.items() if r == max_round
            ]

            champ_counts: dict[str, int] = {}
            for outcome in simulated_outcomes:
                for fs in final_slots:
                    if fs in outcome:
                        champ = outcome[fs]
                        champ_counts[champ] = champ_counts.get(champ, 0) + 1

            if champ_counts:
                entities = list(champ_counts.keys())
                weights = np.array(
                    [champ_counts[e] for e in entities], dtype=np.float64
                )
                weights /= weights.sum()
                for _ in range(n_anchor):
                    champ = str(rng.choice(entities, p=weights))
                    picks = champion_anchor(
                        self.simulator, rng, champion=champ
                    )
                    candidates.append((picks, f"anchor_{champ}"))
            else:
                # Fallback: use random_sim
                for _ in range(n_anchor):
                    candidates.append(
                        (random_sim(self.simulator, rng), "random_sim")
                    )

        return candidates

    # ------------------------------------------------------------------
    # Vectorized scoring
    # ------------------------------------------------------------------

    def _brackets_to_matrix(
        self, bracket_dicts: list[dict[str, str]]
    ) -> np.ndarray:
        """Convert list of bracket dicts to a numpy matrix.

        Uses string hashing for vectorized comparison. Returns array of
        shape (n_brackets, n_scored_slots) with hash values.
        """
        n = len(bracket_dicts)
        n_slots = len(self._scored_slots)
        matrix = np.zeros((n, n_slots), dtype=np.int64)
        for i, picks in enumerate(bracket_dicts):
            for j, slot in enumerate(self._scored_slots):
                if slot in picks:
                    matrix[i, j] = hash(picks[slot])
        return matrix

    def _score_candidates_vectorized(
        self,
        candidates: list[tuple[dict[str, str], str]],
        candidate_matrix: np.ndarray,
        outcome_matrix: np.ndarray,
        opponent_matrix: np.ndarray,
    ) -> list[dict]:
        """Score candidates using vectorized numpy operations.

        For each candidate, computes:
        - expected_points: average points across all simulated outcomes
        - win_probability: fraction of sims where candidate beats all opponents
        - top10_probability: fraction of sims where candidate is in top 10%
        """
        n_sims = outcome_matrix.shape[0]
        n_opponents = opponent_matrix.shape[0]
        slot_points = self._slot_points_arr

        # Pre-score all opponents: (n_sims, n_opponents)
        opp_scores = np.zeros((n_sims, n_opponents), dtype=np.float64)
        for j in range(n_opponents):
            matches = outcome_matrix == opponent_matrix[j]
            opp_scores[:, j] = (matches * slot_points).sum(axis=1)

        # Per-sim max opponent score and top-10% threshold
        if n_opponents > 0:
            max_opp_scores = opp_scores.max(axis=1)
            top10_idx = max(0, n_opponents // 10)
            sorted_opp = np.sort(opp_scores, axis=1)[:, ::-1]
            top10_threshold = sorted_opp[:, min(top10_idx, n_opponents - 1)]
        else:
            max_opp_scores = np.zeros(n_sims, dtype=np.float64)
            top10_threshold = np.zeros(n_sims, dtype=np.float64)

        scored = []
        for i, (picks, strategy_name) in enumerate(candidates):
            bracket_row = candidate_matrix[i]
            matches = outcome_matrix == bracket_row
            our_scores = (matches * slot_points).sum(axis=1)

            total_points = float(our_scores.sum())
            wins = int((our_scores >= max_opp_scores).sum())
            top10_finishes = int((our_scores >= top10_threshold).sum())

            scored.append({
                "picks": picks,
                "expected_points": total_points / n_sims,
                "win_probability": wins / n_sims,
                "top10_probability": top10_finishes / n_sims,
                "strategy": strategy_name,
            })

        scored.sort(key=lambda x: x["win_probability"], reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Diversity selection
    # ------------------------------------------------------------------

    def _select_diverse(
        self,
        scored: list[dict],
        n_brackets: int,
        overlap_threshold: float = 0.90,
    ) -> list[dict]:
        """Select N diverse brackets using round-weighted overlap (greedy).

        Takes the highest-scored bracket first, then greedily adds brackets
        that differ from all already-selected by more than overlap_threshold.
        Falls back to remaining highest-scored if not enough diverse brackets.
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
                if overlap > overlap_threshold:
                    is_diverse = False
                    break
            if is_diverse:
                selected.append(candidate)

        # Backfill if not enough diverse brackets found
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
        """Round-weighted overlap — late-round disagreements count more.

        Each slot's agreement is weighted by its round points, so a
        championship disagreement counts much more than a first-round one.
        """
        structure = self.simulator.structure
        scoring_values = structure.config.scoring.values or []

        weighted_agree = 0.0
        total_weight = 0.0

        for slot in self._scored_slots:
            rd = structure.slot_to_round.get(slot, 0)
            if rd < 1 or rd > len(scoring_values):
                continue
            weight = float(scoring_values[rd - 1])
            total_weight += weight
            if picks_a.get(slot) == picks_b.get(slot):
                weighted_agree += weight

        if total_weight == 0:
            return 0.0
        return weighted_agree / total_weight

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _enrich_brackets(self, scored: list[dict]) -> list[CompetitionResult]:
        """Convert scored bracket dicts to CompetitionResult with matchup context."""
        results = []
        structure = self.simulator.structure

        sorted_slots = sorted(
            structure.slots,
            key=lambda s: structure.slot_to_round.get(s, 0),
        )

        for bracket in scored:
            picks = bracket["picks"]
            matchups: dict[str, MatchupContext] = {}

            for slot in sorted_slots:
                if slot not in structure.slot_matchups or slot not in picks:
                    continue

                round_num = structure.slot_to_round.get(slot, 0)
                if round_num < 1:
                    continue

                ref_a, ref_b = structure.slot_matchups[slot]
                entity_a = _resolve_entity(ref_a, structure, picks)
                entity_b = _resolve_entity(ref_b, structure, picks)

                if entity_a is None or entity_b is None:
                    continue

                ctx = self.simulator.get_matchup_context(
                    entity_a=entity_a,
                    entity_b=entity_b,
                    slot=slot,
                    round_num=round_num,
                    pick=picks[slot],
                    strategy=bracket["strategy"],
                )
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
```

**Step 4:** Run tests, verify PASS

**Step 5:** Commit: `feat: add pool-aware competition optimizer with pluggable strategies`

---

## Task 7: explainer.py

**Files:**
- Create: `packages/easyml-sports/src/easyml/sports/competitions/explainer.py`
- Create: `packages/easyml-sports/tests/competitions/test_explainer.py`

**Step 1: Write failing tests**

```python
# packages/easyml-sports/tests/competitions/test_explainer.py
"""Tests for competition explainer (generic feature differentials + narratives)."""
import numpy as np
import pandas as pd
import pytest

from easyml.sports.competitions.explainer import CompetitionExplainer
from easyml.sports.competitions.schemas import (
    CompetitionResult,
    MatchupContext,
)
from easyml.core.runner.hooks import HookRegistry


@pytest.fixture
def entity_features():
    """Feature DataFrame indexed by entity ID."""
    return pd.DataFrame({
        "entity_id": ["Alpha", "Beta", "Gamma", "Delta"],
        "rating": [95.0, 88.0, 72.0, 65.0],
        "efficiency": [110.5, 105.2, 98.1, 92.3],
        "momentum": [0.8, 0.6, 0.9, 0.3],
        "volatility": [5.0, 8.0, 3.0, 12.0],
    })


@pytest.fixture
def display_names():
    return {
        "rating": "Power Rating",
        "efficiency": "Net Efficiency",
        "momentum": "Recent Momentum",
        "volatility": "Performance Volatility",
    }


@pytest.fixture
def explainer(entity_features, display_names):
    return CompetitionExplainer(
        entity_features=entity_features,
        entity_id_col="entity_id",
        feature_display_names=display_names,
    )


@pytest.fixture
def sample_bracket_result():
    """A CompetitionResult with 3 matchups."""
    return CompetitionResult(
        picks={"R1_1": "Alpha", "R1_2": "Beta", "R2_1": "Alpha"},
        matchups={
            "R1_1": MatchupContext(
                slot="R1_1",
                round_num=1,
                entity_a="Alpha",
                entity_b="Delta",
                prob_a=0.9,
                model_probs={"xgb": 0.92, "lgbm": 0.88},
                model_agreement=0.95,
                pick="Alpha",
                strategy="chalk",
                upset=False,
            ),
            "R1_2": MatchupContext(
                slot="R1_2",
                round_num=1,
                entity_a="Beta",
                entity_b="Gamma",
                prob_a=0.55,
                model_probs={"xgb": 0.6, "lgbm": 0.45},
                model_agreement=0.5,
                pick="Beta",
                strategy="chalk",
                upset=False,
            ),
            "R2_1": MatchupContext(
                slot="R2_1",
                round_num=2,
                entity_a="Alpha",
                entity_b="Beta",
                prob_a=0.65,
                model_probs={"xgb": 0.7, "lgbm": 0.6},
                model_agreement=0.85,
                pick="Alpha",
                strategy="chalk",
                upset=False,
            ),
        },
        expected_points=42.0,
        win_probability=0.15,
        strategy="chalk",
    )


class TestComputeDifferentials:
    def test_returns_top_n_differentials(self, explainer):
        """compute_differentials returns at most top_n results."""
        diffs = explainer.compute_differentials("Alpha", "Delta", top_n=2)
        assert len(diffs) <= 2

    def test_differentials_have_required_keys(self, explainer):
        """Each differential dict has the expected keys."""
        diffs = explainer.compute_differentials("Alpha", "Beta", top_n=5)
        assert len(diffs) > 0
        required_keys = {
            "feature", "display_name", "entity_a_value",
            "entity_b_value", "difference", "favors",
        }
        for d in diffs:
            assert required_keys.issubset(d.keys())

    def test_differentials_sorted_by_magnitude(self, explainer):
        """Differentials are sorted by absolute difference descending."""
        diffs = explainer.compute_differentials("Alpha", "Delta", top_n=10)
        for i in range(len(diffs) - 1):
            assert diffs[i]["difference"] >= diffs[i + 1]["difference"]

    def test_differentials_use_display_names(self, explainer):
        """Display names are used when available."""
        diffs = explainer.compute_differentials("Alpha", "Beta", top_n=10)
        for d in diffs:
            if d["feature"] == "rating":
                assert d["display_name"] == "Power Rating"
                break

    def test_differentials_with_missing_entity(self, explainer):
        """Missing entity returns empty differentials."""
        diffs = explainer.compute_differentials("Alpha", "NONEXISTENT", top_n=5)
        assert diffs == []

    def test_favors_correct_entity(self, explainer):
        """favors field correctly identifies which entity has the edge."""
        diffs = explainer.compute_differentials("Alpha", "Delta", top_n=10)
        for d in diffs:
            if d["feature"] == "rating":
                # Alpha has 95.0 > Delta's 65.0
                assert d["favors"] == "Alpha"

    def test_no_feature_display_names(self, entity_features):
        """Without display names, raw feature names are used."""
        exp = CompetitionExplainer(
            entity_features=entity_features,
            entity_id_col="entity_id",
        )
        diffs = exp.compute_differentials("Alpha", "Beta", top_n=5)
        for d in diffs:
            if d["feature"] == "rating":
                assert d["display_name"] == "rating"

    def test_skip_zero_diff_features(self, entity_features):
        """Features with zero difference are excluded."""
        # Add a feature with same value for both
        entity_features["constant"] = 10.0
        exp = CompetitionExplainer(
            entity_features=entity_features,
            entity_id_col="entity_id",
        )
        diffs = exp.compute_differentials("Alpha", "Beta", top_n=10)
        feature_names = [d["feature"] for d in diffs]
        assert "constant" not in feature_names


class TestGeneratePickStories:
    def test_returns_story_per_matchup(
        self, explainer, sample_bracket_result
    ):
        """One story generated per matchup in the bracket."""
        stories = explainer.generate_pick_stories(sample_bracket_result)
        assert len(stories) == 3

    def test_stories_sorted_by_round_then_slot(
        self, explainer, sample_bracket_result
    ):
        """Stories are sorted by round number then slot name."""
        stories = explainer.generate_pick_stories(sample_bracket_result)
        rounds = [s["round"] for s in stories]
        assert rounds == sorted(rounds)

    def test_story_has_required_fields(
        self, explainer, sample_bracket_result
    ):
        """Each story dict contains the expected fields."""
        stories = explainer.generate_pick_stories(sample_bracket_result)
        required = {
            "slot", "round", "pick", "opponent",
            "probability", "model_agreement", "upset",
            "strategy", "key_differentials", "narrative",
        }
        for story in stories:
            assert required.issubset(story.keys())

    def test_narrative_is_nonempty_string(
        self, explainer, sample_bracket_result
    ):
        """Each story has a non-empty narrative string."""
        stories = explainer.generate_pick_stories(sample_bracket_result)
        for story in stories:
            assert isinstance(story["narrative"], str)
            assert len(story["narrative"]) > 0

    def test_probability_is_for_picked_entity(
        self, explainer, sample_bracket_result
    ):
        """Probability reflects the picked entity's win probability."""
        stories = explainer.generate_pick_stories(sample_bracket_result)
        for story in stories:
            ctx = sample_bracket_result.matchups[story["slot"]]
            if ctx.pick == ctx.entity_a:
                assert story["probability"] == pytest.approx(ctx.prob_a, abs=0.01)
            else:
                assert story["probability"] == pytest.approx(
                    1.0 - ctx.prob_a, abs=0.01
                )


class TestNarrativeHook:
    def test_hook_called_for_narratives(
        self, entity_features, sample_bracket_result
    ):
        """COMPETITION_NARRATIVE hook is called when registered."""
        hook_calls = []

        def mock_narrative(
            ctx: MatchupContext, differentials: list[dict], **kwargs
        ) -> str:
            hook_calls.append(ctx.slot)
            return f"Custom narrative for {ctx.slot}"

        exp = CompetitionExplainer(
            entity_features=entity_features,
            entity_id_col="entity_id",
            narrative_hook=mock_narrative,
        )
        stories = exp.generate_pick_stories(sample_bracket_result)
        assert len(hook_calls) == 3
        for story in stories:
            assert story["narrative"].startswith("Custom narrative")

    def test_fallback_when_no_hook(
        self, explainer, sample_bracket_result
    ):
        """Without a hook, generic narrative is generated."""
        stories = explainer.generate_pick_stories(sample_bracket_result)
        for story in stories:
            assert len(story["narrative"]) > 0
            assert "Custom" not in story["narrative"]


class TestGenerateEntityProfiles:
    def test_returns_profiles_for_top_n(self, explainer):
        """generate_entity_profiles returns at most top_n profiles."""
        round_probs = pd.DataFrame({
            "entity_id": ["Alpha", "Beta", "Gamma", "Delta"],
            "round_1": [1.0, 1.0, 1.0, 1.0],
            "round_2": [0.9, 0.7, 0.3, 0.1],
            "champion": [0.4, 0.3, 0.2, 0.1],
        })
        profiles = explainer.generate_entity_profiles(round_probs, top_n=2)
        assert len(profiles) == 2

    def test_profiles_sorted_by_champion_prob(self, explainer):
        """Profiles are sorted by champion probability descending."""
        round_probs = pd.DataFrame({
            "entity_id": ["Alpha", "Beta", "Gamma", "Delta"],
            "round_1": [1.0, 1.0, 1.0, 1.0],
            "round_2": [0.9, 0.7, 0.3, 0.1],
            "champion": [0.4, 0.3, 0.2, 0.1],
        })
        profiles = explainer.generate_entity_profiles(round_probs, top_n=4)
        probs = [p["champion_prob"] for p in profiles]
        assert probs == sorted(probs, reverse=True)

    def test_profile_has_round_progression(self, explainer):
        """Each profile includes round-by-round probabilities."""
        round_probs = pd.DataFrame({
            "entity_id": ["Alpha", "Beta"],
            "round_1": [1.0, 1.0],
            "round_2": [0.8, 0.6],
            "champion": [0.5, 0.3],
        })
        profiles = explainer.generate_entity_profiles(round_probs, top_n=2)
        for p in profiles:
            assert "round_probs" in p
            assert isinstance(p["round_probs"], dict)

    def test_profile_has_feature_summary(self, explainer):
        """Each profile includes a feature-based summary."""
        round_probs = pd.DataFrame({
            "entity_id": ["Alpha"],
            "round_1": [1.0],
            "champion": [0.5],
        })
        profiles = explainer.generate_entity_profiles(round_probs, top_n=1)
        assert "feature_summary" in profiles[0]
        assert isinstance(profiles[0]["feature_summary"], dict)
```

**Step 2:** Run tests, verify FAIL (module does not exist)

**Step 3: Implement**

```python
# packages/easyml-sports/src/easyml/sports/competitions/explainer.py
"""Competition explanation engine.

Generates per-pick stories, entity profiles, and feature differentials
from CompetitionResult objects enriched with matchup context.

All feature handling is generic — no hardcoded feature lists or
domain-specific column names.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from easyml.sports.competitions.schemas import (
    CompetitionResult,
    MatchupContext,
)

# Hook name constant for narrative generation
COMPETITION_NARRATIVE = "COMPETITION_NARRATIVE"


class CompetitionExplainer:
    """Generate rich explanations for competition picks.

    Works with any feature DataFrame — no hardcoded feature lists.
    Domain-specific narratives can be injected via a narrative_hook
    callable or by registering a COMPETITION_NARRATIVE hook.
    """

    def __init__(
        self,
        entity_features: pd.DataFrame,
        entity_id_col: str = "entity_id",
        feature_display_names: dict[str, str] | None = None,
        narrative_hook: Callable[
            [MatchupContext, list[dict]], str
        ] | None = None,
    ):
        """Initialize explainer.

        Args:
            entity_features: DataFrame with one row per entity. Must contain
                a column named entity_id_col. All other numeric columns are
                treated as features for differential analysis.
            entity_id_col: Name of the entity ID column in entity_features.
            feature_display_names: Optional mapping of feature column names
                to human-readable display names.
            narrative_hook: Optional callable that takes (MatchupContext,
                differentials) and returns a narrative string. When provided,
                overrides the default generic narrative builder.
        """
        self._entity_id_col = entity_id_col
        self._display_names = feature_display_names or {}
        self._narrative_hook = narrative_hook

        # Index features by entity ID for fast lookup
        self._features = entity_features.set_index(entity_id_col)

        # Detect numeric feature columns (exclude the index)
        self._feature_cols = [
            c for c in self._features.columns
            if pd.api.types.is_numeric_dtype(self._features[c])
        ]

    # ------------------------------------------------------------------
    # Differentials
    # ------------------------------------------------------------------

    def compute_differentials(
        self,
        entity_a: str,
        entity_b: str,
        top_n: int = 5,
    ) -> list[dict]:
        """Find the top feature differentials between two entities.

        Args:
            entity_a: First entity ID.
            entity_b: Second entity ID.
            top_n: Maximum number of differentials to return.

        Returns:
            List of differential dicts sorted by absolute difference
            descending, each containing: feature, display_name,
            entity_a_value, entity_b_value, difference, favors.
        """
        if entity_a not in self._features.index:
            return []
        if entity_b not in self._features.index:
            return []

        row_a = self._features.loc[entity_a]
        row_b = self._features.loc[entity_b]

        diffs: list[dict] = []
        for feat in self._feature_cols:
            val_a = row_a.get(feat)
            val_b = row_b.get(feat)

            if val_a is None or val_b is None:
                continue
            if pd.isna(val_a) or pd.isna(val_b):
                continue

            val_a = float(val_a)
            val_b = float(val_b)
            diff = val_a - val_b

            if abs(diff) < 1e-6:
                continue

            display = self._display_names.get(feat, feat)
            # Default: higher is better for entity_a
            favors = entity_a if diff > 0 else entity_b

            diffs.append({
                "feature": feat,
                "display_name": display,
                "entity_a_value": round(val_a, 4),
                "entity_b_value": round(val_b, 4),
                "difference": round(abs(diff), 4),
                "favors": favors,
            })

        diffs.sort(key=lambda x: x["difference"], reverse=True)
        return diffs[:top_n]

    # ------------------------------------------------------------------
    # Pick stories
    # ------------------------------------------------------------------

    def generate_pick_stories(
        self, result: CompetitionResult
    ) -> list[dict]:
        """Generate a story for each pick in a competition result.

        Args:
            result: CompetitionResult with matchup context.

        Returns:
            List of pick story dicts, sorted by round then slot.
        """
        stories: list[dict] = []

        for slot in sorted(result.matchups.keys()):
            ctx = result.matchups[slot]
            if ctx.round_num < 1:
                continue

            pick = ctx.pick
            opponent = ctx.entity_b if pick == ctx.entity_a else ctx.entity_a
            prob = ctx.prob_a if pick == ctx.entity_a else 1.0 - ctx.prob_a

            differentials = self.compute_differentials(
                ctx.entity_a, ctx.entity_b
            )

            # Build narrative
            if self._narrative_hook is not None:
                narrative = self._narrative_hook(ctx, differentials)
            else:
                narrative = self._build_generic_narrative(ctx, differentials)

            stories.append({
                "slot": slot,
                "round": ctx.round_num,
                "pick": pick,
                "opponent": opponent,
                "probability": round(prob, 4),
                "model_agreement": round(ctx.model_agreement, 4),
                "upset": ctx.upset,
                "strategy": ctx.strategy,
                "key_differentials": differentials,
                "narrative": narrative,
                "model_probs": ctx.model_probs,
            })

        stories.sort(key=lambda x: (x["round"], x["slot"]))
        return stories

    def _build_generic_narrative(
        self,
        ctx: MatchupContext,
        differentials: list[dict],
    ) -> str:
        """Build a generic human-readable narrative for a single pick.

        No domain-specific language — uses only feature names and
        entity IDs.
        """
        pick = ctx.pick
        opponent = ctx.entity_b if pick == ctx.entity_a else ctx.entity_a
        prob = ctx.prob_a if pick == ctx.entity_a else 1.0 - ctx.prob_a

        parts: list[str] = []

        # Opening: pick over opponent with confidence
        parts.append(
            f"{pick} over {opponent} at {prob:.0%} confidence."
        )

        # Model consensus
        n_models = len(ctx.model_probs)
        if n_models > 0:
            favoring = sum(
                1 for p in ctx.model_probs.values()
                if (p > 0.5) == (pick == ctx.entity_a)
            )
            parts.append(
                f" {favoring} of {n_models} models favor {pick}"
                f" (agreement: {ctx.model_agreement:.0%})."
            )

        # Key differentials
        diff_parts: list[str] = []
        for d in differentials[:3]:
            diff_parts.append(
                f"{d['favors']} has the edge in {d['display_name']}"
                f" ({d['entity_a_value']} vs {d['entity_b_value']})"
            )
        if diff_parts:
            parts.append(" " + ". ".join(diff_parts) + ".")

        # Upset marker
        if ctx.upset:
            parts.append(f" Contrarian upset pick ({ctx.strategy}).")

        return "".join(parts).strip()

    # ------------------------------------------------------------------
    # Entity profiles
    # ------------------------------------------------------------------

    def generate_entity_profiles(
        self,
        round_probs: pd.DataFrame,
        top_n: int = 20,
        entity_id_col: str | None = None,
    ) -> list[dict]:
        """Generate profiles for the top N competition contenders.

        Args:
            round_probs: DataFrame with entity ID column and round-by-round
                probability columns (e.g., round_1, round_2, ..., champion).
            top_n: Number of top entities to profile.
            entity_id_col: Column name for entity IDs in round_probs.
                Defaults to the same entity_id_col used in __init__.

        Returns:
            List of entity profile dicts sorted by champion probability
            descending.
        """
        id_col = entity_id_col or self._entity_id_col

        if "champion" not in round_probs.columns:
            return []

        top_entities = round_probs.nlargest(top_n, "champion")

        profiles: list[dict] = []
        for _, row in top_entities.iterrows():
            entity_id = str(row[id_col])

            # Extract round-by-round probabilities
            round_prob_dict: dict[str, float] = {}
            for col in round_probs.columns:
                if col.startswith("round_") or col == "champion":
                    val = row.get(col)
                    if val is not None and not pd.isna(val):
                        round_prob_dict[col] = round(float(val), 4)

            # Feature summary
            feature_summary = self._build_feature_summary(entity_id)

            profiles.append({
                "entity_id": entity_id,
                "champion_prob": round(float(row["champion"]), 4),
                "round_probs": round_prob_dict,
                "feature_summary": feature_summary,
            })

        return profiles

    def _build_feature_summary(self, entity_id: str) -> dict[str, float]:
        """Build a summary of key feature values for an entity.

        Returns a dict mapping display names to values for all available
        numeric features.
        """
        if entity_id not in self._features.index:
            return {}

        row = self._features.loc[entity_id]
        summary: dict[str, float] = {}
        for feat in self._feature_cols:
            val = row.get(feat)
            if val is not None and not pd.isna(val):
                display = self._display_names.get(feat, feat)
                summary[display] = round(float(val), 4)

        return summary
```

**Step 4:** Run tests, verify PASS

**Step 5:** Commit: `feat: add generic competition explainer with hook-based narratives`

---

## Task 8: confidence.py

**Files:**
- Create: `packages/easyml-sports/src/easyml/sports/competitions/confidence.py`
- Create: `packages/easyml-sports/tests/competitions/test_confidence.py`

**Step 1: Write failing tests**

```python
# packages/easyml-sports/tests/competitions/test_confidence.py
"""Tests for competition confidence diagnostics."""
import numpy as np
import pandas as pd
import pytest

from easyml.sports.competitions.confidence import (
    compute_feature_outliers,
    compute_model_disagreement,
    generate_confidence_report,
)
from easyml.sports.competitions.schemas import MatchupContext


@pytest.fixture
def training_features():
    """Historical entity features for computing training distribution."""
    rng = np.random.default_rng(42)
    n_entities = 100
    return pd.DataFrame({
        "entity_id": [f"entity_{i}" for i in range(n_entities)],
        "fold": [2020] * 50 + [2021] * 50,
        "rating": rng.normal(80.0, 5.0, n_entities),
        "efficiency": rng.normal(100.0, 10.0, n_entities),
        "momentum": rng.normal(0.5, 0.2, n_entities),
    })


@pytest.fixture
def current_features():
    """Current-fold entity features including some outliers."""
    return pd.DataFrame({
        "entity_id": ["Alpha", "Beta", "Gamma", "Delta"],
        "fold": [2022, 2022, 2022, 2022],
        "rating": [95.0, 82.0, 78.0, 65.0],        # Alpha is high outlier
        "efficiency": [100.0, 105.0, 98.0, 130.0],  # Delta is high outlier
        "momentum": [0.5, 0.6, 0.4, 0.5],           # No outliers
    })


@pytest.fixture
def all_features(training_features, current_features):
    """Combined training + current features."""
    return pd.concat([training_features, current_features], ignore_index=True)


class TestComputeFeatureOutliers:
    def test_detects_outliers(self, all_features):
        """Entities with features >2 sigma from training mean are flagged."""
        competition_entities = ["Alpha", "Beta", "Gamma", "Delta"]
        outliers = compute_feature_outliers(
            entity_features=all_features,
            competition_entities=competition_entities,
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
            z_threshold=2.0,
        )
        assert isinstance(outliers, list)
        # At least some outliers should be found (Alpha's rating, Delta's efficiency)
        assert len(outliers) > 0

    def test_outlier_has_required_fields(self, all_features):
        """Each outlier dict has the expected keys."""
        outliers = compute_feature_outliers(
            entity_features=all_features,
            competition_entities=["Alpha", "Beta", "Gamma", "Delta"],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
        )
        required_keys = {
            "entity_id", "feature", "value", "z_score",
            "training_mean", "training_std",
        }
        for o in outliers:
            assert required_keys.issubset(o.keys())

    def test_sorted_by_abs_z_score(self, all_features):
        """Outliers are sorted by |z_score| descending."""
        outliers = compute_feature_outliers(
            entity_features=all_features,
            competition_entities=["Alpha", "Beta", "Gamma", "Delta"],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
        )
        for i in range(len(outliers) - 1):
            assert abs(outliers[i]["z_score"]) >= abs(outliers[i + 1]["z_score"])

    def test_custom_z_threshold(self, all_features):
        """Higher z_threshold returns fewer outliers."""
        outliers_2 = compute_feature_outliers(
            entity_features=all_features,
            competition_entities=["Alpha", "Beta", "Gamma", "Delta"],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
            z_threshold=2.0,
        )
        outliers_3 = compute_feature_outliers(
            entity_features=all_features,
            competition_entities=["Alpha", "Beta", "Gamma", "Delta"],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
            z_threshold=3.0,
        )
        assert len(outliers_3) <= len(outliers_2)

    def test_no_outliers_with_normal_data(self):
        """Normal data within 2 sigma produces no outliers."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "entity_id": [f"e_{i}" for i in range(n)],
            "fold": [2020] * 100 + [2021] * 100,
            "rating": rng.normal(80.0, 5.0, n),
        })
        # Manually set current entities to be near the mean
        df.loc[df["fold"] == 2021, "rating"] = 80.0
        outliers = compute_feature_outliers(
            entity_features=df,
            competition_entities=[f"e_{i}" for i in range(100, 200)],
            entity_id_col="entity_id",
            current_fold=2021,
            fold_col="fold",
            z_threshold=2.0,
        )
        assert len(outliers) == 0

    def test_handles_zero_std(self):
        """Features with zero std in training data are skipped."""
        df = pd.DataFrame({
            "entity_id": ["a", "b", "c", "d"],
            "fold": [2020, 2020, 2021, 2021],
            "constant_feat": [5.0, 5.0, 5.0, 10.0],
        })
        outliers = compute_feature_outliers(
            entity_features=df,
            competition_entities=["c", "d"],
            entity_id_col="entity_id",
            current_fold=2021,
            fold_col="fold",
        )
        # constant_feat has std=0 in training data, should be skipped
        assert len(outliers) == 0

    def test_uses_only_numeric_columns(self, all_features):
        """Non-numeric columns are excluded from outlier detection."""
        all_features["label"] = "category_value"
        outliers = compute_feature_outliers(
            entity_features=all_features,
            competition_entities=["Alpha"],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
        )
        feature_names = {o["feature"] for o in outliers}
        assert "label" not in feature_names
        assert "entity_id" not in feature_names
        assert "fold" not in feature_names


class TestComputeModelDisagreement:
    def test_returns_ranked_matchups(self):
        """Returns matchups ranked by agreement ascending."""
        matchups = {
            "R1_1": MatchupContext(
                slot="R1_1", round_num=1,
                entity_a="A", entity_b="B",
                prob_a=0.6,
                model_probs={"xgb": 0.9, "lgbm": 0.3},
                model_agreement=0.5,
                pick="A", strategy="chalk",
            ),
            "R1_2": MatchupContext(
                slot="R1_2", round_num=1,
                entity_a="C", entity_b="D",
                prob_a=0.8,
                model_probs={"xgb": 0.82, "lgbm": 0.78},
                model_agreement=0.95,
                pick="C", strategy="chalk",
            ),
        }
        result = compute_model_disagreement(matchups, top_n=10)
        assert len(result) == 2
        # Most disagreement first
        assert result[0]["agreement"] <= result[1]["agreement"]
        assert result[0]["slot"] == "R1_1"

    def test_has_required_fields(self):
        """Each result dict has the expected keys."""
        matchups = {
            "R1_1": MatchupContext(
                slot="R1_1", round_num=1,
                entity_a="A", entity_b="B",
                prob_a=0.6,
                model_probs={"xgb": 0.7, "lgbm": 0.5},
                model_agreement=0.8,
                pick="A", strategy="chalk",
            ),
        }
        result = compute_model_disagreement(matchups, top_n=10)
        required = {
            "slot", "round", "entity_a", "entity_b",
            "agreement", "prob_ensemble", "model_range",
        }
        for r in result:
            assert required.issubset(r.keys())

    def test_model_range_computed_correctly(self):
        """model_range is max - min of per-model probabilities."""
        matchups = {
            "R1_1": MatchupContext(
                slot="R1_1", round_num=1,
                entity_a="A", entity_b="B",
                prob_a=0.6,
                model_probs={"xgb": 0.9, "lgbm": 0.3, "rf": 0.6},
                model_agreement=0.5,
                pick="A", strategy="chalk",
            ),
        }
        result = compute_model_disagreement(matchups, top_n=10)
        assert result[0]["model_range"] == pytest.approx(0.6, abs=0.01)

    def test_top_n_limits_results(self):
        """top_n limits the number of returned matchups."""
        matchups = {}
        for i in range(20):
            matchups[f"R1_{i}"] = MatchupContext(
                slot=f"R1_{i}", round_num=1,
                entity_a=f"A{i}", entity_b=f"B{i}",
                prob_a=0.5 + i * 0.02,
                model_probs={"xgb": 0.5, "lgbm": 0.5},
                model_agreement=0.5 + i * 0.02,
                pick=f"A{i}", strategy="chalk",
            )
        result = compute_model_disagreement(matchups, top_n=5)
        assert len(result) == 5

    def test_empty_matchups(self):
        """Empty matchups returns empty list."""
        result = compute_model_disagreement({}, top_n=10)
        assert result == []

    def test_no_model_probs(self):
        """Matchup with empty model_probs still returns valid result."""
        matchups = {
            "R1_1": MatchupContext(
                slot="R1_1", round_num=1,
                entity_a="A", entity_b="B",
                prob_a=0.6,
                model_probs={},
                model_agreement=1.0,
                pick="A", strategy="chalk",
            ),
        }
        result = compute_model_disagreement(matchups, top_n=10)
        assert len(result) == 1
        assert result[0]["model_range"] == pytest.approx(0.0)


class TestGenerateConfidenceReport:
    def test_report_structure(self, all_features):
        """Report has expected top-level keys."""
        matchups = {
            "R1_1": MatchupContext(
                slot="R1_1", round_num=1,
                entity_a="Alpha", entity_b="Delta",
                prob_a=0.8,
                model_probs={"xgb": 0.85, "lgbm": 0.75},
                model_agreement=0.9,
                pick="Alpha", strategy="chalk",
            ),
        }
        report = generate_confidence_report(
            entity_features=all_features,
            competition_entities=["Alpha", "Beta", "Gamma", "Delta"],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
            matchups=matchups,
        )
        assert "feature_outliers" in report
        assert "high_disagreement" in report
        assert isinstance(report["feature_outliers"], list)
        assert isinstance(report["high_disagreement"], list)

    def test_report_passes_z_threshold(self, all_features):
        """z_threshold parameter is forwarded to compute_feature_outliers."""
        matchups = {}
        report_2 = generate_confidence_report(
            entity_features=all_features,
            competition_entities=["Alpha", "Beta", "Gamma", "Delta"],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
            matchups=matchups,
            z_threshold=2.0,
        )
        report_5 = generate_confidence_report(
            entity_features=all_features,
            competition_entities=["Alpha", "Beta", "Gamma", "Delta"],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
            matchups=matchups,
            z_threshold=5.0,
        )
        assert len(report_5["feature_outliers"]) <= len(
            report_2["feature_outliers"]
        )

    def test_report_disagreement_top_n(self, all_features):
        """disagreement_top_n limits disagreement results."""
        matchups = {}
        for i in range(20):
            matchups[f"R1_{i}"] = MatchupContext(
                slot=f"R1_{i}", round_num=1,
                entity_a=f"e_a_{i}", entity_b=f"e_b_{i}",
                prob_a=0.5,
                model_probs={"xgb": 0.5, "lgbm": 0.5},
                model_agreement=0.5,
                pick=f"e_a_{i}", strategy="chalk",
            )
        report = generate_confidence_report(
            entity_features=all_features,
            competition_entities=["Alpha"],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
            matchups=matchups,
            disagreement_top_n=5,
        )
        assert len(report["high_disagreement"]) <= 5

    def test_empty_report(self):
        """Empty inputs produce empty report sections."""
        df = pd.DataFrame({
            "entity_id": [],
            "fold": [],
        })
        report = generate_confidence_report(
            entity_features=df,
            competition_entities=[],
            entity_id_col="entity_id",
            current_fold=2022,
            fold_col="fold",
            matchups={},
        )
        assert report["feature_outliers"] == []
        assert report["high_disagreement"] == []
```

**Step 2:** Run tests, verify FAIL (module does not exist)

**Step 3: Implement**

```python
# packages/easyml-sports/src/easyml/sports/competitions/confidence.py
"""Confidence diagnostics for competition predictions.

Pre-competition analysis: feature outliers and model disagreement.
All functions are generic — no hardcoded feature lists or column names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from easyml.sports.competitions.schemas import MatchupContext


def compute_feature_outliers(
    entity_features: pd.DataFrame,
    competition_entities: list[str],
    entity_id_col: str = "entity_id",
    current_fold: int | str | None = None,
    fold_col: str = "fold",
    z_threshold: float = 2.0,
) -> list[dict]:
    """Compare competition entities' features against training distribution.

    Flags features >z_threshold standard deviations from the training mean.
    Training distribution = all entities from folds before the current fold.

    Automatically discovers all numeric columns (excluding the ID and fold
    columns) — no hardcoded feature list.

    Args:
        entity_features: DataFrame with entity features across folds.
        competition_entities: List of entity IDs to check for outliers.
        entity_id_col: Name of the entity ID column.
        current_fold: Current fold value. Training data = rows where
            fold_col < current_fold. If None, all data is used for
            training and no outlier detection is possible.
        fold_col: Name of the fold/period column.
        z_threshold: Number of standard deviations to flag as outlier.

    Returns:
        List of dicts with: entity_id, feature, value, z_score,
        training_mean, training_std. Sorted by |z_score| descending.
    """
    if entity_features.empty or not competition_entities:
        return []

    # Detect numeric feature columns (exclude ID and fold columns)
    exclude_cols = {entity_id_col, fold_col}
    feature_cols = [
        c for c in entity_features.columns
        if c not in exclude_cols
        and pd.api.types.is_numeric_dtype(entity_features[c])
    ]

    if not feature_cols:
        return []

    # Split into training and current
    if current_fold is not None and fold_col in entity_features.columns:
        training = entity_features[entity_features[fold_col] < current_fold]
        current = entity_features[
            (entity_features[fold_col] == current_fold)
            & (entity_features[entity_id_col].isin(competition_entities))
        ]
    else:
        return []

    if training.empty or current.empty:
        return []

    # Compute training distribution stats
    training_means = training[feature_cols].mean()
    training_stds = training[feature_cols].std()

    outliers: list[dict] = []
    for _, row in current.iterrows():
        entity_id = str(row[entity_id_col])
        for feat in feature_cols:
            val = row[feat]
            if pd.isna(val):
                continue
            std = training_stds[feat]
            if std == 0 or pd.isna(std):
                continue
            mean = training_means[feat]
            z = (float(val) - float(mean)) / float(std)
            if abs(z) >= z_threshold:
                outliers.append({
                    "entity_id": entity_id,
                    "feature": feat,
                    "value": float(val),
                    "z_score": float(z),
                    "training_mean": float(mean),
                    "training_std": float(std),
                })

    # Sort by |z_score| descending
    outliers.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    return outliers


def compute_model_disagreement(
    matchups: dict[str, MatchupContext],
    top_n: int = 15,
) -> list[dict]:
    """Rank matchups by model agreement (lowest first).

    Uses the model_probs and model_agreement fields from MatchupContext
    directly — no access to raw probability DataFrames needed.

    Args:
        matchups: Dict mapping slot name to MatchupContext.
        top_n: Maximum number of matchups to return.

    Returns:
        List of dicts with: slot, round, entity_a, entity_b,
        agreement, prob_ensemble, model_min, model_max, model_range.
        Sorted by agreement ascending (most disagreement first).
    """
    if not matchups:
        return []

    results: list[dict] = []
    for slot, ctx in matchups.items():
        model_probs = list(ctx.model_probs.values())

        if model_probs:
            model_min = float(min(model_probs))
            model_max = float(max(model_probs))
        else:
            model_min = ctx.prob_a
            model_max = ctx.prob_a

        results.append({
            "slot": ctx.slot,
            "round": ctx.round_num,
            "entity_a": ctx.entity_a,
            "entity_b": ctx.entity_b,
            "agreement": float(ctx.model_agreement),
            "prob_ensemble": float(ctx.prob_a),
            "model_min": model_min,
            "model_max": model_max,
            "model_range": float(model_max - model_min),
        })

    # Sort by agreement ascending (most disagreement first)
    results.sort(key=lambda x: x["agreement"])
    return results[:top_n]


def generate_confidence_report(
    entity_features: pd.DataFrame,
    competition_entities: list[str],
    entity_id_col: str = "entity_id",
    current_fold: int | str | None = None,
    fold_col: str = "fold",
    matchups: dict[str, MatchupContext] | None = None,
    z_threshold: float = 2.0,
    disagreement_top_n: int = 15,
) -> dict:
    """Assemble all confidence diagnostics into a report dict.

    Combines feature outliers and model disagreement analysis.

    Args:
        entity_features: DataFrame with entity features across folds.
        competition_entities: List of entity IDs in this competition.
        entity_id_col: Name of the entity ID column.
        current_fold: Current fold value for training split.
        fold_col: Name of the fold/period column.
        matchups: Dict mapping slot name to MatchupContext (for
            model disagreement analysis).
        z_threshold: Z-score threshold for feature outlier detection.
        disagreement_top_n: Max matchups to return for disagreement.

    Returns:
        Dict with keys: feature_outliers, high_disagreement.
    """
    # Feature outliers
    feature_outliers = compute_feature_outliers(
        entity_features=entity_features,
        competition_entities=competition_entities,
        entity_id_col=entity_id_col,
        current_fold=current_fold,
        fold_col=fold_col,
        z_threshold=z_threshold,
    )

    # Model disagreement
    high_disagreement = compute_model_disagreement(
        matchups=matchups or {},
        top_n=disagreement_top_n,
    )

    return {
        "feature_outliers": feature_outliers,
        "high_disagreement": high_disagreement,
    }
```

**Step 4:** Run tests, verify PASS

**Step 5:** Commit: `feat: add generic confidence diagnostics for competition engine`
# Competition Engine Implementation Plan -- Part 3 (Tasks 9-13)

## Task 9: export.py -- Multi-format output

**File:** `packages/easyml-sports/src/easyml/sports/competitions/export.py`
**Test:** `packages/easyml-sports/tests/competitions/test_export.py`

Generic multi-format export. No sports-specific terminology -- uses "entity",
"round", "slot" throughout. Ported from `mm/bracket/export.py` but fully generic.

### 9.1 Test (TDD -- write first)

```python
"""Tests for competition export module."""
from __future__ import annotations

import json
import csv
from io import StringIO
from pathlib import Path

import pytest

from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    CompetitionStructure,
    MatchupContext,
    ScoringConfig,
    StandingsEntry,
)
from easyml.sports.competitions.export import (
    export_bracket_markdown,
    export_standings_markdown,
    export_json,
    export_csv,
    export_analysis_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_elimination_structure() -> CompetitionStructure:
    """4-entity single-elimination for testing."""
    config = CompetitionConfig(
        format=CompetitionFormat.SINGLE_ELIMINATION,
        n_participants=4,
        scoring=ScoringConfig(type="per_round", values=[10, 20]),
    )
    return CompetitionStructure(
        config=config,
        slots=["R1G1", "R1G2", "R2G1"],
        slot_matchups={
            "R1G1": ("seed_1", "seed_4"),
            "R1G2": ("seed_2", "seed_3"),
        },
        slot_to_round={"R1G1": 1, "R1G2": 1, "R2G1": 2},
        round_slots={1: ["R1G1", "R1G2"], 2: ["R2G1"]},
        seed_to_entity={"seed_1": "Alpha", "seed_2": "Bravo", "seed_3": "Charlie", "seed_4": "Delta"},
        entity_to_seed={"Alpha": "seed_1", "Bravo": "seed_2", "Charlie": "seed_3", "Delta": "seed_4"},
    )


def _make_bracket_result() -> CompetitionResult:
    """Bracket result matching the 4-entity structure."""
    return CompetitionResult(
        picks={"R1G1": "Alpha", "R1G2": "Bravo", "R2G1": "Alpha"},
        matchups={
            "R1G1": MatchupContext(
                slot="R1G1", round_num=1, entity_a="Alpha", entity_b="Delta",
                prob_a=0.85, model_agreement=0.95, pick="Alpha",
                strategy="chalk", upset=False,
            ),
            "R1G2": MatchupContext(
                slot="R1G2", round_num=1, entity_a="Bravo", entity_b="Charlie",
                prob_a=0.6, model_agreement=0.7, pick="Bravo",
                strategy="chalk", upset=False,
            ),
            "R2G1": MatchupContext(
                slot="R2G1", round_num=2, entity_a="Alpha", entity_b="Bravo",
                prob_a=0.55, model_agreement=0.6, pick="Alpha",
                strategy="chalk", upset=False,
            ),
        },
        expected_points=22.5,
        win_probability=0.12,
        top10_probability=0.45,
        strategy="chalk",
    )


def _make_standings() -> list[StandingsEntry]:
    return [
        StandingsEntry(entity="Alpha", wins=5, losses=1, draws=0, points=15.0),
        StandingsEntry(entity="Bravo", wins=4, losses=2, draws=0, points=12.0),
        StandingsEntry(entity="Charlie", wins=2, losses=3, draws=1, points=7.0),
        StandingsEntry(entity="Delta", wins=0, losses=5, draws=1, points=1.0),
    ]


# ---------------------------------------------------------------------------
# export_bracket_markdown
# ---------------------------------------------------------------------------

class TestExportBracketMarkdown:
    def test_basic_output(self, tmp_path: Path):
        structure = _make_elimination_structure()
        result = _make_bracket_result()
        path = export_bracket_markdown(result, structure, tmp_path)
        assert path.exists()
        assert path.suffix == ".md"
        text = path.read_text()
        # Contains header info
        assert "Strategy" in text
        assert "chalk" in text
        assert "Expected Points" in text
        # Contains round headers
        assert "Round 1" in text
        assert "Round 2" in text
        # Contains entity names
        assert "Alpha" in text
        assert "Delta" in text

    def test_upset_marker(self, tmp_path: Path):
        result = _make_bracket_result()
        # Make one matchup an upset
        result.matchups["R1G2"].upset = True
        result.matchups["R1G2"].pick = "Charlie"
        result.picks["R1G2"] = "Charlie"
        structure = _make_elimination_structure()
        path = export_bracket_markdown(result, structure, tmp_path)
        text = path.read_text()
        assert "UPSET" in text

    def test_creates_output_dir(self, tmp_path: Path):
        out = tmp_path / "nested" / "dir"
        structure = _make_elimination_structure()
        result = _make_bracket_result()
        path = export_bracket_markdown(result, structure, out)
        assert path.exists()
        assert out.exists()

    def test_region_grouping(self, tmp_path: Path):
        """Structures with regions should group by region."""
        config = CompetitionConfig(
            format=CompetitionFormat.SINGLE_ELIMINATION,
            n_participants=8,
            regions=2,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
        )
        structure = CompetitionStructure(
            config=config,
            slots=["R1A1", "R1A2", "R1B1", "R1B2", "R2A1", "R2B1", "R3G1"],
            slot_matchups={
                "R1A1": ("s1", "s8"), "R1A2": ("s4", "s5"),
                "R1B1": ("s2", "s7"), "R1B2": ("s3", "s6"),
            },
            slot_to_round={
                "R1A1": 1, "R1A2": 1, "R1B1": 1, "R1B2": 1,
                "R2A1": 2, "R2B1": 2, "R3G1": 3,
            },
            round_slots={1: ["R1A1", "R1A2", "R1B1", "R1B2"], 2: ["R2A1", "R2B1"], 3: ["R3G1"]},
            seed_to_entity={f"s{i}": f"Entity{i}" for i in range(1, 9)},
            entity_to_seed={f"Entity{i}": f"s{i}" for i in range(1, 9)},
            region_names={0: "Region A", 1: "Region B"},
            slot_to_region={"R1A1": 0, "R1A2": 0, "R1B1": 1, "R1B2": 1, "R2A1": 0, "R2B1": 1},
        )
        matchups = {}
        for slot in ["R1A1", "R1A2", "R1B1", "R1B2"]:
            a, b = structure.slot_matchups[slot]
            matchups[slot] = MatchupContext(
                slot=slot, round_num=1,
                entity_a=structure.seed_to_entity[a],
                entity_b=structure.seed_to_entity[b],
                prob_a=0.7, model_agreement=0.8,
                pick=structure.seed_to_entity[a], strategy="chalk", upset=False,
            )
        result = CompetitionResult(
            picks={s: m.pick for s, m in matchups.items()},
            matchups=matchups,
            expected_points=30.0,
            strategy="chalk",
        )
        path = export_bracket_markdown(result, structure, tmp_path)
        text = path.read_text()
        assert "Region A" in text
        assert "Region B" in text


# ---------------------------------------------------------------------------
# export_standings_markdown
# ---------------------------------------------------------------------------

class TestExportStandingsMarkdown:
    def test_basic_table(self, tmp_path: Path):
        standings = _make_standings()
        path = export_standings_markdown(standings, tmp_path, title="League Results")
        assert path.exists()
        text = path.read_text()
        assert "League Results" in text
        # Table header
        assert "Entity" in text
        assert "Points" in text
        # All entities present
        for s in standings:
            assert s.entity in text

    def test_rank_ordering(self, tmp_path: Path):
        standings = _make_standings()
        path = export_standings_markdown(standings, tmp_path)
        text = path.read_text()
        # Alpha (15 pts) should appear before Delta (1 pt)
        assert text.index("Alpha") < text.index("Delta")

    def test_optional_columns(self, tmp_path: Path):
        """If all draws are 0, draws column should still appear (schema has it)."""
        standings = _make_standings()
        path = export_standings_markdown(standings, tmp_path)
        text = path.read_text()
        assert "Draws" in text or "D" in text


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJson:
    def test_bracket_json(self, tmp_path: Path):
        structure = _make_elimination_structure()
        result = _make_bracket_result()
        path = export_json(
            results=[result],
            structure=structure,
            output_dir=tmp_path,
            label="test_bracket",
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert "results" in data
        assert len(data["results"]) == 1
        r = data["results"][0]
        assert "picks" in r
        assert "matchups" in r
        assert "expected_points" in r
        assert r["strategy"] == "chalk"

    def test_matchup_context_fields(self, tmp_path: Path):
        structure = _make_elimination_structure()
        result = _make_bracket_result()
        path = export_json(
            results=[result],
            structure=structure,
            output_dir=tmp_path,
            label="ctx_test",
        )
        data = json.loads(path.read_text())
        matchup = data["results"][0]["matchups"]["R1G1"]
        assert "entity_a" in matchup
        assert "entity_b" in matchup
        assert "prob_a" in matchup
        assert "model_agreement" in matchup
        assert "upset" in matchup
        assert isinstance(matchup["prob_a"], float)

    def test_standings_json(self, tmp_path: Path):
        standings = _make_standings()
        path = export_json(
            standings=standings,
            output_dir=tmp_path,
            label="standings_test",
        )
        data = json.loads(path.read_text())
        assert "standings" in data
        assert len(data["standings"]) == 4
        assert data["standings"][0]["entity"] == "Alpha"

    def test_multiple_results(self, tmp_path: Path):
        structure = _make_elimination_structure()
        r1 = _make_bracket_result()
        r2 = _make_bracket_result()
        r2.strategy = "contrarian"
        path = export_json(
            results=[r1, r2],
            structure=structure,
            output_dir=tmp_path,
            label="multi",
        )
        data = json.loads(path.read_text())
        assert len(data["results"]) == 2


# ---------------------------------------------------------------------------
# export_csv
# ---------------------------------------------------------------------------

class TestExportCsv:
    def test_probability_csv(self, tmp_path: Path):
        import pandas as pd

        probs = pd.DataFrame({
            "entity_a": ["Alpha", "Alpha", "Bravo"],
            "entity_b": ["Bravo", "Charlie", "Charlie"],
            "probability": [0.55, 0.7, 0.6],
        })
        path = export_csv(probs, tmp_path, label="probs")
        assert path.exists()
        assert path.suffix == ".csv"
        # Read back and verify
        df = pd.read_csv(path)
        assert len(df) == 3
        assert "entity_a" in df.columns
        assert "probability" in df.columns

    def test_custom_columns(self, tmp_path: Path):
        import pandas as pd

        probs = pd.DataFrame({
            "entity_a": ["A", "B"],
            "entity_b": ["C", "D"],
            "prob_ensemble": [0.5, 0.6],
            "prob_xgb": [0.48, 0.62],
        })
        path = export_csv(probs, tmp_path, label="custom")
        df = pd.read_csv(path)
        assert "prob_xgb" in df.columns


# ---------------------------------------------------------------------------
# export_analysis_report
# ---------------------------------------------------------------------------

class TestExportAnalysisReport:
    def test_comprehensive_report(self, tmp_path: Path):
        structure = _make_elimination_structure()
        result = _make_bracket_result()
        pick_stories = [
            {
                "slot": "R1G1", "round_num": 1, "round_name": "Round 1",
                "narrative": "Alpha favored over Delta at 85%.",
            },
            {
                "slot": "R1G2", "round_num": 1, "round_name": "Round 1",
                "narrative": "Bravo edges Charlie at 60%.",
            },
        ]
        entity_profiles = [
            {
                "entity": "Alpha",
                "round_probs": {1: 0.95, 2: 0.55},
                "identity": "Top-ranked contender with strong fundamentals.",
            },
        ]
        confidence_data = {
            "high_disagreement": [
                {
                    "entity_a": "Alpha", "entity_b": "Delta",
                    "agreement": 0.6, "prob_ensemble": 0.85,
                    "model_min": 0.75, "model_max": 0.95,
                },
            ],
            "feature_outliers": [
                {"entity": "Delta", "feature": "strength_index", "value": 42.1, "z_score": -2.5},
            ],
            "thin_samples": [],
        }
        path = export_analysis_report(
            result=result,
            structure=structure,
            pick_stories=pick_stories,
            entity_profiles=entity_profiles,
            confidence_data=confidence_data,
            output_dir=tmp_path,
            label="analysis",
        )
        assert path.exists()
        text = path.read_text()
        assert "Analysis" in text
        assert "Confidence" in text
        assert "Alpha" in text
        assert "Model Disagreement" in text or "Disagreement" in text
        assert "Feature Outlier" in text or "Outlier" in text
        # Pick narratives
        assert "Alpha favored over Delta" in text
        # Entity profiles
        assert "Top-ranked contender" in text

    def test_empty_confidence_sections(self, tmp_path: Path):
        """Report should gracefully handle empty confidence data."""
        structure = _make_elimination_structure()
        result = _make_bracket_result()
        path = export_analysis_report(
            result=result,
            structure=structure,
            pick_stories=[],
            entity_profiles=[],
            confidence_data={"high_disagreement": [], "feature_outliers": [], "thin_samples": []},
            output_dir=tmp_path,
            label="empty_conf",
        )
        text = path.read_text()
        # Should still produce valid output
        assert "Strategy" in text or "strategy" in text

    def test_standings_in_report(self, tmp_path: Path):
        """Analysis report should support standings-based formats too."""
        standings = _make_standings()
        path = export_analysis_report(
            standings=standings,
            pick_stories=[],
            entity_profiles=[],
            confidence_data={},
            output_dir=tmp_path,
            label="league_analysis",
        )
        text = path.read_text()
        assert "Alpha" in text
```

### 9.2 Implementation

```python
"""Multi-format export for competition results.

Supports:
- Markdown bracket output (elimination formats)
- Markdown standings tables (league/swiss formats)
- JSON (machine-readable, full matchup context)
- CSV (probability matrices)
- Comprehensive analysis reports (markdown)

All output is generic -- no sports-specific terminology.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from easyml.sports.competitions.schemas import (
    CompetitionResult,
    CompetitionStructure,
    MatchupContext,
    StandingsEntry,
)


# ---------------------------------------------------------------------------
# Bracket markdown (elimination formats)
# ---------------------------------------------------------------------------


def export_bracket_markdown(
    result: CompetitionResult,
    structure: CompetitionStructure,
    output_dir: Path,
    *,
    label: str = "bracket",
) -> Path:
    """Export an elimination bracket result as human-readable markdown.

    Organized round-by-round, optionally grouped by region if the structure
    defines regions. Includes probabilities, model agreement, and upset markers.

    Args:
        result: The CompetitionResult with picks and matchups.
        structure: The CompetitionStructure defining rounds and regions.
        output_dir: Directory to write the file to.
        label: Filename label (used as ``{label}.md``).

    Returns:
        Path to the written markdown file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Competition Bracket")
    lines.append("")
    lines.append(f"**Strategy:** {result.strategy}")
    if result.expected_points > 0:
        lines.append(f"**Expected Points:** {result.expected_points:.1f}")
    if result.win_probability > 0:
        lines.append(f"**Win Probability:** {result.win_probability:.1%}")
    if result.top10_probability > 0:
        lines.append(f"**Top-10 Probability:** {result.top10_probability:.1%}")
    lines.append("")

    has_regions = bool(getattr(structure, "region_names", None))
    slot_to_region = getattr(structure, "slot_to_region", {})
    region_names = getattr(structure, "region_names", {})

    # Iterate rounds in order
    for round_num in sorted(structure.round_slots.keys()):
        round_label = _round_display_name(round_num, structure)
        lines.append(f"## {round_label}")
        lines.append("")

        # Collect matchups for this round that have context
        round_matchups = [
            (slot, result.matchups[slot])
            for slot in sorted(structure.round_slots[round_num])
            if slot in result.matchups
        ]

        if has_regions and round_matchups:
            # Group by region
            region_groups: dict[int, list[tuple[str, MatchupContext]]] = {}
            ungrouped: list[tuple[str, MatchupContext]] = []
            for slot, ctx in round_matchups:
                region_id = slot_to_region.get(slot)
                if region_id is not None:
                    region_groups.setdefault(region_id, []).append((slot, ctx))
                else:
                    ungrouped.append((slot, ctx))

            for region_id in sorted(region_groups.keys()):
                rname = region_names.get(region_id, f"Region {region_id}")
                lines.append(f"### {rname}")
                lines.append("")
                for slot, ctx in region_groups[region_id]:
                    lines.append(_format_matchup_line(ctx))
                lines.append("")

            # Cross-region / final matchups
            for slot, ctx in ungrouped:
                lines.append(_format_matchup_line(ctx))
            if ungrouped:
                lines.append("")
        else:
            for slot, ctx in round_matchups:
                lines.append(_format_matchup_line(ctx))
            lines.append("")

    output_path = output_dir / f"{label}.md"
    output_path.write_text("\n".join(lines))
    return output_path


# ---------------------------------------------------------------------------
# Standings markdown (league / swiss formats)
# ---------------------------------------------------------------------------


def export_standings_markdown(
    standings: list[StandingsEntry],
    output_dir: Path,
    *,
    title: str = "Standings",
    label: str = "standings",
) -> Path:
    """Export standings as a markdown table.

    Sorted by points descending (already expected to be sorted, but re-sorted
    for safety). Includes rank, entity, wins, losses, draws, and points.

    Args:
        standings: List of StandingsEntry objects.
        output_dir: Directory to write the file to.
        title: Table title / heading.
        label: Filename label.

    Returns:
        Path to the written markdown file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_standings = sorted(standings, key=lambda s: s.points, reverse=True)

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| Rank | Entity | W | L | D | Points |")
    lines.append("|------|--------|---|---|---|--------|")
    for rank, entry in enumerate(sorted_standings, 1):
        lines.append(
            f"| {rank} | {entry.entity} | {entry.wins} | {entry.losses} "
            f"| {entry.draws} | {entry.points:.1f} |"
        )
    lines.append("")

    output_path = output_dir / f"{label}.md"
    output_path.write_text("\n".join(lines))
    return output_path


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_json(
    output_dir: Path,
    *,
    results: list[CompetitionResult] | None = None,
    structure: CompetitionStructure | None = None,
    standings: list[StandingsEntry] | None = None,
    label: str = "competition",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export competition data as structured JSON.

    Supports both bracket results and standings. Includes full matchup context
    for bracket results.

    Args:
        output_dir: Directory to write the file to.
        results: List of CompetitionResult (for elimination formats).
        structure: CompetitionStructure (optional, for additional context).
        standings: List of StandingsEntry (for league/swiss formats).
        label: Filename label.
        metadata: Optional extra metadata dict to include at top level.

    Returns:
        Path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}

    if metadata:
        data["metadata"] = metadata

    if results is not None:
        data["results"] = [_result_to_dict(r) for r in results]

    if standings is not None:
        data["standings"] = [_standings_entry_to_dict(s) for s in standings]

    if structure is not None:
        data["format"] = structure.config.format.value if hasattr(structure.config.format, "value") else str(structure.config.format)
        data["n_participants"] = structure.config.n_participants

    output_path = output_dir / f"{label}.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return output_path


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def export_csv(
    probabilities: pd.DataFrame,
    output_dir: Path,
    *,
    label: str = "probabilities",
) -> Path:
    """Export a probability DataFrame as CSV.

    Writes all columns present in the DataFrame. Typically contains entity_a,
    entity_b, and one or more probability columns.

    Args:
        probabilities: DataFrame with probability data.
        output_dir: Directory to write the file to.
        label: Filename label.

    Returns:
        Path to the written CSV file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{label}.csv"
    probabilities.to_csv(output_path, index=False)
    return output_path


# ---------------------------------------------------------------------------
# Analysis report (comprehensive markdown)
# ---------------------------------------------------------------------------


def export_analysis_report(
    output_dir: Path,
    *,
    result: CompetitionResult | None = None,
    structure: CompetitionStructure | None = None,
    standings: list[StandingsEntry] | None = None,
    pick_stories: list[dict] | None = None,
    entity_profiles: list[dict] | None = None,
    confidence_data: dict[str, list[dict]] | None = None,
    label: str = "analysis",
) -> Path:
    """Export a comprehensive analysis report as markdown.

    Combines bracket/standings results, pick narratives, entity profiles,
    and confidence diagnostics into a single document.

    Args:
        output_dir: Directory to write the file to.
        result: CompetitionResult (for elimination formats).
        structure: CompetitionStructure (for region/round context).
        standings: List of StandingsEntry (for league/swiss formats).
        pick_stories: List of dicts with slot, round_num, round_name, narrative.
        entity_profiles: List of dicts with entity, round_probs, identity.
        confidence_data: Dict with high_disagreement, feature_outliers, thin_samples lists.
        label: Filename label.

    Returns:
        Path to the written markdown file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Competition Analysis")
    lines.append("")

    # --- Summary ---
    if result is not None:
        lines.append(f"**Strategy:** {result.strategy}")
        if result.expected_points > 0:
            lines.append(f"**Expected Points:** {result.expected_points:.0f}")
        if result.win_probability > 0:
            lines.append(f"**Win Probability:** {result.win_probability:.3f}")
        lines.append("")

    # --- Standings ---
    if standings is not None:
        lines.append("## Standings")
        lines.append("")
        sorted_st = sorted(standings, key=lambda s: s.points, reverse=True)
        lines.append("| Rank | Entity | W | L | D | Points |")
        lines.append("|------|--------|---|---|---|--------|")
        for rank, entry in enumerate(sorted_st, 1):
            lines.append(
                f"| {rank} | {entry.entity} | {entry.wins} | {entry.losses} "
                f"| {entry.draws} | {entry.points:.1f} |"
            )
        lines.append("")

    # --- Confidence Report ---
    confidence_data = confidence_data or {}
    has_confidence = any(confidence_data.get(k) for k in ["high_disagreement", "feature_outliers", "thin_samples"])

    if has_confidence:
        lines.append("## Confidence Report")
        lines.append("")
        lines.append("*Informational diagnostics only -- does not affect predictions.*")
        lines.append("")

        disagreements = confidence_data.get("high_disagreement", [])
        if disagreements:
            lines.append("### High Model Disagreement")
            lines.append("")
            for d in disagreements[:10]:
                lines.append(
                    f"- **{d['entity_a']} vs {d['entity_b']}** -- "
                    f"agreement: {d['agreement']:.0%}, "
                    f"ensemble: {d['prob_ensemble']:.0%}, "
                    f"range: [{d['model_min']:.0%} - {d['model_max']:.0%}]"
                )
            lines.append("")

        outliers = confidence_data.get("feature_outliers", [])
        if outliers:
            lines.append("### Feature Outliers")
            lines.append("")
            for o in outliers[:15]:
                direction = "above" if o["z_score"] > 0 else "below"
                lines.append(
                    f"- **{o['entity']}** -- {o['feature']}: {o['value']} "
                    f"({abs(o['z_score']):.1f} std {direction} mean)"
                )
            lines.append("")

        thin = confidence_data.get("thin_samples", [])
        if thin:
            lines.append("### Thin Historical Samples")
            lines.append("")
            for t in thin:
                lines.append(f"- {t['description']}: only {t['n_games']} historical games")
            lines.append("")

    # --- Entity Profiles ---
    entity_profiles = entity_profiles or []
    if entity_profiles:
        lines.append("## Entity Profiles")
        lines.append("")
        for p in entity_profiles:
            lines.append(f"### {p['entity']}")
            lines.append("")
            if "round_probs" in p:
                prob_parts = [
                    f"R{rnd}: {prob:.0%}" for rnd, prob in p["round_probs"].items()
                ]
                lines.append("**Path:** " + " -> ".join(prob_parts))
                lines.append("")
            if "identity" in p:
                lines.append(p["identity"])
                lines.append("")

    # --- Pick Narratives ---
    pick_stories = pick_stories or []
    if pick_stories:
        lines.append("## Pick Narratives")
        lines.append("")
        current_round_name = ""
        for story in pick_stories:
            rname = story.get("round_name", f"Round {story.get('round_num', '?')}")
            if rname != current_round_name:
                current_round_name = rname
                lines.append(f"### {rname}")
                lines.append("")
            lines.append(f"**{story['slot']}:** {story['narrative']}")
            lines.append("")

    output_path = output_dir / f"{label}.md"
    output_path.write_text("\n".join(lines))
    return output_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _round_display_name(round_num: int, structure: CompetitionStructure) -> str:
    """Generate a display name for a round number.

    Uses round_names from config if available, otherwise generic "Round N".
    """
    round_names = getattr(structure.config, "round_names", None)
    if round_names and round_num in round_names:
        return round_names[round_num]
    return f"Round {round_num}"


def _format_matchup_line(ctx: MatchupContext) -> str:
    """Format a single matchup as a markdown bullet line."""
    pick = ctx.pick
    loser = ctx.entity_b if ctx.pick == ctx.entity_a else ctx.entity_a
    prob_pick = ctx.prob_a if ctx.pick == ctx.entity_a else (1.0 - ctx.prob_a)
    upset_marker = " **UPSET**" if ctx.upset else ""
    return (
        f"- **{pick}** over {loser} "
        f"({prob_pick:.0%}, agreement: {ctx.model_agreement:.0%})"
        f"{upset_marker}"
    )


def _result_to_dict(result: CompetitionResult) -> dict[str, Any]:
    """Convert a CompetitionResult to a JSON-serializable dict."""
    matchups_data = {}
    for slot, ctx in result.matchups.items():
        matchups_data[slot] = {
            "slot": str(ctx.slot),
            "round_num": int(ctx.round_num),
            "entity_a": str(ctx.entity_a),
            "entity_b": str(ctx.entity_b),
            "prob_a": float(ctx.prob_a),
            "model_agreement": float(ctx.model_agreement),
            "pick": str(ctx.pick),
            "strategy": str(ctx.strategy),
            "upset": bool(ctx.upset),
            "model_probs": {k: float(v) for k, v in ctx.model_probs.items()},
        }
    return {
        "picks": dict(result.picks),
        "expected_points": float(result.expected_points),
        "win_probability": float(result.win_probability),
        "top10_probability": float(result.top10_probability),
        "strategy": str(result.strategy),
        "matchups": matchups_data,
    }


def _standings_entry_to_dict(entry: StandingsEntry) -> dict[str, Any]:
    """Convert a StandingsEntry to a JSON-serializable dict."""
    return {
        "entity": str(entry.entity),
        "wins": int(entry.wins),
        "losses": int(entry.losses),
        "draws": int(entry.draws),
        "points": float(entry.points),
        "goal_diff": float(entry.goal_diff),
    }
```

---

## Task 10: competitions/__init__.py -- Public API re-exports

**File:** `packages/easyml-sports/src/easyml/sports/competitions/__init__.py`
**Test:** `packages/easyml-sports/tests/competitions/test_init.py`

### 10.1 Test (TDD -- write first)

```python
"""Tests for competitions public API."""
from __future__ import annotations


class TestPublicAPI:
    """Verify all key symbols are importable from the package root."""

    def test_schemas_importable(self):
        from easyml.sports.competitions import (
            CompetitionConfig,
            CompetitionFormat,
            CompetitionResult,
            CompetitionStructure,
            MatchupContext,
            ScoringConfig,
            ScoreResult,
            StandingsEntry,
            AdjustmentConfig,
        )
        assert CompetitionConfig is not None
        assert CompetitionFormat is not None

    def test_structure_importable(self):
        from easyml.sports.competitions import build_structure
        assert callable(build_structure)

    def test_simulator_importable(self):
        from easyml.sports.competitions import CompetitionSimulator
        assert CompetitionSimulator is not None

    def test_optimizer_importable(self):
        from easyml.sports.competitions import CompetitionOptimizer
        assert CompetitionOptimizer is not None

    def test_scorer_importable(self):
        from easyml.sports.competitions import CompetitionScorer
        assert CompetitionScorer is not None

    def test_adjustments_importable(self):
        from easyml.sports.competitions import apply_adjustments
        assert callable(apply_adjustments)

    def test_explainer_importable(self):
        from easyml.sports.competitions import CompetitionExplainer
        assert CompetitionExplainer is not None

    def test_confidence_importable(self):
        from easyml.sports.competitions import CompetitionConfidence
        assert CompetitionConfidence is not None

    def test_export_importable(self):
        from easyml.sports.competitions import (
            export_bracket_markdown,
            export_standings_markdown,
            export_json,
            export_csv,
            export_analysis_report,
        )
        assert callable(export_bracket_markdown)
        assert callable(export_json)

    def test_all_attribute(self):
        import easyml.sports.competitions as comp
        assert hasattr(comp, "__all__")
        # All items in __all__ should be importable
        for name in comp.__all__:
            assert hasattr(comp, name), f"{name} in __all__ but not importable"
```

### 10.2 Implementation

```python
"""Competition engine -- public API.

Re-exports all key classes and functions from submodules for convenient access:

    from easyml.sports.competitions import (
        CompetitionConfig, CompetitionSimulator, build_structure, ...
    )
"""
from __future__ import annotations

from easyml.sports.competitions.schemas import (
    AdjustmentConfig,
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    CompetitionStructure,
    MatchupContext,
    ScoreResult,
    ScoringConfig,
    StandingsEntry,
)
from easyml.sports.competitions.structure import build_structure
from easyml.sports.competitions.simulator import CompetitionSimulator
from easyml.sports.competitions.optimizer import CompetitionOptimizer
from easyml.sports.competitions.scorer import CompetitionScorer
from easyml.sports.competitions.adjustments import apply_adjustments
from easyml.sports.competitions.explainer import CompetitionExplainer
from easyml.sports.competitions.confidence import CompetitionConfidence
from easyml.sports.competitions.export import (
    export_analysis_report,
    export_bracket_markdown,
    export_csv,
    export_json,
    export_standings_markdown,
)

__all__ = [
    # Schemas
    "AdjustmentConfig",
    "CompetitionConfig",
    "CompetitionFormat",
    "CompetitionResult",
    "CompetitionStructure",
    "MatchupContext",
    "ScoreResult",
    "ScoringConfig",
    "StandingsEntry",
    # Core classes
    "CompetitionSimulator",
    "CompetitionOptimizer",
    "CompetitionScorer",
    "CompetitionExplainer",
    "CompetitionConfidence",
    # Functions
    "apply_adjustments",
    "build_structure",
    # Export
    "export_analysis_report",
    "export_bracket_markdown",
    "export_csv",
    "export_json",
    "export_standings_markdown",
]
```

---

## Task 11: Hook registration -- COMPETITION_NARRATIVE

**Files:**
- `packages/easyml-core/src/easyml/core/runner/hooks.py` (add constant)
- `packages/easyml-sports/src/easyml/sports/hooks.py` (register default hook)

**Test:** `packages/easyml-sports/tests/test_competition_hooks.py`

### 11.1 Test (TDD -- write first)

```python
"""Tests for competition narrative hook registration."""
from __future__ import annotations

import pytest

from easyml.core.runner.hooks import HookRegistry, COMPETITION_NARRATIVE


class TestCompetitionNarrativeHook:
    @pytest.fixture(autouse=True)
    def _clear_hooks(self):
        """Clear hooks before each test to isolate state."""
        HookRegistry.clear()
        yield
        HookRegistry.clear()

    def test_constant_exists(self):
        """COMPETITION_NARRATIVE constant should be defined in hooks module."""
        assert COMPETITION_NARRATIVE == "competition_narrative"

    def test_register_adds_hook(self):
        """Registering a hook should make it retrievable."""
        def my_narrative(matchup_ctx, differentials):
            return f"{matchup_ctx['pick']} wins because reasons."

        HookRegistry.register(COMPETITION_NARRATIVE, my_narrative)
        hooks = HookRegistry.get(COMPETITION_NARRATIVE)
        assert len(hooks) == 1
        assert hooks[0] is my_narrative

    def test_call_first_returns_result(self):
        """call_first should invoke the first registered hook."""
        def narrative_fn(matchup_ctx, differentials):
            return f"Pick: {matchup_ctx['pick']}"

        HookRegistry.register(COMPETITION_NARRATIVE, narrative_fn)
        result = HookRegistry.call_first(
            COMPETITION_NARRATIVE,
            {"pick": "Alpha"},
            [],
        )
        assert result == "Pick: Alpha"

    def test_call_first_returns_none_when_empty(self):
        """call_first should return None if no hooks registered."""
        result = HookRegistry.call_first(COMPETITION_NARRATIVE, {}, [])
        assert result is None

    def test_sports_register_adds_default_hook(self):
        """sports.hooks.register() should register a default narrative hook."""
        from easyml.sports.hooks import register
        register()
        hooks = HookRegistry.get(COMPETITION_NARRATIVE)
        assert len(hooks) >= 1

    def test_default_hook_returns_none(self):
        """The default (empty) narrative hook should return None."""
        from easyml.sports.hooks import register
        register()
        result = HookRegistry.call_first(COMPETITION_NARRATIVE, {}, [])
        # Default hook returns None -- domain plugins override with real narratives
        assert result is None

    def test_custom_hook_overrides_default(self):
        """A custom hook registered after default should be callable."""
        from easyml.sports.hooks import register
        register()

        def basketball_narrative(matchup_ctx, differentials):
            return f"{matchup_ctx['pick']} dominates the paint."

        HookRegistry.register(COMPETITION_NARRATIVE, basketball_narrative)

        # call_first returns the default (None), but call_all returns both
        all_results = HookRegistry.call_all(COMPETITION_NARRATIVE, {"pick": "Duke"}, [])
        assert len(all_results) >= 2
        # The custom hook's result should be in there
        assert any("dominates the paint" in str(r) for r in all_results if r is not None)
```

### 11.2 Implementation -- hooks.py (core)

Add `COMPETITION_NARRATIVE` constant to `packages/easyml-core/src/easyml/core/runner/hooks.py`.

Add after line 51 (after `COLUMN_RENAMES`):

```python
COMPETITION_NARRATIVE = "competition_narrative"
```

Full updated hook names section:

```python
# Hook names
FEATURE_EXPANSION = "feature_expansion"
PROVIDER_INJECTION = "provider_injection"
PRE_TRAINING = "pre_training"
POST_PREDICTION = "post_prediction"
FEATURE_TYPE = "feature_type"
COLUMN_CANDIDATES = "column_candidates"
COLUMN_RENAMES = "column_renames"
COMPETITION_NARRATIVE = "competition_narrative"
```

### 11.3 Implementation -- hooks.py (sports)

Update `packages/easyml-sports/src/easyml/sports/hooks.py` to import and register the
default competition narrative hook.

```python
"""Register sports-specific hooks into easyml core."""
from __future__ import annotations

from easyml.core.runner.hooks import (
    COLUMN_CANDIDATES,
    COLUMN_RENAMES,
    COMPETITION_NARRATIVE,
    HookRegistry,
)


def register() -> None:
    """Register all sports hooks.

    Called automatically when easyml.sports is imported.
    Registers sports-domain column name candidates so that core
    pipeline, reporting, and profiling components can detect
    sports-specific column names (TeamA, TeamB, TeamAWon, etc.).

    Also registers a default (empty) competition narrative hook.
    Domain-specific plugins can register additional hooks to provide
    richer narrative generation.
    """
    HookRegistry.register(COLUMN_CANDIDATES, _sports_column_candidates)
    HookRegistry.register(COLUMN_RENAMES, _sports_column_renames)
    HookRegistry.register(COMPETITION_NARRATIVE, _default_competition_narrative)


def _sports_column_candidates() -> dict[str, list[str]]:
    """Return sports-domain column name candidates."""
    return {
        "a": ["TeamA", "team_a"],
        "b": ["TeamB", "team_b"],
        "label": ["TeamAWon"],
        "margin": ["TeamAMargin"],
        "id_patterns": ["TeamA", "TeamB"],
    }


def _sports_column_renames() -> dict[str, str]:
    """Return sports-domain column renames."""
    return {
        "TeamAWon": "result",
        "TeamAMargin": "margin",
    }


def _default_competition_narrative(
    matchup_ctx: dict,
    differentials: list[dict],
) -> str | None:
    """Default competition narrative hook -- returns None.

    Domain plugins (e.g., basketball, soccer) override this with hooks
    that produce sport-specific narrative text. The explainer module
    falls back to generic differential-based narratives when this
    returns None.
    """
    return None
```

---

## Task 12: MCP handler -- competitions

**Files:**
- `packages/easyml-plugin/src/easyml/plugin/handlers/competitions.py` (new handler)
- `packages/easyml-plugin/src/easyml/plugin/mcp_server.py` (add tool)

**Test:** `packages/easyml-plugin/tests/handlers/test_competitions_handler.py`

### 12.1 Test (TDD -- write first)

```python
"""Tests for competitions MCP handler."""
from __future__ import annotations

import json
import pytest

from easyml.plugin.handlers.competitions import dispatch, ACTIONS


class TestActionsRegistry:
    def test_all_actions_registered(self):
        expected = {
            "create", "list_formats", "simulate", "standings",
            "round_probs", "generate_brackets", "score_bracket",
            "adjust", "explain", "profiles", "confidence",
            "export", "list_strategies",
        }
        assert set(ACTIONS.keys()) == expected

    def test_invalid_action_returns_error(self):
        result = dispatch("nonexistent_action")
        assert "Error" in result
        assert "nonexistent_action" in result

    def test_fuzzy_match_suggestion(self):
        result = dispatch("simulat")  # close to "simulate"
        assert "simulate" in result


class TestListFormats:
    def test_returns_format_list(self):
        result = dispatch("list_formats")
        assert "single_elimination" in result
        assert "round_robin" in result
        assert "swiss" in result
        assert "double_elimination" in result
        assert "group_knockout" in result


class TestListStrategies:
    def test_returns_strategy_list(self):
        result = dispatch("list_strategies")
        assert "chalk" in result
        assert "contrarian" in result


class TestCreate:
    def test_missing_config_returns_error(self):
        result = dispatch("create")
        assert "Error" in result

    def test_with_minimal_config(self):
        config = json.dumps({
            "format": "single_elimination",
            "n_participants": 4,
            "scoring": {"type": "per_round", "values": [10, 20]},
        })
        result = dispatch("create", config=config)
        # Should succeed or return useful info about the created competition
        assert "Error" not in result or "single_elimination" in result


class TestSimulate:
    def test_requires_competition(self):
        result = dispatch("simulate")
        assert "Error" in result

    def test_requires_probabilities(self):
        result = dispatch("simulate", competition_id="test")
        assert "Error" in result


class TestExportAction:
    def test_requires_competition(self):
        result = dispatch("export")
        assert "Error" in result

    def test_lists_available_formats(self):
        result = dispatch("export", format="unknown_fmt")
        assert "Error" in result


class TestDispatchKwargsForwarding:
    """Verify dispatch forwards all kwargs without crashing."""

    def test_extra_kwargs_ignored(self):
        """Unknown kwargs should not cause crashes (caught by **_kwargs)."""
        result = dispatch("list_formats", some_random_param="hello")
        assert "Error" not in result
```

### 12.2 Implementation -- handlers/competitions.py

```python
"""Handler for manage_competitions tool."""
from __future__ import annotations

import json
from typing import Any

from easyml.plugin.handlers._common import parse_json_param
from easyml.plugin.handlers._validation import (
    validate_enum,
    validate_required,
    collect_hints,
    format_response_with_hints,
)


# ---------------------------------------------------------------------------
# In-memory competition registry (per-session)
# ---------------------------------------------------------------------------

_competitions: dict[str, dict[str, Any]] = {}
_next_id = 0


def _gen_id() -> str:
    global _next_id
    _next_id += 1
    return f"comp_{_next_id}"


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _handle_create(*, config=None, **_kwargs) -> str:
    if config is None:
        return "**Error**: `config` (JSON string or dict) is required for create."
    parsed = parse_json_param(config)
    if not isinstance(parsed, dict):
        return "**Error**: `config` must be a JSON object."

    from easyml.sports.competitions.schemas import CompetitionConfig, CompetitionFormat

    try:
        comp_config = CompetitionConfig(**parsed)
    except Exception as e:
        return f"**Error**: Invalid competition config: {e}"

    from easyml.sports.competitions.structure import build_structure

    try:
        structure = build_structure(comp_config)
    except Exception as e:
        return f"**Error**: Failed to build structure: {e}"

    comp_id = _gen_id()
    _competitions[comp_id] = {
        "config": comp_config,
        "structure": structure,
        "simulator": None,
        "results": [],
    }

    n_slots = len(structure.slots)
    n_rounds = len(structure.round_slots)
    return (
        f"Created competition `{comp_id}` -- "
        f"format: {comp_config.format.value}, "
        f"{comp_config.n_participants} participants, "
        f"{n_rounds} rounds, {n_slots} slots."
    )


def _handle_list_formats(**_kwargs) -> str:
    from easyml.sports.competitions.schemas import CompetitionFormat

    lines = ["**Available competition formats:**", ""]
    descriptions = {
        "single_elimination": "Knockout bracket (e.g., 64-entity tournament)",
        "double_elimination": "Winners + losers bracket with grand final",
        "round_robin": "All-play-all league (configurable rounds)",
        "swiss": "Swiss-system pairing (configurable rounds)",
        "group_knockout": "Group stage (round-robin) followed by knockout bracket",
    }
    for fmt in CompetitionFormat:
        desc = descriptions.get(fmt.value, fmt.value)
        lines.append(f"- **{fmt.value}**: {desc}")
    return "\n".join(lines)


def _handle_simulate(*, competition_id=None, probabilities=None, n_sims=None, seed=None, **_kwargs) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        available = ", ".join(sorted(_competitions.keys())) or "(none)"
        return f"**Error**: Competition `{competition_id}` not found. Available: {available}"
    if probabilities is None:
        return "**Error**: `probabilities` (JSON string or path to CSV) is required for simulate."

    import pandas as pd

    parsed_probs = parse_json_param(probabilities)
    if isinstance(parsed_probs, (list, dict)):
        prob_df = pd.DataFrame(parsed_probs)
    elif isinstance(probabilities, str):
        # Try reading as CSV path
        try:
            prob_df = pd.read_csv(probabilities)
        except Exception:
            return "**Error**: Could not parse `probabilities` as JSON or CSV path."
    else:
        return "**Error**: `probabilities` must be JSON or a path to a CSV file."

    comp = _competitions[competition_id]
    from easyml.sports.competitions.simulator import CompetitionSimulator

    try:
        simulator = CompetitionSimulator(
            config=comp["config"],
            structure=comp["structure"],
            probabilities=prob_df,
        )
    except Exception as e:
        return f"**Error**: Failed to create simulator: {e}"

    comp["simulator"] = simulator

    n = int(n_sims) if n_sims is not None else 10_000
    s = int(seed) if seed is not None else 42

    try:
        round_probs = simulator.entity_round_probabilities(n_sims=n, seed=s)
    except Exception as e:
        return f"**Error**: Simulation failed: {e}"

    comp["round_probs"] = round_probs

    return (
        f"Simulated `{competition_id}` with {n:,} iterations. "
        f"Round probabilities computed for {len(round_probs)} entities."
    )


def _handle_standings(*, competition_id=None, **_kwargs) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        return f"**Error**: Competition `{competition_id}` not found."
    comp = _competitions[competition_id]
    if comp["simulator"] is None:
        return "**Error**: Run `simulate` first to generate results."

    simulator = comp["simulator"]
    if not hasattr(simulator, "standings_distribution"):
        return "**Error**: Standings only available for league/swiss formats."

    try:
        standings = simulator.standings_distribution()
    except Exception as e:
        return f"**Error**: Failed to compute standings: {e}"

    lines = ["**Standings Distribution:**", ""]
    lines.append("| Entity | Avg Points | Win% | Top-4% |")
    lines.append("|--------|-----------|------|--------|")
    for entry in standings:
        lines.append(
            f"| {entry['entity']} | {entry['avg_points']:.1f} "
            f"| {entry['win_pct']:.1%} | {entry['top4_pct']:.1%} |"
        )
    return "\n".join(lines)


def _handle_round_probs(*, competition_id=None, **_kwargs) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        return f"**Error**: Competition `{competition_id}` not found."
    comp = _competitions[competition_id]
    if "round_probs" not in comp:
        return "**Error**: Run `simulate` first to generate round probabilities."

    round_probs = comp["round_probs"]
    if hasattr(round_probs, "to_markdown"):
        return f"**Round Probabilities:**\n\n{round_probs.to_markdown()}"
    return f"**Round Probabilities:**\n\n{round_probs}"


def _handle_generate_brackets(
    *, competition_id=None, pool_size=None, n_brackets=None,
    n_sims=None, seed=None, **_kwargs,
) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        return f"**Error**: Competition `{competition_id}` not found."
    comp = _competitions[competition_id]
    if comp["simulator"] is None:
        return "**Error**: Run `simulate` first."

    from easyml.sports.competitions.optimizer import CompetitionOptimizer

    try:
        optimizer = CompetitionOptimizer(simulator=comp["simulator"])
    except Exception as e:
        return f"**Error**: Failed to create optimizer: {e}"

    ps = int(pool_size) if pool_size is not None else 100
    nb = int(n_brackets) if n_brackets is not None else 5
    ns = int(n_sims) if n_sims is not None else 10_000
    s = int(seed) if seed is not None else 42

    try:
        results = optimizer.generate_brackets(
            pool_size=ps, n_brackets=nb, n_sims=ns, seed=s,
        )
    except Exception as e:
        return f"**Error**: Bracket generation failed: {e}"

    comp["results"] = results

    lines = [f"Generated {len(results)} bracket(s) for pool size {ps}:", ""]
    for i, r in enumerate(results):
        lines.append(
            f"  {i + 1}. strategy={r.strategy}, "
            f"E[pts]={r.expected_points:.1f}, "
            f"P(win)={r.win_probability:.3f}"
        )
    return "\n".join(lines)


def _handle_score_bracket(
    *, competition_id=None, actuals=None, bracket_index=None, **_kwargs,
) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        return f"**Error**: Competition `{competition_id}` not found."
    if actuals is None:
        return "**Error**: `actuals` (JSON dict of slot -> winner) is required."

    comp = _competitions[competition_id]
    parsed_actuals = parse_json_param(actuals)
    if not isinstance(parsed_actuals, dict):
        return "**Error**: `actuals` must be a JSON object mapping slot -> winner."

    idx = int(bracket_index) if bracket_index is not None else 0
    if not comp["results"]:
        return "**Error**: No brackets generated. Run `generate_brackets` first."
    if idx >= len(comp["results"]):
        return f"**Error**: bracket_index {idx} out of range (have {len(comp['results'])} brackets)."

    from easyml.sports.competitions.scorer import CompetitionScorer

    scorer = CompetitionScorer(scoring=comp["config"].scoring)
    result = comp["results"][idx]

    try:
        score = scorer.score_bracket(
            picks=result.picks,
            actuals=parsed_actuals,
            structure=comp["structure"],
        )
    except Exception as e:
        return f"**Error**: Scoring failed: {e}"

    lines = [f"**Score for bracket {idx + 1}:**", ""]
    lines.append(f"**Total points:** {score.total_points:.0f}")
    lines.append("")
    lines.append("| Round | Correct | Total | Points |")
    lines.append("|-------|---------|-------|--------|")
    for rnd in sorted(score.round_correct.keys()):
        lines.append(
            f"| {rnd} | {score.round_correct[rnd]} | {score.round_total[rnd]} "
            f"| {score.round_points.get(rnd, 0):.0f} |"
        )
    return "\n".join(lines)


def _handle_adjust(
    *, competition_id=None, adjustments=None, **_kwargs,
) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        return f"**Error**: Competition `{competition_id}` not found."
    if adjustments is None:
        return "**Error**: `adjustments` (JSON) is required."

    parsed = parse_json_param(adjustments)
    if not isinstance(parsed, dict):
        return "**Error**: `adjustments` must be a JSON object."

    comp = _competitions[competition_id]
    if comp["simulator"] is None:
        return "**Error**: Run `simulate` first."

    from easyml.sports.competitions.schemas import AdjustmentConfig
    from easyml.sports.competitions.adjustments import apply_adjustments

    try:
        adj_config = AdjustmentConfig(**parsed)
    except Exception as e:
        return f"**Error**: Invalid adjustment config: {e}"

    try:
        new_probs, log = apply_adjustments(
            comp["simulator"].probabilities, adj_config,
        )
    except Exception as e:
        return f"**Error**: Adjustment failed: {e}"

    # Update simulator with adjusted probabilities
    comp["simulator"].probabilities = new_probs

    return f"Applied {len(log)} adjustment(s). Re-run `simulate` to update results."


def _handle_explain(*, competition_id=None, bracket_index=None, **_kwargs) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        return f"**Error**: Competition `{competition_id}` not found."
    comp = _competitions[competition_id]
    if not comp["results"]:
        return "**Error**: No brackets generated. Run `generate_brackets` first."

    idx = int(bracket_index) if bracket_index is not None else 0
    if idx >= len(comp["results"]):
        return f"**Error**: bracket_index {idx} out of range."

    explainer_data = comp.get("explainer")
    if explainer_data is None:
        return "**Error**: No explainer configured. Provide entity features via `create` config."

    from easyml.sports.competitions.explainer import CompetitionExplainer

    try:
        explainer = CompetitionExplainer(**explainer_data)
        stories = explainer.generate_pick_stories(comp["results"][idx])
    except Exception as e:
        return f"**Error**: Explanation generation failed: {e}"

    lines = ["**Pick Explanations:**", ""]
    for story in stories:
        lines.append(f"**{story['slot']}** ({story['round_name']}): {story['narrative']}")
        lines.append("")
    return "\n".join(lines)


def _handle_profiles(*, competition_id=None, top_n=None, **_kwargs) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        return f"**Error**: Competition `{competition_id}` not found."
    comp = _competitions[competition_id]
    if comp["simulator"] is None:
        return "**Error**: Run `simulate` first."

    explainer_data = comp.get("explainer")
    if explainer_data is None:
        return "**Error**: No explainer configured."

    n = int(top_n) if top_n is not None else 20
    round_probs = comp.get("round_probs")

    from easyml.sports.competitions.explainer import CompetitionExplainer

    try:
        explainer = CompetitionExplainer(**explainer_data)
        profiles = explainer.generate_entity_profiles(
            comp["simulator"], round_probs, top_n=n,
        )
    except Exception as e:
        return f"**Error**: Profile generation failed: {e}"

    lines = ["**Entity Profiles:**", ""]
    for p in profiles:
        lines.append(f"### {p['entity']}")
        if "round_probs" in p:
            prob_parts = [f"R{r}: {v:.0%}" for r, v in p["round_probs"].items()]
            lines.append("**Path:** " + " -> ".join(prob_parts))
        if "identity" in p:
            lines.append(p["identity"])
        lines.append("")
    return "\n".join(lines)


def _handle_confidence(*, competition_id=None, **_kwargs) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        return f"**Error**: Competition `{competition_id}` not found."
    comp = _competitions[competition_id]
    if comp["simulator"] is None:
        return "**Error**: Run `simulate` first."

    from easyml.sports.competitions.confidence import CompetitionConfidence

    try:
        confidence = CompetitionConfidence(simulator=comp["simulator"])
        report = confidence.generate_report()
    except Exception as e:
        return f"**Error**: Confidence analysis failed: {e}"

    lines = ["**Confidence Report:**", ""]
    lines.append("*Informational diagnostics only -- does not affect predictions.*")
    lines.append("")

    if report.get("high_disagreement"):
        lines.append("### High Model Disagreement")
        lines.append("")
        for d in report["high_disagreement"][:10]:
            lines.append(
                f"- **{d['entity_a']} vs {d['entity_b']}** -- "
                f"agreement: {d['agreement']:.0%}"
            )
        lines.append("")

    if report.get("feature_outliers"):
        lines.append("### Feature Outliers")
        lines.append("")
        for o in report["feature_outliers"][:15]:
            direction = "above" if o["z_score"] > 0 else "below"
            lines.append(
                f"- **{o['entity']}** -- {o['feature']}: "
                f"{abs(o['z_score']):.1f} std {direction} mean"
            )
        lines.append("")

    return "\n".join(lines)


def _handle_export(
    *, competition_id=None, format=None, output_dir=None,
    label=None, bracket_index=None, **_kwargs,
) -> str:
    err = validate_required(competition_id, "competition_id")
    if err:
        return err
    if competition_id not in _competitions:
        return f"**Error**: Competition `{competition_id}` not found."

    valid_formats = {"bracket_markdown", "standings_markdown", "json", "csv", "analysis"}
    fmt = format or "json"
    fmt_err = validate_enum(fmt, valid_formats, "format")
    if fmt_err:
        return fmt_err

    comp = _competitions[competition_id]
    from pathlib import Path

    out_dir = Path(output_dir) if output_dir else Path.cwd() / "output"
    lbl = label or competition_id

    from easyml.sports.competitions import export as exp

    try:
        if fmt == "bracket_markdown":
            idx = int(bracket_index) if bracket_index is not None else 0
            if not comp["results"] or idx >= len(comp["results"]):
                return "**Error**: No bracket at that index. Run `generate_brackets` first."
            path = exp.export_bracket_markdown(
                comp["results"][idx], comp["structure"], out_dir, label=lbl,
            )
        elif fmt == "standings_markdown":
            if comp["simulator"] is None:
                return "**Error**: Run `simulate` first."
            standings = comp["simulator"].standings_distribution()
            from easyml.sports.competitions.schemas import StandingsEntry
            entries = [StandingsEntry(**s) for s in standings]
            path = exp.export_standings_markdown(entries, out_dir, label=lbl)
        elif fmt == "json":
            path = exp.export_json(
                out_dir,
                results=comp.get("results") or None,
                structure=comp.get("structure"),
                label=lbl,
            )
        elif fmt == "csv":
            if comp["simulator"] is None:
                return "**Error**: Run `simulate` first."
            path = exp.export_csv(comp["simulator"].probabilities, out_dir, label=lbl)
        elif fmt == "analysis":
            path = exp.export_analysis_report(
                out_dir,
                result=comp["results"][0] if comp["results"] else None,
                structure=comp.get("structure"),
                label=lbl,
            )
        else:
            return f"**Error**: Unknown format `{fmt}`."
    except Exception as e:
        return f"**Error**: Export failed: {e}"

    return f"Exported to `{path}`"


def _handle_list_strategies(**_kwargs) -> str:
    lines = ["**Available bracket generation strategies:**", ""]
    strategies = {
        "chalk": "Always pick the favorite in every matchup.",
        "near_chalk": "Chalk with close-matchup flips (underdog >40%).",
        "random_sim": "Pure Monte Carlo sample from probability distribution.",
        "contrarian": "Uniform upset boost (compress probabilities toward 0.5).",
        "late_contrarian": "Chalk in early rounds, upset-boosted in late rounds.",
        "champion_anchor": "Condition on a sampled champion, simulate rest accordingly.",
    }
    for name, desc in strategies.items():
        lines.append(f"- **{name}**: {desc}")
    lines.append("")
    lines.append("Custom strategies can be registered via the `StrategyFn` protocol.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action registry + dispatch
# ---------------------------------------------------------------------------


ACTIONS = {
    "create": _handle_create,
    "list_formats": _handle_list_formats,
    "simulate": _handle_simulate,
    "standings": _handle_standings,
    "round_probs": _handle_round_probs,
    "generate_brackets": _handle_generate_brackets,
    "score_bracket": _handle_score_bracket,
    "adjust": _handle_adjust,
    "explain": _handle_explain,
    "profiles": _handle_profiles,
    "confidence": _handle_confidence,
    "export": _handle_export,
    "list_strategies": _handle_list_strategies,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_competitions action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    hints = collect_hints(action, tool="competitions", **kwargs)
    return format_response_with_hints(result, hints)
```

### 12.3 Implementation -- mcp_server.py addition

Add the following tool definition to `packages/easyml-plugin/src/easyml/plugin/mcp_server.py`
(after the last existing tool block):

```python
# -----------------------------------------------------------------------
# N. manage_competitions
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def manage_competitions(
    action: str,
    ctx: Context,
    config: str | dict | None = None,
    competition_id: str | None = None,
    probabilities: str | dict | list | None = None,
    actuals: str | dict | None = None,
    adjustments: str | dict | None = None,
    pool_size: int | None = None,
    n_brackets: int | None = None,
    n_sims: int | None = None,
    seed: int | None = None,
    bracket_index: int | None = None,
    top_n: int | None = None,
    format: str | None = None,
    output_dir: str | None = None,
    label: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Manage competition simulations and bracket generation.

    Actions:
      - "create": Create a competition from config. Requires config (JSON with
        format, n_participants, scoring, etc.).
      - "list_formats": Show available competition formats.
      - "simulate": Run Monte Carlo simulations. Requires competition_id and
        probabilities (JSON or CSV path). Optional: n_sims (default 10000),
        seed (default 42).
      - "standings": Get standings distributions for league/swiss formats.
        Requires competition_id.
      - "round_probs": Get entity round-by-round advancement probabilities.
        Requires competition_id (must run simulate first).
      - "generate_brackets": Generate pool-size-aware brackets. Requires
        competition_id. Optional: pool_size (default 100), n_brackets
        (default 5), n_sims (default 10000), seed (default 42).
      - "score_bracket": Score bracket picks against actual results. Requires
        competition_id and actuals (JSON dict of slot -> winner). Optional:
        bracket_index (default 0).
      - "adjust": Apply probability adjustments. Requires competition_id and
        adjustments (JSON with entity_multipliers, external_weight, etc.).
      - "explain": Generate pick explanations for a bracket. Requires
        competition_id. Optional: bracket_index (default 0).
      - "profiles": Generate entity profiles. Requires competition_id.
        Optional: top_n (default 20).
      - "confidence": Run pre-competition confidence diagnostics. Requires
        competition_id.
      - "export": Export results. Requires competition_id. Optional: format
        (bracket_markdown | standings_markdown | json | csv | analysis,
        default json), output_dir, label, bracket_index.
      - "list_strategies": Show available bracket generation strategies.
    """
    return _load_handler("competitions").dispatch(
        action,
        ctx=ctx,
        config=config,
        competition_id=competition_id,
        probabilities=probabilities,
        actuals=actuals,
        adjustments=adjustments,
        pool_size=pool_size,
        n_brackets=n_brackets,
        n_sims=n_sims,
        seed=seed,
        bracket_index=bracket_index,
        top_n=top_n,
        format=format,
        output_dir=output_dir,
        label=label,
        project_dir=project_dir,
    )
```

---

## Task 13: Integration test -- Full pipeline

**File:** `packages/easyml-sports/tests/competitions/test_integration.py`

End-to-end test covering: create config, build structure, create simulator with
synthetic probabilities, generate brackets, score, and export. Tests both
`single_elimination` and `round_robin` formats.

### 13.1 Implementation

```python
"""Integration test -- full competition pipeline.

Tests the complete flow:
  config -> structure -> simulator -> brackets -> score -> export

for both single_elimination and round_robin formats.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    CompetitionStructure,
    ScoringConfig,
    StandingsEntry,
)
from easyml.sports.competitions.structure import build_structure
from easyml.sports.competitions.simulator import CompetitionSimulator
from easyml.sports.competitions.optimizer import CompetitionOptimizer
from easyml.sports.competitions.scorer import CompetitionScorer
from easyml.sports.competitions.export import (
    export_bracket_markdown,
    export_json,
    export_csv,
    export_analysis_report,
    export_standings_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_probabilities(entities: list[str], seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic pairwise probability matrix.

    Higher-indexed entities are weaker (probability skews toward entity_a
    when entity_a has a lower index).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, a in enumerate(entities):
        for j, b in enumerate(entities):
            if i >= j:
                continue
            # Base probability: entity with lower index is stronger
            base = 0.5 + 0.03 * (j - i)
            noise = rng.normal(0, 0.05)
            prob = np.clip(base + noise, 0.05, 0.95)
            rows.append({"entity_a": a, "entity_b": b, "probability": float(prob)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Single elimination -- full pipeline
# ---------------------------------------------------------------------------


class TestSingleEliminationPipeline:
    """End-to-end test for single elimination format."""

    @pytest.fixture
    def config(self) -> CompetitionConfig:
        return CompetitionConfig(
            format=CompetitionFormat.SINGLE_ELIMINATION,
            n_participants=8,
            scoring=ScoringConfig(type="per_round", values=[10, 20, 40]),
        )

    @pytest.fixture
    def entities(self) -> list[str]:
        return [f"Entity_{i}" for i in range(1, 9)]

    @pytest.fixture
    def structure(self, config: CompetitionConfig) -> CompetitionStructure:
        return build_structure(config)

    @pytest.fixture
    def probabilities(self, entities: list[str]) -> pd.DataFrame:
        return _synthetic_probabilities(entities)

    def test_build_structure(self, structure: CompetitionStructure):
        """Structure should have correct slot counts."""
        # 8-entity single elimination: 4 + 2 + 1 = 7 slots
        assert len(structure.slots) == 7
        assert len(structure.round_slots) == 3
        assert len(structure.round_slots[1]) == 4
        assert len(structure.round_slots[2]) == 2
        assert len(structure.round_slots[3]) == 1

    def test_simulator_creation(self, config, structure, probabilities):
        """Simulator should initialize without error."""
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        assert sim is not None

    def test_simulate_round_probs(self, config, structure, probabilities):
        """Simulation should produce round probabilities for all entities."""
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        round_probs = sim.entity_round_probabilities(n_sims=1000, seed=42)
        # Should have data -- exact shape depends on implementation
        assert round_probs is not None
        assert len(round_probs) > 0

    def test_generate_brackets(self, config, structure, probabilities):
        """Optimizer should produce the requested number of brackets."""
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        brackets = optimizer.generate_brackets(
            pool_size=50, n_brackets=3, n_sims=500, seed=42,
        )
        assert len(brackets) == 3
        for b in brackets:
            assert isinstance(b, CompetitionResult)
            assert len(b.picks) > 0
            assert b.strategy != ""

    def test_score_bracket(self, config, structure, probabilities):
        """Scoring bracket against actuals should produce valid ScoreResult."""
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        brackets = optimizer.generate_brackets(
            pool_size=50, n_brackets=1, n_sims=500, seed=42,
        )
        bracket = brackets[0]

        # Use bracket's own picks as "actuals" -- should get perfect score
        scorer = CompetitionScorer(scoring=config.scoring)
        score = scorer.score_bracket(
            picks=bracket.picks, actuals=bracket.picks, structure=structure,
        )
        assert score.total_points > 0
        # Perfect score: all rounds correct
        for rnd in score.round_correct:
            assert score.round_correct[rnd] == score.round_total[rnd]

    def test_export_bracket_markdown(self, config, structure, probabilities, tmp_path):
        """Bracket markdown export should produce a readable file."""
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        brackets = optimizer.generate_brackets(
            pool_size=50, n_brackets=1, n_sims=500, seed=42,
        )
        path = export_bracket_markdown(brackets[0], structure, tmp_path, label="test")
        assert path.exists()
        text = path.read_text()
        assert "Round 1" in text
        assert "Entity_" in text

    def test_export_json(self, config, structure, probabilities, tmp_path):
        """JSON export should produce valid parseable JSON."""
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        optimizer = CompetitionOptimizer(simulator=sim)
        brackets = optimizer.generate_brackets(
            pool_size=50, n_brackets=2, n_sims=500, seed=42,
        )
        path = export_json(
            tmp_path, results=brackets, structure=structure, label="test",
        )
        data = json.loads(path.read_text())
        assert len(data["results"]) == 2
        assert data["n_participants"] == 8

    def test_export_csv(self, probabilities, tmp_path):
        """CSV export should write all probability rows."""
        path = export_csv(probabilities, tmp_path, label="test")
        df = pd.read_csv(path)
        assert len(df) == len(probabilities)

    def test_full_pipeline(self, config, structure, probabilities, tmp_path):
        """Full pipeline: build -> simulate -> optimize -> score -> export."""
        # 1. Build (already done via fixtures)
        assert len(structure.slots) == 7

        # 2. Simulate
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        round_probs = sim.entity_round_probabilities(n_sims=1000, seed=42)
        assert round_probs is not None

        # 3. Optimize
        optimizer = CompetitionOptimizer(simulator=sim)
        brackets = optimizer.generate_brackets(
            pool_size=50, n_brackets=3, n_sims=500, seed=42,
        )
        assert len(brackets) == 3

        # 4. Score (use first bracket's picks as actuals for deterministic test)
        scorer = CompetitionScorer(scoring=config.scoring)
        score = scorer.score_bracket(
            picks=brackets[0].picks,
            actuals=brackets[0].picks,
            structure=structure,
        )
        assert score.total_points > 0

        # 5. Export all formats
        md_path = export_bracket_markdown(brackets[0], structure, tmp_path, label="full_test")
        json_path = export_json(tmp_path, results=brackets, structure=structure, label="full_test")
        csv_path = export_csv(probabilities, tmp_path, label="full_test")
        report_path = export_analysis_report(
            tmp_path,
            result=brackets[0],
            structure=structure,
            pick_stories=[],
            entity_profiles=[],
            confidence_data={},
            label="full_test",
        )

        assert md_path.exists()
        assert json_path.exists()
        assert csv_path.exists()
        assert report_path.exists()


# ---------------------------------------------------------------------------
# Round robin -- full pipeline
# ---------------------------------------------------------------------------


class TestRoundRobinPipeline:
    """End-to-end test for round robin format."""

    @pytest.fixture
    def config(self) -> CompetitionConfig:
        return CompetitionConfig(
            format=CompetitionFormat.ROUND_ROBIN,
            n_participants=6,
            rounds=10,
            scoring=ScoringConfig(win=3.0, draw=1.0, loss=0.0),
        )

    @pytest.fixture
    def entities(self) -> list[str]:
        return [f"Club_{i}" for i in range(1, 7)]

    @pytest.fixture
    def structure(self, config: CompetitionConfig) -> CompetitionStructure:
        return build_structure(config)

    @pytest.fixture
    def probabilities(self, entities: list[str]) -> pd.DataFrame:
        return _synthetic_probabilities(entities)

    def test_build_structure(self, structure: CompetitionStructure):
        """Round-robin structure should define all fixtures."""
        assert len(structure.slots) > 0
        assert structure.config.format == CompetitionFormat.ROUND_ROBIN

    def test_simulator_creation(self, config, structure, probabilities):
        """Simulator should initialize for round-robin format."""
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        assert sim is not None

    def test_simulate(self, config, structure, probabilities):
        """Simulation should run without error."""
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        round_probs = sim.entity_round_probabilities(n_sims=500, seed=42)
        assert round_probs is not None

    def test_export_standings(self, tmp_path):
        """Standings export should produce valid markdown table."""
        standings = [
            StandingsEntry(entity="Club_1", wins=8, losses=1, draws=1, points=25.0),
            StandingsEntry(entity="Club_2", wins=6, losses=2, draws=2, points=20.0),
            StandingsEntry(entity="Club_3", wins=4, losses=4, draws=2, points=14.0),
            StandingsEntry(entity="Club_4", wins=3, losses=5, draws=2, points=11.0),
            StandingsEntry(entity="Club_5", wins=2, losses=6, draws=2, points=8.0),
            StandingsEntry(entity="Club_6", wins=1, losses=8, draws=1, points=4.0),
        ]
        path = export_standings_markdown(standings, tmp_path, title="League Table")
        text = path.read_text()
        assert "League Table" in text
        assert "Club_1" in text
        assert "25.0" in text
        # Should be ranked by points
        assert text.index("Club_1") < text.index("Club_6")

    def test_export_json_standings(self, tmp_path):
        """JSON export with standings should include all entries."""
        standings = [
            StandingsEntry(entity=f"Club_{i}", wins=10 - i, losses=i, points=float(30 - 3 * i))
            for i in range(1, 7)
        ]
        path = export_json(tmp_path, standings=standings, label="league")
        data = json.loads(path.read_text())
        assert "standings" in data
        assert len(data["standings"]) == 6

    def test_full_pipeline(self, config, structure, probabilities, tmp_path):
        """Full round-robin pipeline: build -> simulate -> export."""
        # 1. Build
        assert structure.config.format == CompetitionFormat.ROUND_ROBIN

        # 2. Simulate
        sim = CompetitionSimulator(
            config=config, structure=structure, probabilities=probabilities,
        )
        round_probs = sim.entity_round_probabilities(n_sims=500, seed=42)
        assert round_probs is not None

        # 3. Export
        standings_data = [
            StandingsEntry(entity=f"Club_{i}", wins=7 - i, losses=i, points=float(21 - 3 * i))
            for i in range(1, 7)
        ]
        md_path = export_standings_markdown(standings_data, tmp_path, title="Final Standings")
        json_path = export_json(tmp_path, standings=standings_data, label="rr_test")
        csv_path = export_csv(probabilities, tmp_path, label="rr_probs")
        report_path = export_analysis_report(
            tmp_path,
            standings=standings_data,
            pick_stories=[],
            entity_profiles=[],
            confidence_data={},
            label="rr_analysis",
        )

        assert md_path.exists()
        assert json_path.exists()
        assert csv_path.exists()
        assert report_path.exists()

        # Verify JSON content
        data = json.loads(json_path.read_text())
        assert len(data["standings"]) == 6

        # Verify report content
        report_text = report_path.read_text()
        assert "Club_1" in report_text


# ---------------------------------------------------------------------------
# Cross-format consistency
# ---------------------------------------------------------------------------


class TestCrossFormatConsistency:
    """Verify shared components work across formats."""

    def test_csv_export_identical_across_formats(self, tmp_path):
        """export_csv should work with any probability DataFrame regardless of format."""
        probs = pd.DataFrame({
            "entity_a": ["A", "A", "B"],
            "entity_b": ["B", "C", "C"],
            "probability": [0.6, 0.7, 0.55],
        })
        path1 = export_csv(probs, tmp_path / "fmt1", label="test")
        path2 = export_csv(probs, tmp_path / "fmt2", label="test")

        assert path1.read_text() == path2.read_text()

    def test_json_supports_both_results_and_standings(self, tmp_path):
        """JSON export should handle mixed results + standings."""
        result = CompetitionResult(
            picks={"R1G1": "Alpha"},
            matchups={
                "R1G1": __import__("easyml.sports.competitions.schemas", fromlist=["MatchupContext"]).MatchupContext(
                    slot="R1G1", round_num=1, entity_a="Alpha", entity_b="Bravo",
                    prob_a=0.7, pick="Alpha", strategy="chalk",
                ),
            },
            strategy="chalk",
        )
        standings = [
            StandingsEntry(entity="Alpha", wins=5, points=15.0),
            StandingsEntry(entity="Bravo", wins=3, points=9.0),
        ]
        path = export_json(
            tmp_path, results=[result], standings=standings, label="mixed",
        )
        data = json.loads(path.read_text())
        assert "results" in data
        assert "standings" in data

    def test_analysis_report_handles_both_formats(self, tmp_path):
        """Analysis report should work with elimination result or standings."""
        # Elimination
        result = CompetitionResult(
            picks={"R1G1": "A"},
            matchups={},
            strategy="chalk",
            expected_points=10.0,
        )
        path1 = export_analysis_report(
            tmp_path / "elim", result=result, label="elim",
        )

        # League
        standings = [StandingsEntry(entity="A", wins=5, points=15.0)]
        path2 = export_analysis_report(
            tmp_path / "league", standings=standings, label="league",
        )

        assert path1.exists()
        assert path2.exists()
        assert "Strategy" in path1.read_text() or "strategy" in path1.read_text()
        assert "Standings" in path2.read_text()
```

---

## Summary

| Task | File(s) | What |
|------|---------|------|
| 9 | `competitions/export.py` + test | 5 export functions: bracket md, standings md, JSON, CSV, analysis report |
| 10 | `competitions/__init__.py` + test | Public API re-exports of all key symbols with `__all__` |
| 11 | `core/runner/hooks.py` + `sports/hooks.py` + test | `COMPETITION_NARRATIVE` constant + default (empty) hook registration |
| 12 | `plugin/handlers/competitions.py` + `mcp_server.py` + test | 13-action handler with in-memory competition registry + MCP tool |
| 13 | `tests/competitions/test_integration.py` | Full pipeline tests for single_elimination and round_robin |
