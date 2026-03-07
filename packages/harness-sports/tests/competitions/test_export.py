"""Tests for competition export module."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from harnessml.sports.competitions.export import (
    export_analysis_report,
    export_bracket_markdown,
    export_csv,
    export_json,
    export_standings_markdown,
)
from harnessml.sports.competitions.schemas import (
    CompetitionConfig,
    CompetitionFormat,
    CompetitionResult,
    CompetitionStructure,
    MatchupContext,
    StandingsEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ENTITY_NAMES = {
    "t1": "Alpha",
    "t2": "Bravo",
    "t3": "Charlie",
    "t4": "Delta",
}


@pytest.fixture()
def entity_names() -> dict[str, str]:
    return dict(ENTITY_NAMES)


@pytest.fixture()
def competition_result() -> CompetitionResult:
    return CompetitionResult(
        picks={"R1G1": "t1", "R1G2": "t3", "R2G1": "t1"},
        matchups={
            "R1G1": MatchupContext(
                slot="R1G1",
                round_num=1,
                entity_a="t1",
                entity_b="t2",
                prob_a=0.75,
                model_agreement=0.9,
                pick="t1",
                strategy="chalk",
                upset=False,
            ),
            "R1G2": MatchupContext(
                slot="R1G2",
                round_num=1,
                entity_a="t3",
                entity_b="t4",
                prob_a=0.40,
                model_agreement=0.6,
                pick="t3",
                strategy="chalk",
                upset=True,
            ),
            "R2G1": MatchupContext(
                slot="R2G1",
                round_num=2,
                entity_a="t1",
                entity_b="t3",
                prob_a=0.65,
                model_agreement=0.85,
                pick="t1",
                strategy="chalk",
                upset=False,
            ),
        },
        expected_points=12.5,
        win_probability=0.32,
        strategy="chalk",
    )


@pytest.fixture()
def structure() -> CompetitionStructure:
    return CompetitionStructure(
        config=CompetitionConfig(
            format=CompetitionFormat.single_elimination,
            n_participants=4,
            rounds=["Semifinals", "Final"],
        ),
        slots=["R1G1", "R1G2", "R2G1"],
        slot_to_round={"R1G1": 1, "R1G2": 1, "R2G1": 2},
        round_slots={1: ["R1G1", "R1G2"], 2: ["R2G1"]},
    )


@pytest.fixture()
def standings() -> list[StandingsEntry]:
    return [
        StandingsEntry(entity="t1", wins=3, losses=0, draws=0, points=9, goal_diff=5),
        StandingsEntry(entity="t2", wins=2, losses=1, draws=0, points=6, goal_diff=2),
        StandingsEntry(entity="t3", wins=1, losses=2, draws=0, points=3, goal_diff=-1),
        StandingsEntry(entity="t4", wins=0, losses=3, draws=0, points=0, goal_diff=-6),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExportBracketMarkdown:
    def test_creates_file(self, tmp_path, competition_result, entity_names, structure):
        path = export_bracket_markdown(
            competition_result, entity_names, structure, tmp_path
        )
        assert path.exists()
        assert path.name == "bracket.md"

    def test_contains_pick_names(
        self, tmp_path, competition_result, entity_names, structure
    ):
        path = export_bracket_markdown(
            competition_result, entity_names, structure, tmp_path
        )
        content = path.read_text()
        assert "**Alpha**" in content
        assert "**Charlie**" in content

    def test_contains_upset_markers(
        self, tmp_path, competition_result, entity_names, structure
    ):
        path = export_bracket_markdown(
            competition_result, entity_names, structure, tmp_path
        )
        content = path.read_text()
        assert "[UPSET]" in content
        # Only the t3 over t4 matchup should be upset
        lines_with_upset = [l for l in content.splitlines() if "[UPSET]" in l]
        assert len(lines_with_upset) == 1
        assert "Charlie" in lines_with_upset[0]

    def test_contains_round_names(
        self, tmp_path, competition_result, entity_names, structure
    ):
        path = export_bracket_markdown(
            competition_result, entity_names, structure, tmp_path
        )
        content = path.read_text()
        assert "## Semifinals" in content
        assert "## Final" in content

    def test_contains_strategy_header(
        self, tmp_path, competition_result, entity_names, structure
    ):
        path = export_bracket_markdown(
            competition_result, entity_names, structure, tmp_path
        )
        content = path.read_text()
        assert "**Strategy:** chalk" in content
        assert "**Expected Points:** 12.50" in content


class TestExportStandingsMarkdown:
    def test_creates_file(self, tmp_path, standings, entity_names):
        path = export_standings_markdown(standings, entity_names, tmp_path)
        assert path.exists()
        assert path.name == "standings.md"

    def test_table_format(self, tmp_path, standings, entity_names):
        path = export_standings_markdown(standings, entity_names, tmp_path)
        content = path.read_text()
        assert "| Rank | Entity | W | L | D | Pts | GD |" in content
        assert "|------|--------|---|---|---|-----|-----|" in content

    def test_contains_entity_names(self, tmp_path, standings, entity_names):
        path = export_standings_markdown(standings, entity_names, tmp_path)
        content = path.read_text()
        assert "Alpha" in content
        assert "Bravo" in content
        assert "Charlie" in content
        assert "Delta" in content

    def test_rank_ordering(self, tmp_path, standings, entity_names):
        path = export_standings_markdown(standings, entity_names, tmp_path)
        content = path.read_text()
        lines = [l for l in content.splitlines() if l.startswith("| ") and "Rank" not in l and "---" not in l]
        assert "1" in lines[0] and "Alpha" in lines[0]
        assert "4" in lines[3] and "Delta" in lines[3]


class TestExportJson:
    def test_creates_file_competition_result(
        self, tmp_path, competition_result, entity_names
    ):
        path = export_json([competition_result], entity_names, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1

    def test_contains_picks_named(self, tmp_path, competition_result, entity_names):
        path = export_json([competition_result], entity_names, tmp_path)
        data = json.loads(path.read_text())
        picks_named = data[0]["picks_named"]
        assert picks_named["R1G1"] == "Alpha"
        assert picks_named["R1G2"] == "Charlie"

    def test_standings_export(self, tmp_path, standings, entity_names):
        path = export_json(standings, entity_names, tmp_path, filename="standings.json")
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 4
        assert data[0]["rank"] == 1
        assert data[0]["entity_name"] == "Alpha"

    def test_empty_list(self, tmp_path, entity_names):
        path = export_json([], entity_names, tmp_path)
        data = json.loads(path.read_text())
        assert data == []


class TestExportCsv:
    def test_creates_valid_csv(self, tmp_path):
        df = pd.DataFrame(
            {
                "entity_a": ["t1", "t3"],
                "entity_b": ["t2", "t4"],
                "prob_a": [0.75, 0.40],
            }
        )
        path = export_csv(df, tmp_path)
        assert path.exists()
        assert path.name == "probabilities.csv"
        loaded = pd.read_csv(path)
        assert len(loaded) == 2
        assert list(loaded.columns) == ["entity_a", "entity_b", "prob_a"]

    def test_custom_filename(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3]})
        path = export_csv(df, tmp_path, filename="custom.csv")
        assert path.name == "custom.csv"
        assert path.exists()


class TestExportAnalysisReport:
    def test_creates_file(self, tmp_path, competition_result, entity_names):
        path = export_analysis_report(
            result=competition_result,
            pick_stories=[
                {"slot": "R1G1", "narrative": "Alpha was favored.", "confidence": "high"}
            ],
            entity_profiles=[
                {"entity": "t1", "strength": "high", "trend": "improving"}
            ],
            confidence_report={
                "overall_confidence": "medium",
                "flags": ["low agreement in R1G2"],
            },
            entity_names=entity_names,
            output_dir=tmp_path,
        )
        assert path.exists()
        assert path.name == "analysis.md"

    def test_combines_all_sections(self, tmp_path, competition_result, entity_names):
        path = export_analysis_report(
            result=competition_result,
            pick_stories=[
                {"slot": "R1G1", "narrative": "Alpha was favored.", "confidence": "high"}
            ],
            entity_profiles=[
                {"entity": "t1", "strength": "high", "trend": "improving"}
            ],
            confidence_report={
                "overall_confidence": "medium",
                "flags": ["low agreement in R1G2"],
            },
            entity_names=entity_names,
            output_dir=tmp_path,
        )
        content = path.read_text()
        assert "## Summary" in content
        assert "## Confidence Diagnostics" in content
        assert "## Entity Profiles" in content
        assert "## Pick Details" in content
        assert "Alpha" in content
        assert "Alpha was favored." in content
        assert "low agreement in R1G2" in content
        assert "**Strategy:** chalk" in content
