"""Tests for the view step execution engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from easyml.runner.schema import (
    CastStep,
    DeriveStep,
    DistinctStep,
    FilterStep,
    GroupByStep,
    JoinStep,
    SelectStep,
    SortStep,
    UnionStep,
    UnpivotStep,
)
from easyml.runner.view_executor import execute_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_games_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [1, 2, 3, 4],
            "Season": [2020, 2020, 2021, 2021],
            "DayNum": [100, 140, 90, 135],
            "Score": [70, 65, 80, 55],
            "OppScore": [60, 70, 75, 60],
        }
    )


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


class TestFilter:
    def test_filter(self):
        df = _make_games_df()
        step = FilterStep(expr="DayNum < 134")
        result = execute_step(df, step)
        assert len(result) == 2
        assert list(result["DayNum"]) == [100, 90]
        # Index should be reset
        assert list(result.index) == [0, 1]


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------


class TestSelect:
    def test_select_list(self):
        df = _make_games_df()
        step = SelectStep(columns=["game_id", "Score"])
        result = execute_step(df, step)
        assert list(result.columns) == ["game_id", "Score"]
        assert len(result) == 4

    def test_select_dict(self):
        df = _make_games_df()
        step = SelectStep(columns={"id": "game_id", "pts": "Score"})
        result = execute_step(df, step)
        assert list(result.columns) == ["id", "pts"]
        assert list(result["id"]) == [1, 2, 3, 4]
        assert list(result["pts"]) == [70, 65, 80, 55]


# ---------------------------------------------------------------------------
# derive
# ---------------------------------------------------------------------------


class TestDerive:
    def test_derive_arithmetic(self):
        df = _make_games_df()
        step = DeriveStep(columns={"margin": "Score - OppScore"})
        result = execute_step(df, step)
        assert "margin" in result.columns
        assert list(result["margin"]) == [10, -5, 5, -5]

    def test_derive_where(self):
        df = pd.DataFrame({"val": [10, 20, 30, 40]})
        step = DeriveStep(columns={"big": "where(val > 20, 1, 0)"})
        result = execute_step(df, step)
        assert list(result["big"]) == [0, 0, 1, 1]

    def test_derive_string_methods(self):
        df = pd.DataFrame({"Seed": ["W01", "X03", "Y16", "Z08"]})
        step = DeriveStep(columns={"region": "Seed.str[0:1]"})
        result = execute_step(df, step)
        assert list(result["region"]) == ["W", "X", "Y", "Z"]

    def test_derive_chained(self):
        df = pd.DataFrame({"Seed": ["W01", "X03", "Y16", "Z08"]})
        step = DeriveStep(columns={"seed_num": "Seed.str[1:3].astype(int)"})
        result = execute_step(df, step)
        assert list(result["seed_num"]) == [1, 3, 16, 8]
        assert result["seed_num"].dtype == int


# ---------------------------------------------------------------------------
# group_by
# ---------------------------------------------------------------------------


class TestGroupBy:
    def test_group_by_single_agg(self):
        df = _make_games_df()
        step = GroupByStep(keys=["Season"], aggs={"Score": "mean"})
        result = execute_step(df, step)
        assert list(result.columns) == ["Season", "Score_mean"]
        assert len(result) == 2
        # Season 2020: mean(70,65)=67.5, Season 2021: mean(80,55)=67.5
        assert list(result["Score_mean"]) == [67.5, 67.5]

    def test_group_by_multi_agg(self):
        df = _make_games_df()
        step = GroupByStep(keys=["Season"], aggs={"Score": ["mean", "max"]})
        result = execute_step(df, step)
        assert "Score_mean" in result.columns
        assert "Score_max" in result.columns
        assert len(result) == 2
        # Season 2020: max(70,65)=70, Season 2021: max(80,55)=80
        assert list(result["Score_max"]) == [70, 80]


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------


class TestJoin:
    def test_join_same_keys(self):
        games = pd.DataFrame({"team_id": [1, 2, 3], "score": [70, 65, 80]})
        teams = pd.DataFrame(
            {"team_id": [1, 2, 3], "name": ["A", "B", "C"], "conf": ["E", "W", "E"]}
        )
        step = JoinStep(other="teams", on=["team_id"])
        resolver = lambda name: teams
        result = execute_step(games, step, resolver=resolver)
        assert "name" in result.columns
        assert list(result["name"]) == ["A", "B", "C"]

    def test_join_dict_keys(self):
        games = pd.DataFrame({"home_id": [1, 2], "pts": [70, 65]})
        teams = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        step = JoinStep(other="teams", on={"home_id": "id"})
        resolver = lambda name: teams
        result = execute_step(games, step, resolver=resolver)
        assert "name" in result.columns
        assert list(result["name"]) == ["A", "B"]

    def test_join_with_prefix(self):
        games = pd.DataFrame({"team_id": [1, 2], "score": [70, 65]})
        ratings = pd.DataFrame({"team_id": [1, 2], "rating": [95.0, 88.0]})
        step = JoinStep(other="ratings", on=["team_id"], prefix="rtg_")
        resolver = lambda name: ratings
        result = execute_step(games, step, resolver=resolver)
        assert "rtg_rating" in result.columns
        assert "team_id" in result.columns  # join key not prefixed
        assert list(result["rtg_rating"]) == [95.0, 88.0]


# ---------------------------------------------------------------------------
# union
# ---------------------------------------------------------------------------


class TestUnion:
    def test_union(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        step = UnionStep(other="extra")
        resolver = lambda name: df2
        result = execute_step(df1, step, resolver=resolver)
        assert len(result) == 4
        assert list(result["a"]) == [1, 2, 5, 6]
        assert list(result.index) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# unpivot
# ---------------------------------------------------------------------------


class TestUnpivot:
    def test_unpivot(self):
        df = pd.DataFrame(
            {
                "game_id": [1, 2],
                "season": [2020, 2021],
                "w_score": [70, 80],
                "l_score": [60, 75],
                "w_team": [101, 201],
                "l_team": [102, 202],
            }
        )
        step = UnpivotStep(
            id_columns=["game_id", "season"],
            unpivot_columns={
                "score": ["w_score", "l_score"],
                "team_id": ["w_team", "l_team"],
            },
            names_column="result",
            names_map={"w_score": "winner", "l_score": "loser"},
        )
        result = execute_step(df, step)
        assert len(result) == 4
        assert set(result.columns) == {"game_id", "season", "score", "team_id", "result"}
        # First two rows are position 0 (winners), next two are position 1 (losers)
        assert list(result["result"]) == ["winner", "winner", "loser", "loser"]
        assert list(result["score"]) == [70, 80, 60, 75]
        assert list(result["team_id"]) == [101, 201, 102, 202]


# ---------------------------------------------------------------------------
# cast
# ---------------------------------------------------------------------------


class TestCast:
    def test_cast(self):
        df = pd.DataFrame({"val": ["1", "2", "3"], "ratio": [1, 2, 3]})
        step = CastStep(columns={"val": "int", "ratio": "float"})
        result = execute_step(df, step)
        assert result["val"].dtype == int
        assert result["ratio"].dtype == float

    def test_cast_with_source_expr(self):
        df = pd.DataFrame({"Seed": ["W01", "X03", "Y16"]})
        step = CastStep(columns={"Seed": "int:str[1:3]"})
        result = execute_step(df, step)
        assert list(result["Seed"]) == [1, 3, 16]
        assert result["Seed"].dtype == int


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------


class TestSort:
    def test_sort(self):
        df = pd.DataFrame({"name": ["c", "a", "b"], "val": [3, 1, 2]})
        step = SortStep(by="val")
        result = execute_step(df, step)
        assert list(result["val"]) == [1, 2, 3]
        assert list(result["name"]) == ["a", "b", "c"]
        assert list(result.index) == [0, 1, 2]

    def test_sort_descending(self):
        df = pd.DataFrame({"name": ["c", "a", "b"], "val": [3, 1, 2]})
        step = SortStep(by="val", ascending=False)
        result = execute_step(df, step)
        assert list(result["val"]) == [3, 2, 1]
        assert list(result["name"]) == ["c", "b", "a"]


# ---------------------------------------------------------------------------
# distinct
# ---------------------------------------------------------------------------


class TestDistinct:
    def test_distinct(self):
        df = pd.DataFrame({"a": [1, 2, 1, 2], "b": [10, 20, 10, 20]})
        step = DistinctStep()
        result = execute_step(df, step)
        assert len(result) == 2
        assert list(result.index) == [0, 1]

    def test_distinct_subset(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
        step = DistinctStep(columns=["a"], keep="last")
        result = execute_step(df, step)
        assert len(result) == 2
        # keep="last" means for a=1 we keep the row with b=20
        assert list(result["b"]) == [20, 30]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_join_without_resolver_raises(self):
        df = pd.DataFrame({"a": [1]})
        step = JoinStep(other="t", on=["a"])
        with pytest.raises(ValueError, match="resolver is required"):
            execute_step(df, step)

    def test_union_without_resolver_raises(self):
        df = pd.DataFrame({"a": [1]})
        step = UnionStep(other="t")
        with pytest.raises(ValueError, match="resolver is required"):
            execute_step(df, step)
