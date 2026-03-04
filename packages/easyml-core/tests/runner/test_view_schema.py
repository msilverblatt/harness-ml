"""Tests for TransformStep variants, ViewDef, and DataConfig view fields."""
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from easyml.core.runner.schema import (
    CastStep,
    DataConfig,
    DeriveStep,
    DistinctStep,
    FilterStep,
    GroupByStep,
    JoinStep,
    SelectStep,
    SortStep,
    TransformStep,
    UnionStep,
    UnpivotStep,
    ViewDef,
)

# Reusable adapter for parsing TransformStep discriminated unions from dicts
_step_adapter = TypeAdapter(TransformStep)


# -----------------------------------------------------------------------
# Individual step instantiation + round-trip
# -----------------------------------------------------------------------


class TestFilterStep:
    def test_instantiate(self):
        step = FilterStep(expr="DayNum < 134")
        assert step.op == "filter"
        assert step.expr == "DayNum < 134"

    def test_round_trip(self):
        step = FilterStep(expr="status == 'active'")
        data = step.model_dump()
        rebuilt = FilterStep.model_validate(data)
        assert rebuilt == step


class TestSelectStep:
    def test_list_form(self):
        step = SelectStep(columns=["a", "b", "c"])
        assert step.op == "select"
        assert step.columns == ["a", "b", "c"]

    def test_dict_form(self):
        step = SelectStep(columns={"new_a": "old_a", "new_b": "old_b"})
        assert step.columns == {"new_a": "old_a", "new_b": "old_b"}

    def test_round_trip(self):
        step = SelectStep(columns=["x", "y"])
        assert SelectStep.model_validate(step.model_dump()) == step


class TestDeriveStep:
    def test_instantiate(self):
        step = DeriveStep(columns={"ratio": "a / b", "total": "a + b"})
        assert step.op == "derive"
        assert step.columns["ratio"] == "a / b"

    def test_round_trip(self):
        step = DeriveStep(columns={"c": "a * 2"})
        assert DeriveStep.model_validate(step.model_dump()) == step


class TestGroupByStep:
    def test_single_agg(self):
        step = GroupByStep(keys=["team_id"], aggs={"score": "mean"})
        assert step.op == "group_by"
        assert step.keys == ["team_id"]
        assert step.aggs == {"score": "mean"}

    def test_multi_agg(self):
        step = GroupByStep(keys=["team_id"], aggs={"score": ["mean", "std"]})
        assert step.aggs["score"] == ["mean", "std"]

    def test_round_trip(self):
        step = GroupByStep(keys=["k"], aggs={"v": ["sum", "count"]})
        assert GroupByStep.model_validate(step.model_dump()) == step


class TestJoinStep:
    def test_defaults(self):
        step = JoinStep(other="teams", on=["team_id"])
        assert step.op == "join"
        assert step.how == "left"
        assert step.select is None
        assert step.prefix is None

    def test_full(self):
        step = JoinStep(
            other="ratings",
            on={"team_id": "id"},
            how="inner",
            select=["rating", "rank"],
            prefix="rtg_",
        )
        assert step.on == {"team_id": "id"}
        assert step.how == "inner"
        assert step.select == ["rating", "rank"]
        assert step.prefix == "rtg_"

    def test_round_trip(self):
        step = JoinStep(other="x", on=["id"], how="outer")
        assert JoinStep.model_validate(step.model_dump()) == step


class TestUnionStep:
    def test_instantiate(self):
        step = UnionStep(other="extra_data")
        assert step.op == "union"
        assert step.other == "extra_data"

    def test_round_trip(self):
        step = UnionStep(other="extra")
        assert UnionStep.model_validate(step.model_dump()) == step


class TestUnpivotStep:
    def test_instantiate(self):
        step = UnpivotStep(
            id_columns=["game_id"],
            unpivot_columns={"score": ["home_score", "away_score"]},
        )
        assert step.op == "unpivot"
        assert step.id_columns == ["game_id"]
        assert step.names_column is None
        assert step.names_map is None

    def test_full(self):
        step = UnpivotStep(
            id_columns=["game_id"],
            unpivot_columns={"score": ["h_score", "a_score"]},
            names_column="side",
            names_map={"h_score": "home", "a_score": "away"},
        )
        assert step.names_column == "side"
        assert step.names_map == {"h_score": "home", "a_score": "away"}

    def test_round_trip(self):
        step = UnpivotStep(
            id_columns=["id"],
            unpivot_columns={"val": ["a", "b"]},
        )
        assert UnpivotStep.model_validate(step.model_dump()) == step


class TestCastStep:
    def test_instantiate(self):
        step = CastStep(columns={"age": "int", "score": "float"})
        assert step.op == "cast"
        assert step.columns == {"age": "int", "score": "float"}

    def test_round_trip(self):
        step = CastStep(columns={"x": "int:str[1:3]"})
        assert CastStep.model_validate(step.model_dump()) == step


class TestSortStep:
    def test_string_by(self):
        step = SortStep(by="score")
        assert step.op == "sort"
        assert step.by == "score"
        assert step.ascending is True

    def test_list_by(self):
        step = SortStep(by=["score", "name"], ascending=[False, True])
        assert step.by == ["score", "name"]
        assert step.ascending == [False, True]

    def test_round_trip(self):
        step = SortStep(by=["x"], ascending=False)
        assert SortStep.model_validate(step.model_dump()) == step


class TestDistinctStep:
    def test_defaults(self):
        step = DistinctStep()
        assert step.op == "distinct"
        assert step.columns is None
        assert step.keep == "first"

    def test_custom(self):
        step = DistinctStep(columns=["id", "date"], keep="last")
        assert step.columns == ["id", "date"]
        assert step.keep == "last"

    def test_round_trip(self):
        step = DistinctStep(columns=["a"])
        assert DistinctStep.model_validate(step.model_dump()) == step


# -----------------------------------------------------------------------
# Discriminated union tests
# -----------------------------------------------------------------------


class TestTransformStepUnion:
    """Parse raw dicts via the discriminated union and verify correct type."""

    @pytest.mark.parametrize(
        "raw, expected_type",
        [
            ({"op": "filter", "expr": "x > 0"}, FilterStep),
            ({"op": "select", "columns": ["a"]}, SelectStep),
            ({"op": "derive", "columns": {"c": "a+b"}}, DeriveStep),
            ({"op": "group_by", "keys": ["k"], "aggs": {"v": "sum"}}, GroupByStep),
            ({"op": "join", "other": "t", "on": ["id"]}, JoinStep),
            ({"op": "union", "other": "t2"}, UnionStep),
            (
                {
                    "op": "unpivot",
                    "id_columns": ["id"],
                    "unpivot_columns": {"val": ["a", "b"]},
                },
                UnpivotStep,
            ),
            ({"op": "cast", "columns": {"x": "int"}}, CastStep),
            ({"op": "sort", "by": "score"}, SortStep),
            ({"op": "distinct"}, DistinctStep),
        ],
    )
    def test_discriminator_resolves(self, raw: dict, expected_type: type):
        parsed = _step_adapter.validate_python(raw)
        assert isinstance(parsed, expected_type)

    def test_unknown_op_raises(self):
        with pytest.raises(ValidationError):
            _step_adapter.validate_python({"op": "nonexistent"})

    def test_round_trip_via_adapter(self):
        raw = {"op": "join", "other": "src", "on": {"a": "b"}, "how": "inner"}
        parsed = _step_adapter.validate_python(raw)
        dumped = _step_adapter.dump_python(parsed)
        reparsed = _step_adapter.validate_python(dumped)
        assert reparsed == parsed


# -----------------------------------------------------------------------
# ViewDef tests
# -----------------------------------------------------------------------


class TestViewDef:
    def test_minimal(self):
        view = ViewDef(source="raw_games")
        assert view.source == "raw_games"
        assert view.steps == []
        assert view.description == ""
        assert view.cache is True

    def test_with_steps(self):
        view = ViewDef(
            source="games",
            steps=[
                {"op": "filter", "expr": "season > 2010"},
                {"op": "select", "columns": ["team", "score"]},
                {"op": "derive", "columns": {"double": "score * 2"}},
            ],
            description="Filtered games",
            cache=False,
        )
        assert len(view.steps) == 3
        assert isinstance(view.steps[0], FilterStep)
        assert isinstance(view.steps[1], SelectStep)
        assert isinstance(view.steps[2], DeriveStep)
        assert view.cache is False

    def test_round_trip(self):
        view = ViewDef(
            source="src",
            steps=[
                {"op": "sort", "by": ["a", "b"], "ascending": [True, False]},
                {"op": "distinct", "columns": ["a"], "keep": "last"},
            ],
            description="sorted distinct",
        )
        data = view.model_dump()
        rebuilt = ViewDef.model_validate(data)
        assert rebuilt == view
        assert isinstance(rebuilt.steps[0], SortStep)
        assert isinstance(rebuilt.steps[1], DistinctStep)


# -----------------------------------------------------------------------
# DataConfig additions tests
# -----------------------------------------------------------------------


class TestDataConfigViewFields:
    def test_defaults(self):
        dc = DataConfig()
        assert dc.views == {}
        assert dc.features_view is None

    def test_with_views(self):
        dc = DataConfig(
            views={
                "clean_games": ViewDef(
                    source="raw",
                    steps=[{"op": "filter", "expr": "valid == 1"}],
                ),
                "team_stats": ViewDef(
                    source="clean_games",
                    steps=[
                        {"op": "group_by", "keys": ["team_id"], "aggs": {"pts": "mean"}},
                    ],
                ),
            },
            features_view="team_stats",
        )
        assert len(dc.views) == 2
        assert dc.features_view == "team_stats"
        assert isinstance(dc.views["clean_games"].steps[0], FilterStep)

    def test_round_trip(self):
        dc = DataConfig(
            views={
                "v1": ViewDef(
                    source="s1",
                    steps=[{"op": "cast", "columns": {"x": "float"}}],
                ),
            },
            features_view="v1",
        )
        data = dc.model_dump()
        rebuilt = DataConfig.model_validate(data)
        assert rebuilt.views == dc.views
        assert rebuilt.features_view == dc.features_view

    def test_existing_fields_untouched(self):
        """Verify that existing DataConfig fields still work as before."""
        dc = DataConfig(
            raw_dir="my/raw",
            task="regression",
            target_column="price",
            key_columns=["id"],
        )
        assert dc.raw_dir == "my/raw"
        assert dc.task == "regression"
        assert dc.target_column == "price"
        assert dc.key_columns == ["id"]
        # New fields should have defaults
        assert dc.views == {}
        assert dc.features_view is None
