"""Tests for reshape transform steps: join, union, unpivot, aggregate, conditional_agg."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from harness.data.transforms.steps import join, union, unpivot, aggregate, conditional_agg


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------

def make_resolver(frames: dict):
    """Return a resolver callable backed by the given dict of DataFrames."""
    def resolver(name):
        return frames[name]
    return resolver


def test_join_left(sample_df):
    other = pd.DataFrame({"id": [1, 2, 3], "department": ["Eng", "HR", "Fin"]})
    resolver = make_resolver({"other": other})
    result = join.step(sample_df, {"other": "other", "on": "id", "how": "left", "_resolver": resolver})
    assert "department" in result.columns
    assert len(result) == len(sample_df)
    # rows 4 and 5 have no match — department should be NaN
    assert pd.isna(result.loc[result["id"] == 4, "department"].iloc[0])
    assert pd.isna(result.loc[result["id"] == 5, "department"].iloc[0])


def test_join_inner(sample_df):
    other = pd.DataFrame({"id": [1, 2, 3], "department": ["Eng", "HR", "Fin"]})
    resolver = make_resolver({"other": other})
    result = join.step(sample_df, {"other": "other", "on": "id", "how": "inner", "_resolver": resolver})
    assert len(result) == 3
    assert set(result["id"]) == {1, 2, 3}


def test_join_missing_other_param(sample_df):
    with pytest.raises(ValueError, match="join step requires"):
        join.step(sample_df, {"on": "id", "_resolver": make_resolver({})})


def test_join_missing_on_param(sample_df):
    with pytest.raises(ValueError, match="join step requires"):
        join.step(sample_df, {"other": "x", "_resolver": make_resolver({})})


def test_join_missing_resolver(sample_df):
    with pytest.raises(ValueError, match="resolver"):
        join.step(sample_df, {"other": "x", "on": "id"})


def test_join_with_prefix(sample_df):
    other = pd.DataFrame({"id": [1, 2], "rank": [1, 2]})
    resolver = make_resolver({"other": other})
    result = join.step(
        sample_df,
        {"other": "other", "on": "id", "how": "left", "prefix": "other_", "_resolver": resolver},
    )
    assert "other_rank" in result.columns


# ---------------------------------------------------------------------------
# union
# ---------------------------------------------------------------------------

def test_union_concat():
    df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df2 = pd.DataFrame({"a": [3, 4], "b": [30, 40]})
    resolver = make_resolver({"df2": df2})
    result = union.step(df1, {"other": "df2", "_resolver": resolver})
    assert len(result) == 4
    assert list(result["a"]) == [1, 2, 3, 4]
    assert list(result.index) == [0, 1, 2, 3]


def test_union_missing_other():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="union step requires 'other'"):
        union.step(df, {"_resolver": make_resolver({})})


def test_union_missing_resolver():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="resolver"):
        union.step(df, {"other": "x"})


# ---------------------------------------------------------------------------
# unpivot
# ---------------------------------------------------------------------------

def test_unpivot_basic():
    df = pd.DataFrame({
        "id": [1, 2],
        "jan": [100, 200],
        "feb": [110, 210],
        "mar": [120, 220],
    })
    result = unpivot.step(df, {"unpivot_columns": {"revenue": ["jan", "feb", "mar"]}})
    assert "revenue" in result.columns
    assert "revenue_source" in result.columns
    assert len(result) == 6  # 2 ids × 3 months
    assert set(result["revenue_source"]) == {"jan", "feb", "mar"}


def test_unpivot_missing_unpivot_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="unpivot step requires"):
        unpivot.step(df, {})


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_aggregate_group_mean(numeric_df):
    result = aggregate.step(numeric_df, {"keys": ["entity_id"], "aggs": {"points": "mean"}})
    assert "entity_id" in result.columns
    # MultiIndex flattening produces "points_mean"
    points_col = "points_mean" if "points_mean" in result.columns else "points"
    assert points_col in result.columns
    entity1_mean = result.loc[result["entity_id"] == 1, points_col].iloc[0]
    assert abs(entity1_mean - (10.0 + 15.0 + 12.0) / 3) < 1e-6


def test_aggregate_group_sum(numeric_df):
    result = aggregate.step(numeric_df, {"keys": ["entity_id"], "aggs": {"rebounds": "sum"}})
    rebounds_col = "rebounds_sum" if "rebounds_sum" in result.columns else "rebounds"
    entity2_sum = result.loc[result["entity_id"] == 2, rebounds_col].iloc[0]
    assert entity2_sum == 10.0 + 9.0 + 11.0


def test_aggregate_missing_keys(numeric_df):
    with pytest.raises(ValueError, match="aggregate step requires"):
        aggregate.step(numeric_df, {"aggs": {"points": "mean"}})


def test_aggregate_missing_aggs(numeric_df):
    with pytest.raises(ValueError, match="aggregate step requires"):
        aggregate.step(numeric_df, {"keys": ["entity_id"]})


# ---------------------------------------------------------------------------
# conditional_agg
# ---------------------------------------------------------------------------

def test_conditional_agg_no_condition(numeric_df):
    result = conditional_agg.step(
        numeric_df,
        {"keys": ["entity_id"], "aggs": {"total_points": "points:sum"}},
    )
    assert "total_points" in result.columns
    entity1_total = result.loc[result["entity_id"] == 1, "total_points"].iloc[0]
    assert entity1_total == 10.0 + 15.0 + 12.0


def test_conditional_agg_with_condition(numeric_df):
    # Sum points only where target == 1
    result = conditional_agg.step(
        numeric_df,
        {"keys": ["entity_id"], "aggs": {"winning_points": "points:sum:target == 1"}},
    )
    assert "winning_points" in result.columns
    # entity_id 1: target=[1,0,1] → points=[10,12] → sum=22
    entity1_val = result.loc[result["entity_id"] == 1, "winning_points"].iloc[0]
    assert entity1_val == 10.0 + 12.0


def test_conditional_agg_missing_keys(numeric_df):
    with pytest.raises(ValueError, match="conditional_agg step requires"):
        conditional_agg.step(numeric_df, {"aggs": {"x": "points:sum"}})


def test_conditional_agg_missing_aggs(numeric_df):
    with pytest.raises(ValueError, match="conditional_agg step requires"):
        conditional_agg.step(numeric_df, {"keys": ["entity_id"]})
