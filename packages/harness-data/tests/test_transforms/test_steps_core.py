"""Tests for core transform steps: derive, cast, fill, sort, head, distinct."""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from harness.data.transforms.steps import derive, cast, fill, sort, head, distinct


# ---------------------------------------------------------------------------
# derive
# ---------------------------------------------------------------------------

def test_derive_arithmetic(sample_df):
    result = derive.step(sample_df, {"columns": {"score_doubled": "score * 2"}})
    assert "score_doubled" in result.columns
    assert list(result["score_doubled"]) == [s * 2 for s in sample_df["score"]]


def test_derive_multiple_columns(sample_df):
    result = derive.step(sample_df, {"columns": {
        "score_plus_id": "score + id",
        "id_squared": "id * id",
    }})
    assert "score_plus_id" in result.columns
    assert "id_squared" in result.columns
    assert list(result["id_squared"]) == [i * i for i in sample_df["id"]]


def test_derive_missing_columns_param(sample_df):
    with pytest.raises(ValueError, match="derive step requires"):
        derive.step(sample_df, {})


def test_derive_invalid_columns_type(sample_df):
    with pytest.raises(ValueError, match="derive step requires"):
        derive.step(sample_df, {"columns": ["score * 2"]})


# ---------------------------------------------------------------------------
# cast
# ---------------------------------------------------------------------------

def test_cast_to_int(sample_df):
    result = cast.step(sample_df, {"columns": {"score": "int"}})
    assert result["score"].dtype == int
    assert list(result["score"]) == [85, 92, 78, 95, 88]


def test_cast_to_str(sample_df):
    result = cast.step(sample_df, {"columns": {"id": "str"}})
    dtype_name = result["id"].dtype.name.lower()
    assert dtype_name in ("object", "str", "string")
    assert list(result["id"]) == ["1", "2", "3", "4", "5"]


def test_cast_missing_column(sample_df):
    with pytest.raises(ValueError, match="Column not found"):
        cast.step(sample_df, {"columns": {"nonexistent": "int"}})


def test_cast_missing_param(sample_df):
    with pytest.raises(ValueError, match="cast step requires"):
        cast.step(sample_df, {})


# ---------------------------------------------------------------------------
# fill
# ---------------------------------------------------------------------------

def test_fill_with_value():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, 3.0]})
    result = fill.step(df, {"columns": {"a": 0, "b": -1}})
    assert result["a"].tolist() == [1.0, 0.0, 3.0]
    assert result["b"].tolist() == [-1.0, 2.0, 3.0]


def test_fill_strategy_median():
    df = pd.DataFrame({"x": [1.0, None, 3.0, None, 5.0]})
    result = fill.step(df, {"strategy": "median"})
    # median of [1, 3, 5] = 3.0
    assert result["x"].isna().sum() == 0
    assert result["x"].iloc[1] == 3.0


def test_fill_strategy_zero():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    result = fill.step(df, {"strategy": "zero"})
    assert result["x"].iloc[1] == 0.0


def test_fill_strategy_ffill():
    df = pd.DataFrame({"x": [1.0, None, None, 4.0]})
    result = fill.step(df, {"strategy": "ffill"})
    assert result["x"].tolist() == [1.0, 1.0, 1.0, 4.0]


def test_fill_unknown_strategy(sample_df):
    with pytest.raises(ValueError, match="Unknown fill strategy"):
        fill.step(sample_df, {"strategy": "bogus"})


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------

def test_sort_ascending(sample_df):
    result = sort.step(sample_df, {"by": "score"})
    assert list(result["score"]) == sorted(sample_df["score"])


def test_sort_descending(sample_df):
    result = sort.step(sample_df, {"by": "score", "ascending": False})
    assert list(result["score"]) == sorted(sample_df["score"], reverse=True)


def test_sort_resets_index(sample_df):
    result = sort.step(sample_df, {"by": "score"})
    assert list(result.index) == list(range(len(result)))


def test_sort_missing_by(sample_df):
    with pytest.raises(ValueError, match="sort step requires"):
        sort.step(sample_df, {})


def test_sort_by_string(sample_df):
    result = sort.step(sample_df, {"by": "name"})
    assert list(result["name"]) == sorted(sample_df["name"])


# ---------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------

def test_head_n_rows(sample_df):
    result = head.step(sample_df, {"n": 3})
    assert len(result) == 3
    assert list(result["id"]) == [1, 2, 3]


def test_head_with_order_by(sample_df):
    result = head.step(sample_df, {"n": 2, "order_by": "score", "ascending": False})
    # top 2 scores: 95.0 (Diana), 92.0 (Bob)
    assert len(result) == 2
    assert result["score"].iloc[0] == 95.0


def test_head_default_n(sample_df):
    # sample_df has 5 rows, default n=10 => all rows
    result = head.step(sample_df, {})
    assert len(result) == 5


# ---------------------------------------------------------------------------
# distinct
# ---------------------------------------------------------------------------

def test_distinct_all_columns(sample_df):
    # All rows already unique
    result = distinct.step(sample_df, {})
    assert len(result) == len(sample_df)


def test_distinct_subset():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
    result = distinct.step(df, {"columns": ["a"]})
    assert len(result) == 2
    assert list(result["a"]) == [1, 2]


def test_distinct_with_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": [10, 20, 30, 30]})
    result = distinct.step(df, {})
    # All rows unique since (1,10), (1,20), (2,30) are unique except last two
    assert len(result) == 3


def test_distinct_keep_last():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    result = distinct.step(df, {"columns": ["a"], "keep": "last"})
    assert result[result["a"] == 1]["b"].iloc[0] == 20
