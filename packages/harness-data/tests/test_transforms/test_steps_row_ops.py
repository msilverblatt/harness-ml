"""Tests for row operation steps: rank, isin, null_indicator."""
from __future__ import annotations
import pytest
import pandas as pd
import numpy as np
from harness.data.transforms.steps import rank, isin, null_indicator


# ---------------------------------------------------------------------------
# rank
# ---------------------------------------------------------------------------

def test_rank_basic(numeric_df):
    result = rank.step(numeric_df, {"columns": {"points_rank": "points"}})
    assert "points_rank" in result.columns
    assert result["points_rank"].notna().all()
    # rank of min value should be 1.0 (ascending average)
    min_idx = numeric_df["points"].idxmin()
    assert result["points_rank"].iloc[min_idx] == 1.0


def test_rank_with_pct(numeric_df):
    result = rank.step(numeric_df, {"columns": {"points_pct": "points"}, "pct": True})
    assert "points_pct" in result.columns
    assert result["points_pct"].between(0, 1).all()


def test_rank_missing_columns_param(numeric_df):
    with pytest.raises(ValueError, match="rank step requires"):
        rank.step(numeric_df, {})


def test_rank_missing_source_col(numeric_df):
    with pytest.raises(ValueError, match="Column not found"):
        rank.step(numeric_df, {"columns": {"r": "nonexistent"}})


def test_rank_partitioned(numeric_df):
    result = rank.step(numeric_df, {
        "columns": {"points_rank": "points"},
        "keys": ["entity_id"],
    })
    assert "points_rank" in result.columns
    # within each entity group of 3 rows, ranks should be 1, 2, 3
    for entity_id, grp in result.groupby("entity_id"):
        assert sorted(grp["points_rank"].tolist()) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# isin
# ---------------------------------------------------------------------------

def test_isin_include(sample_df):
    result = isin.step(sample_df, {"column": "grade", "values": ["A"]})
    assert len(result) == 2
    assert set(result["grade"]) == {"A"}


def test_isin_negate(sample_df):
    result = isin.step(sample_df, {"column": "grade", "values": ["A"], "negate": True})
    assert "A" not in result["grade"].values
    assert len(result) == 3


def test_isin_multiple_values(sample_df):
    result = isin.step(sample_df, {"column": "grade", "values": ["A", "B"]})
    assert set(result["grade"]).issubset({"A", "B"})


def test_isin_missing_column_param(sample_df):
    with pytest.raises(ValueError, match="isin step requires"):
        isin.step(sample_df, {"values": ["A"]})


def test_isin_missing_values_param(sample_df):
    with pytest.raises(ValueError, match="isin step requires"):
        isin.step(sample_df, {"column": "grade"})


def test_isin_resets_index(sample_df):
    result = isin.step(sample_df, {"column": "grade", "values": ["A"]})
    assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# null_indicator
# ---------------------------------------------------------------------------

def test_null_indicator_creates_columns():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, 2.0, None]})
    result = null_indicator.step(df, {"columns": ["a", "b"]})
    assert "missing_a" in result.columns
    assert "missing_b" in result.columns
    assert list(result["missing_a"]) == [0, 1, 0]
    assert list(result["missing_b"]) == [1, 0, 1]


def test_null_indicator_custom_prefix():
    df = pd.DataFrame({"x": [1.0, None]})
    result = null_indicator.step(df, {"columns": ["x"], "prefix": "is_null_"})
    assert "is_null_x" in result.columns


def test_null_indicator_preserves_original(sample_df):
    df = sample_df.copy()
    df["score"] = df["score"].where(df["score"] > 85)
    result = null_indicator.step(df, {"columns": ["score"]})
    assert "score" in result.columns  # original column still present
    assert "missing_score" in result.columns


def test_null_indicator_missing_column(sample_df):
    with pytest.raises(ValueError, match="Column not found"):
        null_indicator.step(sample_df, {"columns": ["nonexistent"]})


def test_null_indicator_missing_param(sample_df):
    with pytest.raises(ValueError, match="null_indicator step requires"):
        null_indicator.step(sample_df, {})
