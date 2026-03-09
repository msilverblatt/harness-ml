import polars as pl
import pytest
from harnessml.core.runner.view_executor_polars import execute_step


@pytest.fixture
def sample_lf():
    return pl.LazyFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "category": ["a", "b", "a", "b", "a"],
            "season": [2020, 2020, 2021, 2021, 2022],
        }
    )


def test_filter_step(sample_lf):
    step = {"op": "filter", "expr": "value > 20"}
    result = execute_step(sample_lf, step).collect()
    assert len(result) == 3


def test_filter_step_equality(sample_lf):
    step = {"op": "filter", "expr": "category = 'a'"}
    result = execute_step(sample_lf, step).collect()
    assert len(result) == 3


def test_select_step_list(sample_lf):
    step = {"op": "select", "columns": ["id", "value"]}
    result = execute_step(sample_lf, step).collect()
    assert result.columns == ["id", "value"]


def test_select_step_rename(sample_lf):
    step = {"op": "select", "columns": {"identifier": "id", "val": "value"}}
    result = execute_step(sample_lf, step).collect()
    assert "identifier" in result.columns
    assert "val" in result.columns


def test_derive_step(sample_lf):
    step = {"op": "derive", "columns": {"doubled": "value * 2"}}
    result = execute_step(sample_lf, step).collect()
    assert result["doubled"].to_list() == [20.0, 40.0, 60.0, 80.0, 100.0]


def test_sort_step(sample_lf):
    step = {"op": "sort", "by": ["value"], "ascending": False}
    result = execute_step(sample_lf, step).collect()
    assert result["value"].to_list() == [50.0, 40.0, 30.0, 20.0, 10.0]


def test_sort_step_ascending(sample_lf):
    step = {"op": "sort", "by": ["value"], "ascending": True}
    result = execute_step(sample_lf, step).collect()
    assert result["value"].to_list() == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_distinct_step(sample_lf):
    step = {"op": "distinct", "columns": ["category"]}
    result = execute_step(sample_lf, step).collect()
    assert len(result) == 2  # "a" and "b"


def test_distinct_all_columns(sample_lf):
    step = {"op": "distinct"}
    result = execute_step(sample_lf, step).collect()
    assert len(result) == 5  # all unique


def test_head_step(sample_lf):
    step = {
        "op": "head",
        "keys": ["category"],
        "n": 1,
        "order_by": "value",
        "ascending": True,
    }
    result = execute_step(sample_lf, step).collect()
    assert len(result) == 2  # One per category


def test_head_step_no_order(sample_lf):
    step = {"op": "head", "keys": ["category"], "n": 2}
    result = execute_step(sample_lf, step).collect()
    assert len(result) == 4  # 2 per category (a has 3, b has 2, so a=2, b=2)


def test_unknown_step(sample_lf):
    step = {"op": "nonexistent"}
    with pytest.raises(ValueError, match="Unknown view step"):
        execute_step(sample_lf, step)


# ---------------------------------------------------------------------------
# Aggregation steps
# ---------------------------------------------------------------------------


def test_group_by_step(sample_lf):
    step = {"op": "group_by", "keys": ["category"], "aggs": {"value": "mean"}}
    result = execute_step(sample_lf, step).collect()
    assert len(result) == 2


def test_group_by_multi_agg(sample_lf):
    step = {
        "op": "group_by",
        "keys": ["category"],
        "aggs": {"value": ["mean", "sum"]},
    }
    result = execute_step(sample_lf, step).collect()
    assert "value_mean" in result.columns
    assert "value_sum" in result.columns


def test_rolling_step():
    lf = pl.LazyFrame(
        {
            "team": ["A"] * 5,
            "game": list(range(5)),
            "points": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    step = {
        "op": "rolling",
        "keys": ["team"],
        "order_by": "game",
        "window": 3,
        "aggs": {"avg_pts_3": "points:mean"},
    }
    result = execute_step(lf, step).collect()
    assert "avg_pts_3" in result.columns


def test_rank_step(sample_lf):
    step = {"op": "rank", "columns": {"value_rank": "value"}, "ascending": False}
    result = execute_step(sample_lf, step).collect()
    assert "value_rank" in result.columns
    # Highest value (50) should get rank 1 when descending
    row_50 = result.filter(pl.col("value") == 50.0)
    assert row_50["value_rank"][0] == 1.0


def test_rank_step_with_keys(sample_lf):
    step = {
        "op": "rank",
        "columns": {"value_rank": "value"},
        "keys": ["category"],
        "ascending": True,
    }
    result = execute_step(sample_lf, step).collect()
    assert "value_rank" in result.columns


def test_cond_agg_step():
    lf = pl.LazyFrame(
        {
            "team": ["A", "A", "A", "B", "B"],
            "result": [1, 0, 1, 1, 0],
            "points": [10, 20, 30, 40, 50],
        }
    )
    step = {
        "op": "cond_agg",
        "keys": ["team"],
        "aggs": {
            "win_avg_pts": "points:mean:result = 1",
            "total_games": "points:count",
        },
    }
    result = execute_step(lf, step).collect()
    assert "win_avg_pts" in result.columns
    assert "total_games" in result.columns


def test_ewm_step():
    lf = pl.LazyFrame(
        {
            "team": ["A"] * 5,
            "game": list(range(5)),
            "points": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    step = {
        "op": "ewm",
        "keys": ["team"],
        "order_by": "game",
        "span": 3,
        "aggs": {"ewm_pts": "points:mean"},
    }
    result = execute_step(lf, step).collect()
    assert "ewm_pts" in result.columns
