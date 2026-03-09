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
