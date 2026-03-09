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


# ---------------------------------------------------------------------------
# Join + Union steps
# ---------------------------------------------------------------------------


def test_join_step():
    left = pl.LazyFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    right = pl.LazyFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    step = {"op": "join", "other": "right_table", "on": ["id"], "how": "left"}
    result = execute_step(left, step, context={"right_table": right}).collect()
    assert "name" in result.columns
    assert len(result) == 3


def test_join_step_dict_keys():
    left = pl.LazyFrame({"team_id": [1, 2, 3], "score": [10, 20, 30]})
    right = pl.LazyFrame({"tid": [1, 2, 3], "name": ["a", "b", "c"]})
    step = {
        "op": "join",
        "other": "right_table",
        "on": {"team_id": "tid"},
        "how": "left",
    }
    result = execute_step(left, step, context={"right_table": right}).collect()
    assert "name" in result.columns


def test_join_step_with_prefix():
    left = pl.LazyFrame({"id": [1, 2], "value": [10, 20]})
    right = pl.LazyFrame({"id": [1, 2], "value": [100, 200]})
    step = {
        "op": "join",
        "other": "right_table",
        "on": ["id"],
        "how": "left",
        "prefix": "other_",
    }
    result = execute_step(left, step, context={"right_table": right}).collect()
    assert "other_value" in result.columns


def test_join_no_context():
    lf = pl.LazyFrame({"id": [1]})
    step = {"op": "join", "other": "x", "on": ["id"], "how": "left"}
    with pytest.raises(ValueError, match="context is required"):
        execute_step(lf, step)


def test_union_step():
    top = pl.LazyFrame({"id": [1, 2], "val": [10, 20]})
    bottom = pl.LazyFrame({"id": [3, 4], "val": [30, 40]})
    step = {"op": "union", "other": "bottom_table"}
    result = execute_step(top, step, context={"bottom_table": bottom}).collect()
    assert len(result) == 4


def test_union_no_context():
    lf = pl.LazyFrame({"id": [1]})
    step = {"op": "union", "other": "x"}
    with pytest.raises(ValueError, match="context is required"):
        execute_step(lf, step)


# ---------------------------------------------------------------------------
# Remaining steps: lag, diff, trend, encode, bin, datetime, cast, unpivot,
#                  isin, null_indicator
# ---------------------------------------------------------------------------


def test_lag_step():
    lf = pl.LazyFrame(
        {
            "team": ["A"] * 5,
            "game": list(range(5)),
            "pts": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    step = {
        "op": "lag",
        "keys": ["team"],
        "order_by": "game",
        "columns": {"pts_lag1": "pts:1"},
    }
    result = execute_step(lf, step).collect()
    assert "pts_lag1" in result.columns
    vals = result.sort("game")["pts_lag1"].to_list()
    assert vals[0] is None
    assert vals[1] == 10.0


def test_diff_step():
    lf = pl.LazyFrame(
        {
            "team": ["A"] * 5,
            "game": list(range(5)),
            "pts": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    step = {
        "op": "diff",
        "keys": ["team"],
        "order_by": "game",
        "columns": {"pts_diff": "pts:1"},
    }
    result = execute_step(lf, step).collect()
    assert "pts_diff" in result.columns
    vals = result.sort("game")["pts_diff"].to_list()
    assert vals[1] == 10.0  # 20 - 10


def test_diff_pct_step():
    lf = pl.LazyFrame(
        {
            "team": ["A"] * 3,
            "game": [0, 1, 2],
            "pts": [100.0, 110.0, 121.0],
        }
    )
    step = {
        "op": "diff",
        "keys": ["team"],
        "order_by": "game",
        "columns": {"pts_pct": "pts:1"},
        "pct": True,
    }
    result = execute_step(lf, step).collect()
    vals = result.sort("game")["pts_pct"].to_list()
    assert abs(vals[1] - 0.1) < 0.001


def test_trend_step():
    lf = pl.LazyFrame(
        {
            "team": ["A"] * 5,
            "game": list(range(5)),
            "pts": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    step = {
        "op": "trend",
        "keys": ["team"],
        "order_by": "game",
        "window": 3,
        "columns": {"pts_trend": "pts"},
    }
    result = execute_step(lf, step).collect()
    assert "pts_trend" in result.columns
    # Linear data should have constant slope = 10
    vals = result.sort("game")["pts_trend"].to_list()
    non_null = [v for v in vals if v is not None]
    assert len(non_null) > 0
    assert abs(non_null[0] - 10.0) < 0.01


def test_encode_frequency():
    lf = pl.LazyFrame(
        {"cat": ["a", "a", "a", "b", "b"], "val": [1, 2, 3, 4, 5]}
    )
    step = {"op": "encode", "column": "cat", "method": "frequency"}
    result = execute_step(lf, step).collect()
    assert "cat_encoded" in result.columns


def test_encode_ordinal():
    lf = pl.LazyFrame(
        {"cat": ["a", "a", "a", "b", "b"], "val": [1, 2, 3, 4, 5]}
    )
    step = {
        "op": "encode",
        "column": "cat",
        "method": "ordinal",
        "output": "cat_ord",
    }
    result = execute_step(lf, step).collect()
    assert "cat_ord" in result.columns
    # "a" has 3 occurrences, "b" has 2, so "a" should be rank 1
    a_rows = result.filter(pl.col("cat") == "a")
    assert a_rows["cat_ord"][0] == 1


def test_bin_quantile():
    lf = pl.LazyFrame({"val": list(range(100))})
    step = {
        "op": "bin",
        "column": "val",
        "method": "quantile",
        "n_bins": 4,
        "output": "val_q",
    }
    result = execute_step(lf, step).collect()
    assert "val_q" in result.columns


def test_bin_uniform():
    lf = pl.LazyFrame({"val": list(range(100))})
    step = {
        "op": "bin",
        "column": "val",
        "method": "uniform",
        "n_bins": 5,
    }
    result = execute_step(lf, step).collect()
    assert "val_binned" in result.columns


def test_datetime_extract():
    lf = pl.LazyFrame(
        {"dt": ["2020-01-15", "2020-06-20", "2020-12-25"]}
    )
    lf = lf.with_columns(pl.col("dt").str.to_datetime())
    step = {
        "op": "datetime",
        "column": "dt",
        "extract": ["year", "month", "dayofweek"],
    }
    result = execute_step(lf, step).collect()
    assert "dt_year" in result.columns
    assert "dt_month" in result.columns
    assert "dt_dayofweek" in result.columns


def test_datetime_cyclical():
    lf = pl.LazyFrame(
        {"dt": ["2020-01-15", "2020-06-20", "2020-12-25"]}
    )
    lf = lf.with_columns(pl.col("dt").str.to_datetime())
    step = {
        "op": "datetime",
        "column": "dt",
        "cyclical": ["month"],
    }
    result = execute_step(lf, step).collect()
    assert "dt_month_sin" in result.columns
    assert "dt_month_cos" in result.columns


def test_cast_step():
    lf = pl.LazyFrame({"val": [1, 2, 3]})
    step = {"op": "cast", "columns": {"val": "float"}}
    result = execute_step(lf, step).collect()
    assert result["val"].dtype == pl.Float64


def test_unpivot_step():
    lf = pl.LazyFrame(
        {
            "game_id": [1, 2],
            "home_score": [3, 1],
            "away_score": [2, 4],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
        }
    )
    step = {
        "op": "unpivot",
        "id_columns": ["game_id"],
        "unpivot_columns": {
            "score": ["home_score", "away_score"],
            "team": ["home_team", "away_team"],
        },
        "names_column": "side",
        "names_map": {"home_score": "home", "away_score": "away"},
    }
    result = execute_step(lf, step).collect()
    assert len(result) == 4
    assert "score" in result.columns
    assert "team" in result.columns
    assert "side" in result.columns


def test_isin_step():
    lf = pl.LazyFrame({"cat": ["a", "b", "c", "d"]})
    step = {"op": "isin", "column": "cat", "values": ["a", "c"]}
    result = execute_step(lf, step).collect()
    assert len(result) == 2


def test_isin_negate():
    lf = pl.LazyFrame({"cat": ["a", "b", "c", "d"]})
    step = {"op": "isin", "column": "cat", "values": ["a", "c"], "negate": True}
    result = execute_step(lf, step).collect()
    assert len(result) == 2
    assert set(result["cat"].to_list()) == {"b", "d"}


def test_null_indicator_step():
    lf = pl.LazyFrame({"a": [1, None, 3], "b": [None, 5, None]})
    step = {"op": "null_indicator", "columns": ["a", "b"]}
    result = execute_step(lf, step).collect()
    assert "missing_a" in result.columns
    assert "missing_b" in result.columns
    assert result["missing_a"].to_list() == [0, 1, 0]
    assert result["missing_b"].to_list() == [1, 0, 1]
