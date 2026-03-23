"""Tests for DataProfiler."""
import numpy as np
import pandas as pd
import pytest

from harness.data.profiling.profiler import DataProfiler, DataProfile, ColumnProfile


@pytest.fixture
def profiler():
    return DataProfiler()


@pytest.fixture
def basic_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "score": [10.0, 20.0, 30.0, 40.0, 50.0],
        "label": ["a", "b", "a", "c", "b"],
        "flag": [True, False, True, True, False],
    })


def test_basic_profile_shape(profiler, basic_df):
    result = profiler.profile(basic_df)
    assert isinstance(result, DataProfile)
    assert result.row_count == 5
    assert result.column_count == 4
    assert len(result.columns) == 4


def test_column_profile_types_returned(profiler, basic_df):
    result = profiler.profile(basic_df)
    names = [c.name for c in result.columns]
    assert "id" in names
    assert "score" in names
    assert "label" in names
    assert "flag" in names
    for cp in result.columns:
        assert isinstance(cp, ColumnProfile)


def test_null_count_and_pct(profiler):
    df = pd.DataFrame({
        "a": [1.0, np.nan, 3.0, np.nan, 5.0],
    })
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.null_count == 2
    assert cp.null_pct == pytest.approx(40.0)


def test_numeric_stats(profiler):
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.mean == pytest.approx(3.0)
    assert cp.min == pytest.approx(1.0)
    assert cp.max == pytest.approx(5.0)
    assert cp.median == pytest.approx(3.0)
    assert cp.std is not None


def test_categorical_inference_object(profiler):
    df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b"] * 2})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.inferred_type == "categorical"


def test_high_cardinality_inference(profiler):
    # >50 unique object values, ratio <= 90%: high_cardinality
    labels = [f"val_{i}" for i in range(60)] + ["dup"] * 40
    df = pd.DataFrame({"col": labels})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.inferred_type == "high_cardinality"


def test_id_inference(profiler):
    # >90% unique object values → "id"
    labels = [f"id_{i}" for i in range(100)]
    df = pd.DataFrame({"col": labels})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.inferred_type == "id"


def test_boolean_detection(profiler):
    df = pd.DataFrame({"b": pd.array([True, False, True], dtype=bool)})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.inferred_type == "boolean"


def test_binary_numeric(profiler):
    df = pd.DataFrame({"bin": [0, 1, 0, 1, 1]})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.inferred_type == "binary"


def test_numeric_categorical(profiler):
    df = pd.DataFrame({"cat": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.inferred_type == "categorical"


def test_numeric_inferred_type(profiler):
    # >20 unique numeric values
    df = pd.DataFrame({"v": list(range(25))})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.inferred_type == "numeric"


def test_datetime_inference(profiler):
    df = pd.DataFrame({"dt": pd.date_range("2020-01-01", periods=5)})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.inferred_type == "datetime"


def test_high_null_detection(profiler):
    df = pd.DataFrame({
        "ok": [1, 2, 3, 4, 5],
        "mostly_null": [np.nan, np.nan, np.nan, 4.0, np.nan],
    })
    result = profiler.profile(df)
    assert "mostly_null" in result.high_null_columns
    assert "ok" not in result.high_null_columns


def test_high_null_threshold_customizable(profiler):
    df = pd.DataFrame({
        "some_null": [np.nan, 2.0, 3.0, 4.0, 5.0],  # 20% null
    })
    result_default = profiler.profile(df)
    result_strict = profiler.profile(df, high_null_threshold=10.0)
    assert "some_null" not in result_default.high_null_columns
    assert "some_null" in result_strict.high_null_columns


def test_zero_variance_detection(profiler):
    df = pd.DataFrame({
        "const": [5.0, 5.0, 5.0, 5.0, 5.0],
        "varied": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    result = profiler.profile(df)
    assert "const" in result.zero_variance_columns
    assert "varied" not in result.zero_variance_columns


def test_no_numeric_stats_for_object(profiler):
    df = pd.DataFrame({"name": ["Alice", "Bob", "Carol"]})
    result = profiler.profile(df)
    cp = result.columns[0]
    assert cp.mean is None
    assert cp.std is None
    assert cp.min is None
    assert cp.max is None
    assert cp.median is None
