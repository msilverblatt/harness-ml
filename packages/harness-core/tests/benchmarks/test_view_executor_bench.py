"""Performance benchmarks comparing Polars vs pandas view executor backends.

Run with: uv run pytest packages/harness-core/tests/benchmarks/ -v
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import polars as pl
import pytest
from harnessml.core.runner.schema import (
    DeriveStep,
    FilterStep,
    GroupByStep,
    JoinStep,
    RankStep,
    RollingStep,
    SortStep,
)
from harnessml.core.runner.views.executor import _execute_step_pandas
from harnessml.core.runner.views.executor_polars import execute_step as polars_execute_step
from harnessml.core.runner.views.polars_compat import to_lazy

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def large_pdf():
    """100k-row pandas DataFrame for benchmarking."""
    rng = np.random.default_rng(42)
    n = 100_000
    return pd.DataFrame(
        {
            "id": range(n),
            "value": rng.standard_normal(n),
            "category": rng.choice(["a", "b", "c", "d"], n),
            "season": rng.choice(range(2015, 2025), n),
            "order": range(n),
        }
    )


@pytest.fixture
def large_lf(large_pdf):
    """Same data as LazyFrame."""
    return to_lazy(large_pdf)


def _time_fn(fn, iterations=3):
    """Run fn multiple times and return best time."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


# ---------------------------------------------------------------------------
# Benchmark: filter
# ---------------------------------------------------------------------------


class TestFilterBench:
    def test_polars_filter(self, large_lf):
        step = {"op": "filter", "expr": "value > 0"}
        t = _time_fn(lambda: polars_execute_step(large_lf, step).collect())
        print(f"\n  Polars filter: {t:.4f}s")

    def test_pandas_filter(self, large_pdf):
        step = FilterStep(expr="value > 0")
        t = _time_fn(lambda: _execute_step_pandas(large_pdf, step))
        print(f"\n  Pandas filter: {t:.4f}s")


# ---------------------------------------------------------------------------
# Benchmark: group_by
# ---------------------------------------------------------------------------


class TestGroupByBench:
    def test_polars_group_by(self, large_lf):
        step = {
            "op": "group_by",
            "keys": ["category", "season"],
            "aggs": {"value": ["mean", "std", "min", "max"]},
        }
        t = _time_fn(lambda: polars_execute_step(large_lf, step).collect())
        print(f"\n  Polars group_by: {t:.4f}s")

    def test_pandas_group_by(self, large_pdf):
        step = GroupByStep(
            keys=["category", "season"],
            aggs={"value": ["mean", "std", "min", "max"]},
        )
        t = _time_fn(lambda: _execute_step_pandas(large_pdf, step))
        print(f"\n  Pandas group_by: {t:.4f}s")


# ---------------------------------------------------------------------------
# Benchmark: derive
# ---------------------------------------------------------------------------


class TestDeriveBench:
    def test_polars_derive(self, large_lf):
        step = {"op": "derive", "columns": {"doubled": "value * 2", "shifted": "value + 10"}}
        t = _time_fn(lambda: polars_execute_step(large_lf, step).collect())
        print(f"\n  Polars derive: {t:.4f}s")

    def test_pandas_derive(self, large_pdf):
        step = DeriveStep(columns={"doubled": "value * 2", "shifted": "value + 10"})
        t = _time_fn(lambda: _execute_step_pandas(large_pdf, step))
        print(f"\n  Pandas derive: {t:.4f}s")


# ---------------------------------------------------------------------------
# Benchmark: sort
# ---------------------------------------------------------------------------


class TestSortBench:
    def test_polars_sort(self, large_lf):
        step = {"op": "sort", "by": ["value"], "ascending": False}
        t = _time_fn(lambda: polars_execute_step(large_lf, step).collect())
        print(f"\n  Polars sort: {t:.4f}s")

    def test_pandas_sort(self, large_pdf):
        step = SortStep(by=["value"], ascending=False)
        t = _time_fn(lambda: _execute_step_pandas(large_pdf, step))
        print(f"\n  Pandas sort: {t:.4f}s")


# ---------------------------------------------------------------------------
# Benchmark: rolling
# ---------------------------------------------------------------------------


class TestRollingBench:
    def test_polars_rolling(self, large_lf):
        step = {
            "op": "rolling",
            "keys": ["category"],
            "order_by": "order",
            "window": 10,
            "aggs": {"val_mean_10": "value:mean"},
        }
        t = _time_fn(lambda: polars_execute_step(large_lf, step).collect())
        print(f"\n  Polars rolling mean: {t:.4f}s")

    def test_pandas_rolling(self, large_pdf):
        step = RollingStep(
            keys=["category"],
            order_by="order",
            window=10,
            aggs={"val_mean_10": "value:mean"},
        )
        t = _time_fn(lambda: _execute_step_pandas(large_pdf, step))
        print(f"\n  Pandas rolling mean: {t:.4f}s")


# ---------------------------------------------------------------------------
# Benchmark: rank
# ---------------------------------------------------------------------------


class TestRankBench:
    def test_polars_rank(self, large_lf):
        step = {
            "op": "rank",
            "columns": {"value_rank": "value"},
            "keys": ["category"],
            "ascending": False,
        }
        t = _time_fn(lambda: polars_execute_step(large_lf, step).collect())
        print(f"\n  Polars rank: {t:.4f}s")

    def test_pandas_rank(self, large_pdf):
        step = RankStep(
            columns={"value_rank": "value"},
            keys=["category"],
            ascending=False,
        )
        t = _time_fn(lambda: _execute_step_pandas(large_pdf, step))
        print(f"\n  Pandas rank: {t:.4f}s")


# ---------------------------------------------------------------------------
# Benchmark: join
# ---------------------------------------------------------------------------


class TestJoinBench:
    def test_polars_join(self, large_lf):
        right = pl.LazyFrame(
            {
                "category": ["a", "b", "c", "d"],
                "label": ["Alpha", "Beta", "Charlie", "Delta"],
            }
        )
        step = {
            "op": "join",
            "other": "right",
            "on": ["category"],
            "how": "left",
        }
        t = _time_fn(
            lambda: polars_execute_step(large_lf, step, context={"right": right}).collect()
        )
        print(f"\n  Polars join: {t:.4f}s")

    def test_pandas_join(self, large_pdf):
        right_pdf = pd.DataFrame(
            {
                "category": ["a", "b", "c", "d"],
                "label": ["Alpha", "Beta", "Charlie", "Delta"],
            }
        )
        step = JoinStep(other="right", on=["category"], how="left")

        def resolver(name):
            return right_pdf

        t = _time_fn(lambda: _execute_step_pandas(large_pdf, step, resolver))
        print(f"\n  Pandas join: {t:.4f}s")


# ---------------------------------------------------------------------------
# Multi-step pipeline benchmark
# ---------------------------------------------------------------------------


class TestMultiStepBench:
    def test_polars_pipeline(self, large_lf):
        """Multiple chained steps -- Polars can optimize the full query plan."""
        steps = [
            {"op": "filter", "expr": "value > -1"},
            {"op": "derive", "columns": {"abs_value": "ABS(value)"}},
            {"op": "sort", "by": ["category", "abs_value"], "ascending": [True, False]},
        ]

        def run():
            lf = large_lf
            for s in steps:
                lf = polars_execute_step(lf, s)
            lf.collect()

        t = _time_fn(run)
        print(f"\n  Polars 3-step pipeline: {t:.4f}s")

    def test_pandas_pipeline(self, large_pdf):
        steps = [
            FilterStep(expr="value > -1"),
            DeriveStep(columns={"abs_value": "value.abs()"}),
            SortStep(by=["category", "abs_value"], ascending=[True, False]),
        ]

        def run():
            df = large_pdf
            for s in steps:
                df = _execute_step_pandas(df, s)

        t = _time_fn(run)
        print(f"\n  Pandas 3-step pipeline: {t:.4f}s")
