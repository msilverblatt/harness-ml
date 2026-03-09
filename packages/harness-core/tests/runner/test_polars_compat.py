import numpy as np
import pandas as pd
import polars as pl
from harnessml.core.runner.views.polars_compat import to_lazy, to_pandas


def test_pandas_to_lazy_roundtrip():
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    lf = to_lazy(pdf)
    assert isinstance(lf, pl.LazyFrame)
    result = to_pandas(lf)
    pd.testing.assert_frame_equal(pdf, result)


def test_lazy_to_lazy_noop():
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    assert to_lazy(lf) is lf


def test_polars_df_to_lazy():
    df = pl.DataFrame({"a": [1, 2, 3]})
    lf = to_lazy(df)
    assert isinstance(lf, pl.LazyFrame)


def test_pandas_to_pandas_noop():
    pdf = pd.DataFrame({"a": [1, 2, 3]})
    result = to_pandas(pdf)
    assert result is pdf


def test_polars_df_to_pandas():
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = to_pandas(df)
    assert isinstance(result, pd.DataFrame)
    assert result["a"].tolist() == [1, 2, 3]


def test_handles_nullable_integers():
    pdf = pd.DataFrame({"a": pd.array([1, None, 3], dtype=pd.Int64Dtype())})
    lf = to_lazy(pdf)
    result = to_pandas(lf)
    assert result["a"].isna().sum() == 1


def test_to_lazy_bad_type():
    import pytest
    with pytest.raises(TypeError):
        to_lazy("not a dataframe")


def test_to_pandas_bad_type():
    import pytest
    with pytest.raises(TypeError):
        to_pandas("not a dataframe")
