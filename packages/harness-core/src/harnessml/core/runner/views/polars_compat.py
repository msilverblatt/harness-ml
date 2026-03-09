"""Polars <-> Pandas conversion utilities for the view executor migration."""
from __future__ import annotations

import pandas as pd
import polars as pl


def to_lazy(df: pd.DataFrame | pl.LazyFrame | pl.DataFrame) -> pl.LazyFrame:
    """Convert any DataFrame type to a Polars LazyFrame."""
    if isinstance(df, pl.LazyFrame):
        return df
    if isinstance(df, pl.DataFrame):
        return df.lazy()
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df).lazy()
    raise TypeError(f"Cannot convert {type(df)} to LazyFrame")


def to_pandas(lf: pl.LazyFrame | pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    """Materialize a LazyFrame to pandas DataFrame."""
    if isinstance(lf, pd.DataFrame):
        return lf
    if isinstance(lf, pl.LazyFrame):
        return lf.collect().to_pandas()
    if isinstance(lf, pl.DataFrame):
        return lf.to_pandas()
    raise TypeError(f"Cannot convert {type(lf)} to pandas DataFrame")
