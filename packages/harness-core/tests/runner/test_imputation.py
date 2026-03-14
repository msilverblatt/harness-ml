import numpy as np
import pandas as pd
from harnessml.core.runner.training.preprocessing import Preprocessor


def test_knn_imputation():
    df = pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    pp = Preprocessor(missing_strategy="knn", knn_neighbors=2)
    result = pp.fit_transform(df)
    assert not result.isna().any().any()
    assert abs(result["x"].iloc[2] - 3.0) < 1.5


def test_iterative_imputation():
    df = pd.DataFrame({
        "x": [1.0, 2.0, np.nan, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    pp = Preprocessor(missing_strategy="iterative")
    result = pp.fit_transform(df)
    assert not result.isna().any().any()


def test_knn_imputation_transform_separate():
    train = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    test = pd.DataFrame({
        "x": [np.nan, 3.0],
        "y": [5.0, np.nan],
    })
    pp = Preprocessor(missing_strategy="knn", knn_neighbors=2)
    pp.fit_transform(train)
    result = pp.transform(test)
    assert not result.isna().any().any()


def test_iterative_transform_separate():
    train = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0],
    })
    test = pd.DataFrame({
        "x": [np.nan, 3.0],
        "y": [5.0, np.nan],
    })
    pp = Preprocessor(missing_strategy="iterative")
    pp.fit_transform(train)
    result = pp.transform(test)
    assert not result.isna().any().any()
