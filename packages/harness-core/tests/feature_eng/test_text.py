import pandas as pd
import pytest
from harnessml.core.feature_eng.text import extract_text_features


def test_tfidf_features():
    df = pd.DataFrame({"text": ["hello world", "foo bar baz", "hello foo"]})
    features = extract_text_features(df, "text", method="tfidf", max_features=5)
    assert features.shape[0] == 3
    assert features.shape[1] == 5


def test_count_features():
    df = pd.DataFrame({"text": ["hello world", "foo bar baz", "hello foo"]})
    features = extract_text_features(df, "text", method="count", max_features=3)
    assert features.shape[0] == 3
    assert features.shape[1] == 3


def test_unknown_method():
    df = pd.DataFrame({"text": ["hello"]})
    with pytest.raises(ValueError, match="Unknown"):
        extract_text_features(df, "text", method="bogus")


def test_handles_nan():
    df = pd.DataFrame({"text": ["hello", None, "world"]})
    features = extract_text_features(df, "text", method="tfidf", max_features=5)
    assert features.shape[0] == 3
    assert not features.isna().any().any()
