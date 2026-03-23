"""Tests for UrlSource adapter."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from harness.data.sources.protocol import SourceConfig
from harness.data.sources.url import UrlSource


def _make_response(content: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = content.decode("latin-1")
    resp.raise_for_status = MagicMock()
    return resp


class TestUrlSourceLoad:
    def test_load_csv(self):
        csv_bytes = b"id,name,score\n1,Alice,85\n2,Bob,92\n"
        resp = _make_response(csv_bytes)
        with patch("harness.data.sources.url.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = UrlSource()
            config = SourceConfig(name="test", source_type="url", url="http://example.com/data.csv")
            df = source.load(config)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name", "score"]
        mock_requests.get.assert_called_once_with("http://example.com/data.csv")

    def test_load_json(self):
        json_bytes = b'[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]'
        resp = _make_response(json_bytes)
        with patch("harness.data.sources.url.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = UrlSource()
            config = SourceConfig(name="test", source_type="url", url="http://example.com/data.json")
            df = source.load(config)
        assert len(df) == 2
        assert "id" in df.columns

    def test_load_parquet(self, sample_df):
        buf = io.BytesIO()
        sample_df.to_parquet(buf, index=False)
        parquet_bytes = buf.getvalue()
        resp = _make_response(parquet_bytes)
        with patch("harness.data.sources.url.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = UrlSource()
            config = SourceConfig(name="test", source_type="url", url="http://example.com/data.parquet")
            df = source.load(config)
        assert len(df) == 5
        assert "score" in df.columns

    def test_raises_on_missing_url(self):
        source = UrlSource()
        config = SourceConfig(name="test", source_type="url", url=None)
        with pytest.raises(ValueError, match="url"):
            source.load(config)

    def test_raises_on_unsupported_format(self):
        resp = _make_response(b"data")
        with patch("harness.data.sources.url.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = UrlSource()
            config = SourceConfig(name="test", source_type="url", url="http://example.com/data.xyz")
            with pytest.raises(ValueError, match="[Uu]nsupported"):
                source.load(config)

    def test_explicit_format_overrides_extension(self):
        csv_bytes = b"a,b\n1,2\n3,4\n"
        resp = _make_response(csv_bytes)
        with patch("harness.data.sources.url.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = UrlSource()
            config = SourceConfig(
                name="test",
                source_type="url",
                url="http://example.com/data",
                format="csv",
            )
            df = source.load(config)
        assert len(df) == 2

    def test_raises_on_http_error(self):
        with patch("harness.data.sources.url.requests") as mock_requests:
            resp = MagicMock()
            resp.raise_for_status.side_effect = Exception("404 Not Found")
            mock_requests.get.return_value = resp
            source = UrlSource()
            config = SourceConfig(name="test", source_type="url", url="http://example.com/data.csv")
            with pytest.raises(Exception):
                source.load(config)


class TestUrlSourceValidate:
    def test_validate_missing_url(self):
        source = UrlSource()
        config = SourceConfig(name="test", source_type="url", url=None)
        errors = source.validate(config)
        assert len(errors) > 0
        assert "url" in errors[0].lower()

    def test_validate_valid_config(self):
        source = UrlSource()
        config = SourceConfig(name="test", source_type="url", url="http://example.com/data.csv")
        errors = source.validate(config)
        assert errors == []


class TestUrlSourceRefresh:
    def test_refresh_delegates_to_load(self):
        csv_bytes = b"x,y\n1,2\n"
        resp = _make_response(csv_bytes)
        with patch("harness.data.sources.url.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = UrlSource()
            config = SourceConfig(name="test", source_type="url", url="http://example.com/data.csv")
            df = source.refresh(config)
        assert len(df) == 1


class TestUrlSourceProtocol:
    def test_implements_source_protocol(self):
        from harness.data.sources.protocol import Source
        assert isinstance(UrlSource(), Source)
