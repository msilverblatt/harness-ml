"""Tests for ApiSource adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from harness.data.sources.protocol import SourceConfig
from harness.data.sources.api import ApiSource


def _make_response(json_data, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


class TestApiSourceLoad:
    def test_load_list_of_records(self):
        data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        resp = _make_response(data)
        with patch("harness.data.sources.api.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = ApiSource()
            config = SourceConfig(name="test", source_type="api", url="http://api.example.com/users")
            df = source.load(config)
        assert len(df) == 2
        assert list(df.columns) == ["id", "name"]
        mock_requests.get.assert_called_once_with("http://api.example.com/users", params={})

    def test_load_nested_records_key(self):
        data = {"total": 2, "results": [{"id": 1, "val": 10}, {"id": 2, "val": 20}]}
        resp = _make_response(data)
        with patch("harness.data.sources.api.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = ApiSource()
            config = SourceConfig(
                name="test",
                source_type="api",
                url="http://api.example.com/data",
                params={"records_key": "results"},
            )
            df = source.load(config)
        assert len(df) == 2
        assert "val" in df.columns

    def test_load_passes_query_params(self):
        data = [{"x": 1}]
        resp = _make_response(data)
        with patch("harness.data.sources.api.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = ApiSource()
            config = SourceConfig(
                name="test",
                source_type="api",
                url="http://api.example.com/data",
                params={"page": 1, "limit": 50},
            )
            df = source.load(config)
        call_kwargs = mock_requests.get.call_args
        assert call_kwargs[1]["params"] == {"page": 1, "limit": 50}

    def test_load_passes_query_params_without_records_key(self):
        data = [{"x": 1}]
        resp = _make_response(data)
        with patch("harness.data.sources.api.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = ApiSource()
            config = SourceConfig(
                name="test",
                source_type="api",
                url="http://api.example.com/data",
                params={"records_key": "items", "page": 2},
            )
            # records_key absent from actual response — should not blow up on key extraction
            # override response to have "items"
            resp2 = _make_response({"items": [{"x": 1}, {"x": 2}]})
            mock_requests.get.return_value = resp2
            df = source.load(config)
        # records_key should be stripped from the GET params
        call_kwargs = mock_requests.get.call_args
        sent_params = call_kwargs[1]["params"]
        assert "records_key" not in sent_params
        assert sent_params.get("page") == 2

    def test_raises_on_missing_url(self):
        source = ApiSource()
        config = SourceConfig(name="test", source_type="api", url=None)
        with pytest.raises(ValueError, match="url"):
            source.load(config)

    def test_raises_on_http_error(self):
        with patch("harness.data.sources.api.requests") as mock_requests:
            resp = MagicMock()
            resp.raise_for_status.side_effect = Exception("500 Server Error")
            mock_requests.get.return_value = resp
            source = ApiSource()
            config = SourceConfig(name="test", source_type="api", url="http://api.example.com/data")
            with pytest.raises(Exception):
                source.load(config)

    def test_raises_when_records_key_missing_from_response(self):
        data = {"total": 0}
        resp = _make_response(data)
        with patch("harness.data.sources.api.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = ApiSource()
            config = SourceConfig(
                name="test",
                source_type="api",
                url="http://api.example.com/data",
                params={"records_key": "results"},
            )
            with pytest.raises(KeyError):
                source.load(config)

    def test_returns_dataframe(self):
        data = [{"a": 1}]
        resp = _make_response(data)
        with patch("harness.data.sources.api.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = ApiSource()
            config = SourceConfig(name="test", source_type="api", url="http://api.example.com/data")
            result = source.load(config)
        assert isinstance(result, pd.DataFrame)


class TestApiSourceValidate:
    def test_validate_missing_url(self):
        source = ApiSource()
        config = SourceConfig(name="test", source_type="api", url=None)
        errors = source.validate(config)
        assert len(errors) > 0
        assert "url" in errors[0].lower()

    def test_validate_valid_config(self):
        source = ApiSource()
        config = SourceConfig(name="test", source_type="api", url="http://api.example.com/data")
        errors = source.validate(config)
        assert errors == []


class TestApiSourceRefresh:
    def test_refresh_delegates_to_load(self):
        data = [{"k": "v"}]
        resp = _make_response(data)
        with patch("harness.data.sources.api.requests") as mock_requests:
            mock_requests.get.return_value = resp
            source = ApiSource()
            config = SourceConfig(name="test", source_type="api", url="http://api.example.com/data")
            df = source.refresh(config)
        assert len(df) == 1


class TestApiSourceProtocol:
    def test_implements_source_protocol(self):
        from harness.data.sources.protocol import Source
        assert isinstance(ApiSource(), Source)
