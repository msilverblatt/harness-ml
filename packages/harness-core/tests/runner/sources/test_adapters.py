"""Tests for built-in data source adapters."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch

import pandas as pd
import pytest
from harnessml.core.runner.sources.adapters import (
    ADAPTERS,
    ApiAdapter,
    ComputedAdapter,
    FileAdapter,
    UrlAdapter,
)

# ---------------------------------------------------------------------------
# FileAdapter
# ---------------------------------------------------------------------------

class TestFileAdapter:
    def test_load_csv(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
        df = FileAdapter.load(str(csv_path))
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_load_parquet(self, tmp_path):
        pq_path = tmp_path / "data.parquet"
        pd.DataFrame({"x": [10, 20]}).to_parquet(pq_path, index=False)
        df = FileAdapter.load(str(pq_path))
        assert list(df.columns) == ["x"]
        assert len(df) == 2

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FileAdapter.load(str(tmp_path / "nope.csv"))

    def test_load_unsupported_format(self, tmp_path):
        bad_path = tmp_path / "data.xyz"
        bad_path.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            FileAdapter.load(str(bad_path))


# ---------------------------------------------------------------------------
# UrlAdapter -- tested with a local HTTP server
# ---------------------------------------------------------------------------

class _CSVHandler(BaseHTTPRequestHandler):
    """Serves a small CSV from /data.csv."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.end_headers()
        self.wfile.write(b"col1,col2\n1,2\n3,4\n")

    def log_message(self, *args):
        pass  # suppress noisy logs in tests


class TestUrlAdapter:
    def test_load_csv_from_url(self):
        server = HTTPServer(("127.0.0.1", 0), _CSVHandler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            df = UrlAdapter.load(f"http://127.0.0.1:{port}/data.csv", format="csv")
            assert list(df.columns) == ["col1", "col2"]
            assert len(df) == 2
        finally:
            server.server_close()

    def test_unsupported_format_raises(self):
        # We don't actually need a real server to test format validation
        # because the error occurs after the data is read. We mock urlopen.
        import io
        mock_response = io.BytesIO(b"some data")
        mock_response.status = 200
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = lambda s, *a: None

        with patch("urllib.request.urlopen", return_value=mock_response):
            with pytest.raises(ValueError, match="Unsupported format"):
                UrlAdapter.load("http://example.com/data", format="yaml")


# ---------------------------------------------------------------------------
# ApiAdapter -- tested with a local HTTP server
# ---------------------------------------------------------------------------

class _JsonHandler(BaseHTTPRequestHandler):
    """Serves a JSON array from any path."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        payload = json.dumps([{"id": 1, "val": "a"}, {"id": 2, "val": "b"}])
        self.wfile.write(payload.encode())

    def log_message(self, *args):
        pass


class _PaginatedJsonHandler(BaseHTTPRequestHandler):
    """Serves paginated JSON: page 1 has 'next' pointing to page 2."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        port = self.server.server_address[1]
        if "page=2" in self.path:
            payload = {"data": [{"id": 3}], "next": None}
        else:
            payload = {"data": [{"id": 1}, {"id": 2}], "next": f"http://127.0.0.1:{port}/?page=2"}
        self.wfile.write(json.dumps(payload).encode())

    def log_message(self, *args):
        pass


class TestApiAdapter:
    def test_load_json_array(self):
        server = HTTPServer(("127.0.0.1", 0), _JsonHandler)
        port = server.server_address[1]
        t = Thread(target=server.handle_request, daemon=True)
        t.start()
        try:
            df = ApiAdapter.load(f"http://127.0.0.1:{port}/api")
            assert len(df) == 2
            assert list(df.columns) == ["id", "val"]
        finally:
            server.server_close()

    def test_load_paginated(self):
        server = HTTPServer(("127.0.0.1", 0), _PaginatedJsonHandler)
        port = server.server_address[1]
        # Need to handle two requests
        t = Thread(target=lambda: (server.handle_request(), server.handle_request()), daemon=True)
        t.start()
        try:
            df = ApiAdapter.load(
                f"http://127.0.0.1:{port}/api",
                pagination={"records_key": "data", "next_key": "next"},
            )
            assert len(df) == 3
            assert list(df["id"]) == [1, 2, 3]
        finally:
            server.server_close()


# ---------------------------------------------------------------------------
# ComputedAdapter
# ---------------------------------------------------------------------------

class TestComputedAdapter:
    def test_load_computed(self):
        src_a = pd.DataFrame({"x": [1, 2, 3]})
        src_b = pd.DataFrame({"x": [10, 20, 30]})

        def combine(sources):
            return pd.DataFrame({"sum": sources["a"]["x"] + sources["b"]["x"]})

        df = ComputedAdapter.load(combine, {"a": src_a, "b": src_b})
        assert list(df["sum"]) == [11, 22, 33]


# ---------------------------------------------------------------------------
# ADAPTERS dispatch dict
# ---------------------------------------------------------------------------

def test_adapters_dict_has_all_types():
    assert set(ADAPTERS.keys()) == {"file", "url", "api", "computed"}
