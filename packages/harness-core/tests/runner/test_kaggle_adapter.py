"""Tests for Kaggle adapter."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestKaggleImportGuard:
    def test_check_deps(self):
        from harnessml.core.runner.drives.kaggle import _check_deps
        _check_deps()


class TestKaggleDatasetUpload:
    @patch("harnessml.core.runner.drives.kaggle._get_api")
    def test_upload_dataset_creates_new(self, mock_get_api, tmp_path):
        from harnessml.core.runner.drives.kaggle import upload_dataset
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2")
        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        result = upload_dataset(files=[data_file], dataset_slug="testuser/test-dataset", title="Test Dataset")
        assert result["status"] == "ok"
        assert "testuser/test-dataset" in result["slug"]

    @patch("harnessml.core.runner.drives.kaggle._get_api")
    def test_upload_dataset_multiple_files(self, mock_get_api, tmp_path):
        from harnessml.core.runner.drives.kaggle import upload_dataset
        f1 = tmp_path / "train.csv"
        f2 = tmp_path / "test.csv"
        f1.write_text("a,b\n1,2")
        f2.write_text("a,b\n3,4")
        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        result = upload_dataset(files=[f1, f2], dataset_slug="testuser/multi", title="Multi File Dataset")
        assert result["status"] == "ok"


class TestKaggleNotebookUpload:
    @patch("harnessml.core.runner.drives.kaggle._get_api")
    def test_upload_notebook(self, mock_get_api, tmp_path):
        from harnessml.core.runner.drives.kaggle import upload_notebook
        nb_file = tmp_path / "notebook.ipynb"
        nb_file.write_text(json.dumps({
            "cells": [], "metadata": {"kernelspec": {"language": "python"}},
            "nbformat": 4, "nbformat_minor": 5,
        }))
        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        result = upload_notebook(notebook_path=nb_file, kernel_slug="testuser/test-kernel", title="Test Kernel")
        assert result["status"] == "ok"
        assert "testuser/test-kernel" in result["slug"]
